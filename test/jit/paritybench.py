#!/usr/bin/env python3
import ast
import inspect
import json
import logging
import os
import re
import time
import types
import zipfile
from collections import Counter, defaultdict
from typing import List

from astor import to_source
import requests

import torch
from torch.testing._internal.jit_utils import JitTestCase

log = logging.getLogger(__name__)

NN_MODULE_RE = re.compile(r"(torch\.)?(nn\.Module|autograd\.Function)\b", re.MULTILINE)
IMPORT_WHITELIST = ("torch",
                    # TODO: "torchvision",
                    "math", "collections")


class Stats(Counter):
    def __str__(self):
        stats_keys = (
            "total",
            "init_ok",
            "deduced_args_ok",
            "jit_compiles",
            "jit_success",
        )
        return str([(k, self[k]) for k in stats_keys])

    def log(self, prefix: str):
        log.info(f"{prefix}: {str(self)}")


class ErrorAggregator(object):
    def __init__(self):
        super(ErrorAggregator, self).__init__()
        self.error_groups = []
        self.bigram_to_group_ids = defaultdict(list)

    def record(self, e: Exception):
        error_msg = f"{e.__class__.__name__}: {e}"
        msg_words = list(re.findall(r"[a-zA-Z]+", error_msg))
        msg_bigrams = [f"{a}_{b}" for a, b in zip(msg_words, msg_words[1:])] or msg_words

        shared_bigrams = Counter()
        for bigram in msg_bigrams:
            shared_bigrams.update(self.bigram_to_group_ids[bigram])

        if shared_bigrams:
            best_match, count = shared_bigrams.most_common(1)[0]
            if count > len(msg_bigrams) // 2:
                self.error_groups[best_match].append(error_msg)
                return

        # No match, create a new error group
        group_id = len(self.error_groups)
        self.error_groups.append([error_msg])
        for bigram in msg_bigrams:
            self.bigram_to_group_ids[bigram].append(group_id)

    def __str__(self):
        errors = sorted(self.error_groups, key=len, reverse=True)
        return '\n'.join(f" - {len(e)} errors like: {e[0]}"
                         for e in  errors[:10])

    @classmethod
    def grouped(cls):
        return defaultdict(cls)


class PyTorchModuleExtractor(object):
    """
    Walk through a filesystem and extract all `torch.nn.Module`,
    then test if they function correctly with the JIT.
    """

    def __init__(self, errors):
        super().__init__()
        self.output_module = types.ModuleType(f"{__name__}.output")
        self.module_names = set()
        self.module_statements = []
        self.imports = dict()
        self.stats = Stats()
        self.errors = errors

    def add_module_alias(self, name: str):
        if "name" in self.output_module.__dict__:
            log.warning("Skipping module name that would clash with existing symbol: %s", name)
        self.output_module.__dict__[name] = self.output_module
        self.module_names.add(name)

    def search_file(self, filename: str, open_fn=open):
        if not filename.endswith(".py") or re.search(r"output\.py$", filename):
            return

        m = re.match(r"([a-z0-9_]+)/__init__.py$", filename, re.I)
        if m:
            self.add_module_alias(m.group(1))

        with open_fn(filename, 'r') as fp:
            source = fp.read()
            if isinstance(source, bytes):
                source = source.decode('utf-8')

        if not NN_MODULE_RE.search(source):
            return  # fast exit path

        self.add_module_alias(os.path.splitext(os.path.basename(filename))[0])

        log.debug("Searching %s", filename)
        try:
            tree = ast.parse(source, filename)
        except Exception as e:
            return self.errors["parse"].record(e)

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                bases = [to_source(x).strip() for x in node.bases]
                if any(NN_MODULE_RE.match(x) for x in bases):
                    self.module_statements.append(node)
                elif "torch" in str(bases):
                    log.warning("Maybe need to add a base: %s", bases)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                node_str = to_source(node)
                if re.match(r"(from ({0})\b)|(import ({0})\b)".format("|".join(IMPORT_WHITELIST)),
                            node_str):
                    self.imports[node_str] = node

    def search_directory(self, filename: str):
        for root, _, files in os.walk(filename, topdown=False):
            for name in files:
                self.search_file(os.path.join(root, name))

    def search_zipfile(self, filename: str):
        with zipfile.ZipFile(filename) as archive:
            for name in archive.namelist():
                self.search_file(name, archive.open)

    def main(self, filename: str):
        if os.path.isdir(filename):
            self.search_directory(filename)
        else:
            self.search_zipfile(filename)

        # self.debug_output()
        self.construct_module()
        self.test_modules()
        self.stats.log(os.path.basename(filename).replace('.zip', ''))

        return self.stats

    def construct_module(self):
        for statement in self.imports.values():
            try:
                self.add_statement(statement)
            except Exception as e:
                self.errors["import"].record(e)
        for statement in self.module_statements:
            try:
                self.add_statement(statement)
            except Exception as e:
                self.errors["define"].record(e)

    def add_statement(self, statement):
        code = compile(ast.Module([statement], []), "<string>", "exec")
        exec(code, self.output_module.__dict__, self.output_module.__dict__)

    @staticmethod
    def needed_args(signature: inspect.Signature):
        for name, param in signature.parameters.items():
            if param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
                continue  # ignore args/kwargs
            if param.default is inspect.Parameter.empty:
                yield name, param

    def instantiate_nn_module(self, nn_cls: type):
        signature = inspect.signature(nn_cls)
        kwargs = {name: self.guess_arg_value(name, param.annotation)
                  for name, param in self.needed_args(signature)}
        return nn_cls(**kwargs)

    def guess_arg_value(self, name: str, annotation=inspect.Parameter.empty):
        common_args = {
            "channels": 3,
            "num_classes": 2,
            "scale_factor": 1.0,
        }
        for search_name, placeholder_value in common_args.items():
            if search_name in name:
                return placeholder_value
        # TODO(jansel): use annotation / support guessing tensor values
        raise NotImplementedError(f"Cant guess value for: {name}")

    def test_modules(self):
        for name, value in list(self.output_module.__dict__.items()):
            if (isinstance(value, type) and
                    issubclass(value, torch.nn.Module) and
                    value.__module__ == self.output_module.__name__):
                self.test_nn_module(name, value)

    def test_nn_module(self, name: str, nn_cls: type):
        self.stats["total"] += 1

        try:
            nn_module = self.instantiate_nn_module(nn_cls)
        except Exception as e:
            return self.errors['init'].record(e)
        self.stats["init_ok"] += 1

        try:
            signature = inspect.signature(nn_module.forward)
            num_args = len(list(self.needed_args(signature)))
            arg_shapes = DeduceArgShapes(nn_module, num_args).search()
            if not arg_shapes:
                return
            args = [torch.rand(shape) for shape in arg_shapes]
        except Exception as e:
            return self.errors['deduce'].record(e)
        self.stats["deduced_args_ok"] += 1

        try:
            torch.jit.script(nn_module)
        except Exception as e:
            return self.errors['compile'].record(e)
        self.stats["jit_compiles"] += 1

        try:
            JitTestCase().checkScript(nn_module, args)
        except Exception as e:
            return self.errors['check'].record(e)

        self.stats["jit_success"] += 1
        log.info(f"CORRECT: {name}")

    def debug_output(self, filename: str = "output.py"):
        with open(filename, "w") as out:
            out.writelines(["import sys\n",
                            "_module = sys.modules[__name__]\n"])
            out.writelines([f"{module_name} = _module\n" for module_name in self.module_names])

            for import_line in self.imports.keys():
                out.write(import_line)
            out.write("\n" * 2)

            for node in self.module_statements:
                out.write(to_source(node))
                out.write("\n")


class DeduceArgShapes(object):
    """
    Try to figure out a valid input for an NN module by repeated
    guessing based on error messages.
    """
    _default_dim = 16

    def __init__(self, nn_module, num_args):
        super(DeduceArgShapes, self).__init__()
        self.nn_module = nn_module
        self.name = nn_module.__class__.__name__
        self.shape_guess = [[self._default_dim] + [n + self._default_dim] * (n + 3) for n in range(num_args)]
        self.shape_tried = [set() for _ in range(num_args)]
        self.attempt_log = []
        self.give_up = False

    def search_once(self):
        guess = list(self.shape_guess)
        args = [torch.rand(shape) for shape in guess]
        for index, shape in enumerate(guess):
            self.shape_tried[index].add(repr(shape))

        try:
            self.nn_module(*args)
            log.info(f"{self.name}: OK ({guess})")
            return guess
        except Exception as e:
            msg = str(e)
            self.attempt_log.append((guess, e.__class__.__name__, msg))

        priority_index_idea = []
        for index in range(len(guess)):
            ideas = sorted(self.ideas_from_error(guess[index], msg))
            if ideas:
                new_ideas = [(priority, idea) for priority, idea in ideas
                             if repr(idea) not in self.shape_tried[index]]
                if new_ideas:
                    ideas = new_ideas  # drop repeat ideas
                    priority_offset = 1
                else:
                    priority_offset = 0

                priority, idea = ideas.pop()
                priority_index_idea.append((priority + priority_offset, index, idea))

        if priority_index_idea:
            max_priority = max(x[0] for x in priority_index_idea)
            for priority, index, idea in priority_index_idea:
                if priority == max_priority:
                    self.shape_guess[index] = idea
        else:
            self.give_up = True

    def search(self, limit: int = 10):
        for attempt in range(limit):
            result = self.search_once()
            if result:
                return result
            if self.give_up:
                break

        log.info("---")
        for attempt in self.attempt_log:
            log.info(f"{self.name}: {attempt}")
        log.warning(f"{self.name}: GIVING UP")

    def ideas_from_error(self, guessed_shape: List[int], error_msg: str) -> List[List[int]]:
        guesses = []

        match = re.search(
            r"Given groups=(?P<groups>\d+).*(?P<weight>\[[\d, ]+\]), expected input(?P<got>\[[\d, ]+\])",
            error_msg)
        if match:
            groups = int(match.group("groups"))
            weight = ast.literal_eval(match.group("weight"))
            got = ast.literal_eval(match.group("got"))
            priority = int(got == guessed_shape)
            guesses.append((priority, self.convolution_guess(weight, groups)))

        match = re.search(
            r"Expected \d+-dimensional.*for.*(?P<weight>\[[\d, ]+\]).*got.*(?P<got>\[[\d, ]+\])",
            error_msg)
        if match:
            weight = ast.literal_eval(match.group("weight"))
            got = ast.literal_eval(match.group("got"))
            priority = int(got == guessed_shape)
            guesses.append((priority, self.convolution_guess(weight)))

        match = re.search(r"(\d+) channels, but got (\d+) channels", error_msg)
        if match:
            want = int(match.group(1))
            got = int(match.group(2))
            guess = list(guessed_shape)
            guess[1] = guess[1] * want // got
            if guess[1] > 0:
                guesses.append((-10, guess))

        match = re.search(r"same number of dimensions: got (\d+) and (\d+)",
                          error_msg)
        if match:
            a = int(match.group(1))
            b = int(match.group(2))
            if len(guessed_shape) in (a, b):
                if len(guessed_shape) == b:
                    priority = 10
                    want_len = a
                else:
                    priority = 0
                    want_len = b
                guesses.extend([
                    (priority, list(other))
                    for other in self.shape_guess
                    if len(other) == want_len])

        match = re.search(r"Got (\d+)D .*needs (\d+)D", error_msg)
        if match:
            got = int(match.group(1))
            need = int(match.group(2))
            if len(guessed_shape) == got:
                guess = [self._default_dim] * need
                guesses.append((0, guess))

        match = re.search(
            r"The size.*[(](\d+)[)] must match.*[(](\d+)[)] at.*dimension (\d+)",
            error_msg) or re.search(
            r"must match except in dimension \d+. Got (\d+) and (\d+) in dimension (\d+)",
            error_msg)
        if match:
            a = int(match.group(1))
            b = int(match.group(2))
            dim = int(match.group(3))
            if dim < len(guessed_shape):
                if guessed_shape[dim] == a:
                    priority = -20  # prefer changing B
                    a, b = b, a
                elif guessed_shape[dim] == b:
                    priority = 10
                else:
                    priority = -50

                others = ([x for x in self.shape_guess if len(x) > dim and x[dim] == a] or
                          [x for x in self.shape_guess if x != guessed_shape])
                if others:
                    guesses.extend((priority, list(other)) for other in others)

                if guessed_shape[dim] != a:
                    guess = list(guessed_shape)
                    guess[dim] = a
                    priority = int(guessed_shape[dim] == b)
                    guesses.append((priority, guess))

        return guesses

    def convolution_guess(self, weight: List[int], groups: int = 1):
        return [self._default_dim, weight[1] * groups] + [x * self._default_dim for x in weight[2:]]


class CrawlGitHub(object):
    download_dir = "../paritybench_download"

    def github_search(self):
        base = "https://api.github.com/search/repositories?per_page=100&q="
        query = "pytorch+language:Python+stars:>1000+size:<10000"
        total_count = None
        page = 1
        while True:
            time.sleep(6)  # https://developer.github.com/v3/search/#rate-limit
            rs = requests.get(f"{base}{query}&per_page=100&page={page}")
            rs.raise_for_status()
            result = rs.json()
            assert not result['incomplete_results']
            yield from result["items"]
            total_count = total_count or result['total_count']
            page += 1
            if len(result["items"]) == 0 or page > (total_count + 99) // 100:
                return

    def download_project(self, project: dict):
        name = project["full_name"]
        url = project["html_url"]
        default_branch = project["default_branch"]
        output_filename = re.sub(r"[^a-zA-Z0-9]+", "_", name) + ".zip"
        output_path = os.path.join(self.download_dir, output_filename)
        if os.path.exists(output_path):
            return output_filename
        time.sleep(10)
        rs = requests.get(f"{url}/archive/{default_branch}.zip", stream=True)
        rs.raise_for_status()
        with open(output_path, "wb") as fd:
            for chunk in rs.iter_content(chunk_size=8192):
                fd.write(chunk)
        return output_filename

    def download(self):
        metadata_path = os.path.join(self.download_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            return

        os.path.exists(self.download_dir) or os.mkdir(self.download_dir)
        projects = list(self.github_search())
        metadata = dict()
        for i, project in enumerate(projects):
            log.info(f"Downloading {project['full_name']} ({i + 1} of {len(projects)})")
            metadata[self.download_project(project)] = project
        with open(metadata_path, "w") as fd:
            json.dump(metadata, fd)

    def test_all(self):
        stats = Stats()
        error_aggregators = defaultdict(ErrorAggregator)
        for filename in os.listdir(self.download_dir):
            if filename.endswith('.zip'):
                stats_part = PyTorchModuleExtractor(error_aggregators).main(os.path.join(self.download_dir, filename))
                stats.update(stats_part)
        log.info(f"{len(error_aggregators)}")
        for name, errors in sorted(error_aggregators.items()):
            log.info(f"Top 10 errors in {name}:\n{errors}")
        stats.log('TOTAL')



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # CrawlGitHub().download()
    # PyTorchModuleExtractor().main()
    CrawlGitHub().test_all()
