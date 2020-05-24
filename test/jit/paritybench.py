#!/usr/bin/env python3
import ast
import inspect
import json
import logging
import os
import random
import re
import tempfile
import time
import types
import unittest
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
                    # TODO: "torchvision",  is used by many
                    "math", "collections")


class Stats(Counter):
    """
    Counter used to report totals by module
    """

    def __str__(self):
        stats_keys = [
            "total",
            "init_ok",
            "deduced_args_ok",
            "jit_success",
        ]
        stats_keys = stats_keys + list(set(self.keys()) - set(stats_keys))
        return str([(k, self[k]) for k in stats_keys])

    def log(self, prefix: str):
        log.info(f"{prefix}: {str(self)}")


class ErrorAggregator(object):
    """
    Collect and group error messages for report at the end
    """

    def __init__(self):
        super(ErrorAggregator, self).__init__()
        self.error_groups = []
        self.bigram_to_group_ids = defaultdict(list)

    def record(self, e: Exception):
        ex_msg = str(e).strip().split('\n')[0]
        error_msg = f"{e.__class__.__name__}: {ex_msg}"
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
        return '\n'.join(f"  - {len(e)} errors like: {e[0]}"
                         for e in errors[:10])

    def __len__(self):
        return sum(map(len, self.error_groups))


class ErrorAggregatorDict(defaultdict):
    def __init__(self):
        super(ErrorAggregatorDict, self).__init__(ErrorAggregator)

    def log(self):
        for name in sorted(list(self.keys())):
            log.info(f"Top 10 errors in {name} ({len(self[name])} total):\n{self[name]}")


class PyTorchModuleExtractor(object):
    """
    Walk through a filesystem and extract all `torch.nn.Module`,
    then test if they function correctly with the JIT.
    """

    def __init__(self, errors=None, stats=None):
        super().__init__()
        if errors is None:
            errors = ErrorAggregatorDict()
        if stats is None:
            stats = Stats()
        self.stats = stats
        self.errors = errors

        self.output_module = types.ModuleType(f"{__name__}.output")
        self._tempdir = tempfile.TemporaryDirectory(prefix="paritybench")
        self.module_names = set()

        self.imports = dict()
        self.constants = []
        self.module_statements = []

    def __enter__(self):
        self.tempdir = self._tempdir.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._tempdir.__exit__(exc_type, exc_val, exc_tb)

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

        self.search_ast(tree)

    def search_ast(self, tree: ast.AST):
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
            elif isinstance(node, ast.Assign) and self.is_literal(node.value):
                self.constants.append(node)

    @staticmethod
    def is_literal(node):
        try:
            ast.literal_eval(node)
            return True
        except ValueError:
            return False

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
                self.run_statement(statement)
            except Exception as e:
                self.errors["import"].record(e)
        for statement in self.constants:
            try:
                self.run_statement(statement)
            except Exception as e:
                self.errors["constant"].record(e)
        for statement in self.module_statements:
            try:
                fixed = ast.fix_missing_locations(ASTCleanup().visit(statement))
                self.run_statement(fixed, source_required=True)
            except Exception as e:
                self.errors["define"].record(e)

    def run_statement(self, statement, source_required=False):
        if not source_required:
            code = compile(ast.Module([statement], []), "<string>", "exec")
        else:
            # TorchScript requires source code to exist on disk
            assert self.tempdir
            fn, filename = tempfile.mkstemp(suffix='.py', dir=self.tempdir, )
            with os.fdopen(fn, "w") as fd:
                fd.write(to_source(statement))
                fd.flush()
            code = compile(open(filename).read(), filename, "exec")
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
        # TODO(jansel): use annotation / support guessing tensor values
        return MockConfig()[name]

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

        signature = inspect.signature(nn_module.forward)
        try:
            arg_shapes = DeduceArgShapes.search_retry(nn_module, list(self.needed_args(signature)))
            args = [torch.rand(shape) for shape in arg_shapes]
            python_output = nn_module(*args)
        except Exception as e:
            return self.errors['deduce'].record(e)
        self.stats["deduced_args_ok"] += 1

        try:
            # JitTestCase().checkScript(nn_module, args)
            script = torch.jit.script(nn_module)
            script_output = script(*args)
            self.assertEqual(script_output, python_output)
        except Exception as e:
            return self.errors['compile'].record(e)

        self.stats["jit_success"] += 1
        log.info(f"CORRECT: {name}")

    def assertEqual(self, a, b):
        tc = unittest.TestCase()
        if isinstance(a, torch.Tensor):
            tc.assertTrue(torch.allclose(a, b))
        elif isinstance(a, (list, tuple)):
            tc.assertEqual(len(a), len(b))
            for a_, b_ in zip(a, b):
                self.assertEqual(a_, b_)
        elif isinstance(a, dict):
            keys = set(a.keys())
            tc.assertEqual(keys, set(b.keys()))
            for key in keys:
                self.assertEqual(a[key], b[key])
        else:
            tc.assertEqual(a, b)

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


class MockConfig(object):
    """
    Try to get an unknown nn.Module's init to run by guessing values
    """
    def __getitem__(self, name):
        common_args = {
            "channels": 3,
            "num_classes": 2,
            "scale_factor": 1.0,
            "nclass": 2,
            "dim": DeduceArgShapes.default_size,
            "planes": DeduceArgShapes.default_size,
            "size": DeduceArgShapes.default_size,
            "config": self,
            "cfg": self,
        }
        for search_name, placeholder_value in common_args.items():
            if search_name in name:
                return placeholder_value
        raise NotImplementedError(f"{name}")


class ASTCleanup(ast.NodeTransformer):
    pass


class DeduceArgShapes(object):
    """
    Try to figure out a valid input for an NN module by repeated
    guessing based on error messages.
    """
    default_size = 8

    def __init__(self, nn_module, initial_guess):
        super(DeduceArgShapes, self).__init__()
        self.nn_module = nn_module
        self.name = nn_module.__class__.__name__
        self.shape_guess = initial_guess
        self.shape_tried = [set() for _ in range(len(initial_guess))]
        self.attempt_log = []
        self.give_up = False
        self.last_error = None

    @classmethod
    def search_retry(cls, nn_module, needed_args):
        num_args = len(needed_args)
        deducers = [cls(nn_module, initial_guess)
                    for initial_guess in (
                        # cls.initial_guess1(num_args),
                        cls.initial_guess2(num_args),
                    )]
        for deducer in deducers:
            shape = deducer.search()
            if shape:
                return shape
        raise random.choice(deducers).last_error

    @classmethod
    def initial_guess1(cls, num_args):
        return [[cls.default_size] * 3 for _ in range(num_args)]

    @classmethod
    def initial_guess2(cls, num_args):
        return [[cls.default_size] + [n + cls.default_size] * (n + 3) for n in range(num_args)]

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
            self.last_error = e
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
                guess = [self.default_size] * need
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
        return [self.default_size, weight[1] * groups] + [x * self.default_size for x in weight[2:]]


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
        error_aggregators = ErrorAggregatorDict()
        for filename in os.listdir(self.download_dir):
            if filename.endswith('.zip'):
                with PyTorchModuleExtractor(error_aggregators) as extractor:
                    extractor.main(os.path.join(self.download_dir, filename))
                    stats.update(extractor.stats)
        error_aggregators.log()
        stats.log('TOTAL')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # CrawlGitHub().download()
    # PyTorchModuleExtractor().main()
    CrawlGitHub().test_all()
