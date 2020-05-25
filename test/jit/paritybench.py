#!/usr/bin/env python3
import argparse
import ast
import inspect
import json
import logging
import multiprocessing
import os
import random
import re
import resource
import tempfile
import time
import types
import unittest
import zipfile
from collections import Counter, defaultdict
from typing import List

import requests
from astor import to_source

import torch

log = logging.getLogger(__name__)

NN_MODULE_RE = re.compile(r"(\btorch[.]nn\b)|(\bnn[.]Module\b)", re.MULTILINE)
IMPORT_WHITELIST = {
    # TODO: "torchvision"/"torchaudio"/etc,  is used by many
    "torch",
    "math",
    "collections",
    "numpy",
    "scipy",
    "inspect",
    "abc",
    "typing",
    "types",
    "functools",
    "itertools"
}


class PyTorchModuleExtractor(object):
    """
    Walk through a filesystem and extract all `torch.nn.Module`,
    then test if they function correctly with the JIT.
    """

    def __init__(self, tempdir: str, errors, stats):
        super(PyTorchModuleExtractor, self).__init__()
        self.tempdir = tempdir
        self.errors = errors
        self.stats = stats

        self.output_module = types.ModuleType(f"{__name__}.output")
        self.module_names = set()

        self.imports = dict()
        self.constants = []
        self.module_statements = []

        self.available_symbols = dict()

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

        overwrite = bool(NN_MODULE_RE.search(source))

        log.debug("Searching %s", filename)
        try:
            tree = ast.parse(source, filename)
        except Exception as e:
            return self.errors["parse"].record(e)

        self.add_module_alias(os.path.splitext(os.path.basename(filename))[0], overwrite)

        self.search_ast(tree, overwrite)

    def search_ast(self, tree: ast.AST, overwrite: bool):
        scope = types.ModuleType("_scope")
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                bases = [to_source(x).strip() for x in node.bases]
                if overwrite and any(self.is_torch_nn_module(scope, x) for x in bases):
                    self.module_statements.append(node)
                else:
                    self.add_available_symbol(node, overwrite)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if overwrite:
                    for module_name, import_node in self.split_import(node):
                        if module_name == "torch":
                            # Run torch imports
                            try:
                                exec(compile(ast.Module([import_node], []), "<string>", "exec"),
                                     scope.__dict__,
                                     scope.__dict__)
                            except Exception:
                                log.exception('Bad torch import')
                                continue
                        if module_name in IMPORT_WHITELIST:
                            self.imports[to_source(import_node)] = import_node

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Assign)):
                self.add_available_symbol(node, overwrite)

    @staticmethod
    def is_torch_nn_module(scope: types.ModuleType, base: str):
        if base in ('torch.nn.Module', 'nn.Module', 'Module'):
            return True
        try:
            for part in base.split('.'):
                scope = getattr(scope, part, object)
            return issubclass(scope, torch.nn.Module)
        except Exception:
            log.exception("Error in is_torch_nn_module()")

    @staticmethod
    def split_import(node):
        if isinstance(node, ast.Import):
            for name in node.names:
                tmp = ast.Import([name])
                ast.copy_location(tmp, node)
                module_name = re.sub(r"[.].*$", "", name.name)
                yield module_name, tmp
        else:
            assert isinstance(node, ast.ImportFrom)
            if node.level != 0:
                return  # not supported
            module_name = re.sub(r"[.].*$", "", node.module)
            for name in node.names:
                tmp = ast.ImportFrom(re.sub(r"^torch.legacy\b", "torch", node.module),
                                     [name],
                                     level=0)
                ast.copy_location(tmp, node)
                yield module_name, tmp

    def search_directory(self, filename: str):
        for root, _, files in os.walk(filename, topdown=False):
            for name in files:
                self.search_file(os.path.join(root, name))

    def search_zipfile(self, filename: str):
        with zipfile.ZipFile(filename) as archive:
            for name in archive.namelist():
                self.search_file(name, archive.open)

    def add_available_symbol(self, node, overwrite=False):
        try:
            if overwrite:
                self.available_symbols[node.name] = node
            else:
                self.available_symbols.setdefault(node.name, node)
        except AttributeError:
            reads, writes = ExtractReadsWrites.run(node)
            for name in writes:
                if overwrite:
                    self.available_symbols[name] = node
                else:
                    self.available_symbols.setdefault(name, node)

    def add_module_alias(self, name: str, overwrite: bool):
        if "name" in self.output_module.__dict__ and not overwrite:
            return
        self.output_module.__dict__[name] = self.output_module
        self.module_names.add(name)

    @staticmethod
    def is_literal(node):
        try:
            ast.literal_eval(node)
            return True
        except ValueError:
            return False

    def main(self, filename: str):
        if os.path.isdir(filename):
            self.search_directory(filename)
        else:
            self.search_zipfile(filename)

        # self.debug_output()
        self.construct_module()
        self.test_modules()
        basename = re.sub(r"[.]zip$", "", os.path.basename(filename))
        log.info(f"{basename}: {self.stats}")
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
                # fixed = ast.fix_missing_locations(ASTCleanup().visit(statement))
                self.add_requirements(statement)
                self.run_statement(statement, source_required=True)
            except Exception as e:
                self.errors["define"].record(e)

    def add_requirements(self, statement):
        reads, writes = ExtractReadsWrites.run(statement)
        for name in reads - writes:
            if not hasattr(self.output_module, name) and name in self.available_symbols:
                requirement = self.available_symbols.pop(name)
                self.add_requirements(requirement)
                self.run_statement(requirement, source_required=True)

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

    def instantiate_nn_module(self, nn_cls: type):
        config = MockConfig()
        signature = inspect.signature(nn_cls)
        kwargs = {name: config[name] for name, param in self.needed_args(signature)}
        return nn_cls(**kwargs)

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
            # TODO: only run once
            args = [torch.rand(shape) for shape in arg_shapes]
            python_output = nn_module(*args)
        except Exception as e:
            return self.errors['deduce'].record(e)

        self.stats["deduced_args_ok"] += 1

        try:
            # JitTestCase().checkScript(nn_module, args)  doesn't work
            script = torch.jit.script(nn_module)
            script_output = script(*args)
            self.assertEqual(script_output, python_output)
        except Exception as e:
            return self.errors['compile'].record(e)

        self.stats["jit_success"] += 1
        log.info(f"CORRECT: {name}")

    def assertEqual(self, a, b):
        # TODO(jansel): find/resuse an existing version of this
        tc = unittest.TestCase()
        if isinstance(a, torch.Tensor):
            tc.assertTrue(torch.allclose(a, b))
        elif isinstance(a, (list, tuple)):
            tc.assertEqual(len(a), len(b))
            for a_, b_ in zip(a, b):
                self.assertEqual(a_, b_)
        elif isinstance(a, dict):
            tc.assertEqual(set(a.keys()), set(b.keys()))
            for key in a.keys():
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

    @staticmethod
    def needed_args(signature: inspect.Signature):
        for name, param in signature.parameters.items():
            if param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
                continue  # ignore args/kwargs
            if param.default is inspect.Parameter.empty:
                yield name, param


class MockConfig(object):
    """
    Try to get an unknown nn.Module's init to run by guessing values
    """

    def __getitem__(self, name):
        common_args = {
            "num_modules": 1,
            "stride": 1,
            "channel": 3,
            "n_tag": 2,
            "num_classes": 2,
            "scale": 1.0,
            "class": 2,
            "layer": 1,
            "padding": 0,
            "dilation": 1,
            "gpu": False,
            "groups": 1,
            "block": 1,
            "_in": DeduceArgShapes.default_size,
            "_out": DeduceArgShapes.default_size,
            "in_": DeduceArgShapes.default_size,
            "out_": DeduceArgShapes.default_size,
            "dim": DeduceArgShapes.default_size,
            "planes": DeduceArgShapes.default_size,
            "filters": DeduceArgShapes.default_size,
            "size": DeduceArgShapes.default_size,
            "n_embd": DeduceArgShapes.default_size,
            "features": DeduceArgShapes.default_size,
            "word": DeduceArgShapes.default_size,
            "char": DeduceArgShapes.default_size,
            "width": DeduceArgShapes.default_size,
            "config": self,
            "cfg": self,
            "options": self,
            "depth": 1,
            "hidden": 1,
            "loss": torch.nn.MSELoss(),
            "dropout": 0.5,
        }
        for search_name, placeholder_value in common_args.items():
            if search_name in name:
                return placeholder_value
        raise NotImplementedError(f"{name}")

    __getattr__ = __getitem__


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


class ASTCleanup(ast.NodeTransformer):
    def visit_Import(self, node):
        return None  # Remove the node

    visit_ImportFrom = visit_Import

    def visit_Call(self, node: ast.Call):
        if getattr(node.func, 'id', '') == 'print':
            # Strip print() calls
            return ast.Constant(value=None, kind=None)
        return None


class ExtractReadsWrites(ast.NodeVisitor):
    @classmethod
    def run(cls, tree):
        visitor = cls()
        visitor.visit(tree)
        assert len(visitor.context) == 1
        return visitor.context[0]

    def __init__(self):
        super(ExtractReadsWrites, self).__init__()
        self.context = [(set(), set())]  # Read/Writs

    def visit_Global(self, node):
        global_reads, global_writes = self.context[0]
        global_reads.update(node.names)
        global_writes.update(node.names)

    def visit_Name(self, node):
        reads, writes = self.context[-1]
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            writes.add(node.id)
        else:
            assert isinstance(node.ctx, ast.Load)
            reads.add(node.id)

    def visit_Import(self, node):
        reads, writes = self.context[-1]
        for alias in node.names:
            if alias.asname:
                writes.add(alias.asname)
            else:
                writes.add(re.findall(r"[^.]+$", alias.name)[0])

    visit_ImportFrom = visit_Import

    def visit_FunctionDef(self, node):
        _, parent_writes = self.context[-1]
        try:
            parent_writes.add(node.name)
        except AttributeError:
            pass  # Lambda
        self.context.append((set(), set()))
        self.generic_visit(node)
        reads, writes = self.context.pop()
        self.context[-1][0].update(reads - writes)

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef


class Stats(Counter):
    """
    Counter used to report totals by module
    """

    def __str__(self):
        """
        Reorder key print order by stage in the process
        """
        stats_keys = [
            "total",
            "init_ok",
            "deduced_args_ok",
            "jit_success",
        ]
        stats_keys = stats_keys + list(set(self.keys()) - set(stats_keys))
        return str([(k, self[k]) for k in stats_keys])


class ErrorAggregator(object):
    """
    Collect and group error messages for report at the end
    """

    def __init__(self, context=None, log=None):
        super(ErrorAggregator, self).__init__()
        self.context = context
        self.error_groups = []
        self.bigram_to_group_ids = defaultdict(list)
        self.log = log or logging.getLogger(__name__)

    def record(self, e: Exception):
        ex_msg = str(e).strip().split('\n')[0]
        error_msg = f"{e.__class__.__name__}: {ex_msg}"
        if self._add(error_msg, [(error_msg, self.context)]):
            log.exception('New Error')

    def update(self, other):
        for errors in other.error_groups:
            self._add(errors[0][0], errors)

    def _add(self, error_msg: str, errors: List):
        msg_words = list(re.findall(r"[a-zA-Z]+", error_msg))
        msg_bigrams = [f"{a}_{b}" for a, b in zip(msg_words, msg_words[1:])] or msg_words

        shared_bigrams = Counter()
        for bigram in msg_bigrams:
            shared_bigrams.update(self.bigram_to_group_ids[bigram])

        if shared_bigrams:
            best_match, count = shared_bigrams.most_common(1)[0]
            if count > len(msg_bigrams) // 2:
                self.error_groups[best_match].extend(errors)
                return False

        # No match, create a new error group
        group_id = len(self.error_groups)
        self.error_groups.append(errors)
        for bigram in msg_bigrams:
            self.bigram_to_group_ids[bigram].append(group_id)

        return True

    @staticmethod
    def format_error_group(errors):
        context, context_count = Counter(context for msg, context in errors).most_common(1)[0]
        return f"  - {len(errors)} errors like: {errors[0][0]} ({context_count} from {context})"

    def __str__(self):
        errors = sorted(self.error_groups, key=len, reverse=True)
        return '\n'.join(map(self.format_error_group, errors[:20]))

    def __len__(self):
        return sum(map(len, self.error_groups))


class ErrorAggregatorDict(object):
    @classmethod
    def single(cls, name: str, e: Exception, context=None):
        errors = cls(context)
        errors[name].record(e)
        return errors

    def __init__(self, context=None):
        super(ErrorAggregatorDict, self).__init__()
        self.aggregator = dict()
        self.context = context
        if context:
            self.name = re.sub(r"[.]zip$", "", os.path.basename(context))
        else:
            self.name = __name__

    def __getitem__(self, item):
        if item not in self.aggregator:
            self.aggregator[item] = ErrorAggregator(self.context, logging.getLogger(f"{item}.{self.name}"))
        return self.aggregator[item]

    def update(self, other):
        for key, value in other.aggregator.items():
            self[key].update(other=value)

    def print_report(self):
        for name in sorted(list(self.aggregator.keys())):
            self[name].log.info(f"\nTop errors in {name} ({len(self[name])} total):\n{self[name]}\n")


class CrawlGitHub(object):
    """
    Download projects from github with 100+ stars and the word "pytorch"
    """

    def __init__(self, download_dir):
        super(CrawlGitHub, self).__init__()
        self.download_dir = download_dir

    def github_search(self):
        base = "https://api.github.com/search/repositories?per_page=100&sort=stars"
        query = "pytorch+language:Python+stars:>100+size:<10000"
        seen = set()
        # both orders gets us 20 pages (past 10 limit), need 12 for current query
        for order in ("desc", "asc"):
            page = 1
            while True:
                time.sleep(6)  # https://developer.github.com/v3/search/#rate-limit
                rs = requests.get(f"{base}&page={page}&order={order}&q={query}")
                rs.raise_for_status()
                result = rs.json()
                assert not result['incomplete_results']
                for project in result["items"]:
                    name = project["full_name"]
                    if name not in seen:
                        seen.add(name)
                        yield project
                total_count = result['total_count']
                log.info(f"total_count={total_count} seen={len(seen)} page={page} {order}")
                page += 1
                if len(result["items"]) == 0 or len(seen) >= total_count:
                    return
                if page == 11:
                    break  # not allowed by API

    def download_project(self, project: dict):
        name = project["full_name"]
        url = project["html_url"]
        default_branch = project["default_branch"]
        output_filename = re.sub(r"[^a-zA-Z0-9]+", "_", name) + ".zip"
        output_path = os.path.join(self.download_dir, output_filename)
        if os.path.exists(output_path):
            return output_filename
        time.sleep(60)
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


def test_all(download_dir, limit=None):
    limit = limit or float('inf')
    stats = Stats()
    errors = ErrorAggregatorDict()
    for filename in os.listdir(download_dir):
        if filename.endswith('.zip'):
            errors_part, stats_part = test_zipfile(os.path.join(download_dir, filename))
            errors.update(errors_part)
            stats.update(stats_part)
            limit -= 1
            if limit == 0:
                break
    errors.print_report()
    log.info(f"TOTAL: {stats}")


def test_zipfile(path):
    parent_conn, child_conn = multiprocessing.Pipe()
    start = time.time()

    with tempfile.TemporaryDirectory(prefix="paritybench") as tempdir:
        proc = multiprocessing.Process(target=test_zipfile_subproc, args=(tempdir, path, child_conn))
        proc.start()
        while proc.is_alive():
            if parent_conn.poll(1):
                result = parent_conn.recv()
                proc.join()
                return result
            if time.time() - start > 60:
                proc.terminate()
                proc.join(10)
                return ErrorAggregatorDict.single(
                    "meta",
                    TimeoutError("Timeout testing module"),
                    path
                ), Stats({"timeout": 1})

        proc.join()
        if proc.exitcode == 0:
            return parent_conn.recv()
        else:
            return ErrorAggregatorDict.single(
                "meta",
                MemoryError("Crash testing module"),
                path
            ), Stats({"crash": 1})


def test_zipfile_subproc(tempdir: str, path: str, return_pipe: multiprocessing.Pipe):
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (10 * 1024 ** 3, hard))

    errors = ErrorAggregatorDict(path)
    stats = Stats()
    extractor = PyTorchModuleExtractor(tempdir, errors, stats)
    extractor.main(path)
    return_pipe.send((errors, stats))


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--download', action='store_true')
    group.add_argument('--run')
    parser.add_argument('--directory')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()

    if args.download:
        download_dir = args.directory or "../paritybench_download_100star"
        CrawlGitHub(download_dir).download()
        return

    if args.run:
        assert os.path.isfile(args.run)
        errors, stats = test_zipfile(args.run)
        errors.print_report()
        log.info(f"Stats: {stats}")
        return

    # Run them all
    download_dir = args.directory or "../paritybench_download_1000star"
    test_all(download_dir, args.limit)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
