#!/usr/bin/env python3
from collections import Counter, defaultdict
from functools import reduce
from typing import List, Callable
import abc
import argparse
import ast
import inspect
import itertools
import json
import logging
import multiprocessing
import os
import re
import resource
import sys
import tempfile
import time
import traceback
import types
import unittest
import zipfile

import requests
from astor import to_source

import torch

log = logging.getLogger(__name__)

NN_MODULE_RE = re.compile(r"(\btorch[.]nn\b)|(\bnn[.]Module\b)", re.MULTILINE)
IMPORT_WHITELIST = {
    # TODO: torchvision/torchaudio/etc is used by many
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
    "itertools",
    "logging",
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

    def test_modules(self):
        for name, value in list(self.output_module.__dict__.items()):
            if (isinstance(value, type) and
                    issubclass(value, torch.nn.Module) and
                    value.__module__ == self.output_module.__name__):
                self.test_nn_module(name, value)

    def test_nn_module(self, name: str, nn_cls: type):
        self.stats["total"] += 1

        init_signature = inspect.signature(nn_cls)
        try:
            init_deducer = DeduceParameters(
                nn_cls,
                *DeduceParameters.initial_args_init(init_signature))
            init_deducer.search()
            nn_module = init_deducer.last_result
        except Exception as e:
            return self.errors['init'].record(e)

        self.stats["init_ok"] += 1

        forward_signature = inspect.signature(nn_module.forward)
        try:
            forward_deducer = DeduceParameters(
                nn_module,
                *DeduceParameters.initial_args_forward(forward_signature))
            forward_deducer.search()
            args = forward_deducer.last_args
            kwargs = forward_deducer.last_kwargs
            python_output = forward_deducer.last_result
        except Exception as e:
            return self.errors['deduce'].record(e)

        self.stats["deduced_args_ok"] += 1

        try:
            # JitTestCase().checkScript(nn_module, args)  doesn't work
            script = torch.jit.script(nn_module)
            script_output = script(*args, **kwargs)
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


class DeductionFailed(RuntimeError):
    def __init__(self, attempt_log, name=''):
        super().__init__(
            f"{attempt_log[-1][1]}\n{name}:\n" +
            "\n".join(f" - {attempt}" for attempt in attempt_log)
        )


class DeduceParameters(object):
    """
    Try to figure out a valid input for an NN module by repeated
    guessing based on error messages.
    """
    default_size = 4

    @staticmethod
    def needed_args(signature: inspect.Signature):
        for name, param in signature.parameters.items():
            if param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
                continue  # ignore args/kwargs
            if param.default is inspect.Parameter.empty:
                yield param

    @classmethod
    def initial_args_init(cls, signature: inspect.Signature = None):
        return [], {param.name: DeduceParameter.initial_arg_init(param, position)
                    for position, param in enumerate(cls.needed_args(signature))}

    @classmethod
    def initial_args_forward(cls, signature: inspect.Signature):
        return [DeduceParameter.initial_arg_forward(param, position)
                for position, param in enumerate(cls.needed_args(signature))], {}

    def __init__(self, nn_module: Callable, args: list, kwargs: dict):
        super(DeduceParameters, self).__init__()
        self.nn_module = nn_module
        self.args = args
        self.kwargs = kwargs
        self.tried = set()

        self.attempt_log = []
        self.last_args = None
        self.last_kwargs = None
        self.last_result = None

    def __str__(self):
        return ", ".join(itertools.chain(
            map(str, self.args),
            [f"{name}={arg}" for name, arg in self.kwargs.items()]))

    @classmethod
    def run(cls, nn_module: Callable, needed_args: List[inspect.Parameter]):
        return DeduceParameters(nn_module, needed_args).search()

    def search_once(self):
        self.last_args = [arg.guess() for arg in self.args]
        self.last_kwargs = {name: arg.guess() for name, arg in self.kwargs.items()}
        guess_str = str(self)
        self.tried.add(guess_str)

        try:
            self.last_result = self.nn_module(*self.last_args, **self.last_kwargs)
            return True
        except Exception:
            error_type, error_value, tb = sys.exc_info()
            error_msg = f"{error_type.__name__}: {error_value}"
            sorted_args = self.sorted_args(tb)

        self.attempt_log.append((guess_str, error_msg))

        if Guess.apply_fixors(self.get_fixors(), error_msg):
            if str(self) not in self.tried:
                return False

        for pass_number in (0, 1):
            for arg in sorted_args:
                if arg.try_to_fix(error_msg, pass_number):
                    if str(self) not in self.tried:
                        return False
                    arg.rollback()

        raise DeductionFailed(self.attempt_log, str(type(self.nn_module)))

    def all_args(self):
        return list(self.args) + list(self.kwargs.values())

    def sorted_args(self, trackback) -> List:
        """
        Order args by when they are seen in the traceback so we can fix
        relevant to the error args first.

        :param trackback: from sys.exc_info()
        :return: parameters ordered by where they are seen in the traceback
        """
        this_file = os.path.basename(__file__)
        args = self.all_args()
        for frame in reversed(traceback.extract_tb(trackback, limit=-10)):
            if this_file in frame.filename:
                break
            line = frame.line
            args.sort(key=lambda x: x.contained_in_line(line), reverse=True)
        return args

    def search(self, limit: int = 10):
        for attempt in range(limit):
            if self.search_once():
                return self.last_result
        raise DeductionFailed(self.attempt_log, str(type(self.nn_module)))

    def get_fixors(self):
        return [
            (r"missing.*argument: (?P<name>['][a-zA-Z0-9_]+['])",
             self.fix_missing_arg),
            (r"unexpected keyword argument (?P<name>['][a-zA-Z0-9_]+['])",
             self.fix_extra_arg),
            (r"size mismatch, m1: (?P<a>\[.*\]), m2: (?P<b>\[.*\])",
             self.fix_size_mismatch),

        ]

    def fix_size_mismatch(self, a, b):
        matches_a = [arg for arg in self.all_args() if arg.is_shape_match(a)]
        matches_b = [arg for arg in self.all_args() if arg.is_shape_match(b)]
        if not matches_a and not matches_b:
            matches_a = [arg for arg in self.all_args() if arg.is_element_count_match(a)]
            matches_b = [arg for arg in self.all_args() if arg.is_element_count_match(b)]

        if matches_a and matches_b:
            if max(x.created for x in matches_a) > max(x.created for x in matches_b):
                # prefer changing the old one
                matches_a, matches_b = matches_b, matches_a
            guess_a = min(matches_a, key=lambda x: x.created)
            guess_b = max(matches_b, key=lambda x: x.created)
            guess_a.change_guess(guess_b._guesses[-1].clone())
            return True

        if matches_b:
            matches_a, matches_b = matches_b, matches_a
            a, b = b, a

        if matches_a:
            guess = min(matches_a, key=lambda x: x.created)
            guess.change_guess(TensorGuess(shape=b))
            return True

    def fix_missing_arg(self, name):
        if any(arg.name == name for arg in self.args):
            return
        if name in self.kwargs:
            # try moving it to args?
            self.args.append(self.kwargs.pop(name))
            self.args.sort(key=lambda x: x.position)
        else:
            self.kwargs[name] = DeduceParameter.initial_arg_init(name, float('inf'))
        return True

    def fix_extra_arg(self, name):
        if name in self.kwargs:
            del self.kwargs[name]
            return True


class DeduceParameter(object):
    @classmethod
    def initial_arg_init(cls, name, position):
        name = getattr(name, 'name', name)
        if 'dataset' in name:
            # TODO: this likely wont work...
            return cls.initial_arg_forward(name, position)

        common_args = {
            "stride": 1,
            "scale": 1.0,
            "layer": 1,
            "padding": 0,
            "dilation": 1,
            "groups": 1,
            "block": 1,
            "depth": 1,
            "hidden": 1,
            "gpu": False,
            "loss": torch.nn.MSELoss(),
            "dropout": 0.5,
        }
        for search_name, placeholder_value in common_args.items():
            if search_name in name:
                return cls(name, position, LiteralGuess(placeholder_value))

        for search_name in ('cfg', 'config', 'options'):
            if search_name in name:
                return cls(name, position, ConfigGuess())

        return cls(name, position, LiteralGuess(DeduceParameters.default_size))

    @property
    def created(self):
        return self._guesses[-1].created

    @classmethod
    def initial_arg_forward(cls, name, position=None):
        name = getattr(name, 'name', name)
        return cls(name, position, TensorGuess([TensorGuess.default_size] * 4))

    def __init__(self, name: str, position, initial_guess):
        super(DeduceParameter, self).__init__()
        self.name = name
        self.position = position
        self._guesses = [initial_guess]

    def guess(self):
        return self._guesses[-1].guess()

    def try_to_fix(self, error_message: str, pass_number: int) -> bool:
        new_guess = self._guesses[-1].get_fix(error_message, pass_number)
        if new_guess is not None:
            self.change_guess(new_guess)
            return True
        return False

    def change_guess(self, guess):
        self._guesses.append(guess)

    def contained_in_line(self, line: str):
        pattern = r"\b{}\b".format(re.escape(self.name))
        return bool(re.search(pattern, line))

    def rollback(self):
        self._guesses.pop()

    def __str__(self):
        return str(self._guesses[-1])

    def is_shape_match(self, shape):
        if isinstance(self._guesses[-1], TensorGuess):
            return self._guesses[-1].shape == shape

    def is_element_count_match(self, shape):
        if isinstance(self._guesses[-1], TensorGuess):
            count = reduce(lambda a, b: a * b, shape)
            this_count = reduce(lambda a, b: a * b, self._guesses[-1].shape)
            return count == this_count


class Guess(object):
    def __init__(self, value=None):
        super(Guess, self).__init__()
        self.value = value
        self.created = time.time()

    @staticmethod
    def apply_fixors(fixors, error_msg):
        for pattern, fixor in fixors:
            match = re.search(pattern, error_msg, flags=re.I)
            if match:
                fix = fixor(**{k: Guess.literal(v) for k, v in match.groupdict().items()})
                if fix is not None:
                    log.debug(f"FIX: {fixor.__name__} {error_msg}")
                    return fix

    @staticmethod
    def literal(value):
        return ast.literal_eval(value.replace(" x ", ","))

    def guess(self):
        return self.value

    @abc.abstractmethod
    def get_fix(self, error_message: str, pass_number: int):
        raise NotImplementedError()

    def __str__(self):
        return str(self.value)

    def get_fix(self, error_message: str, pass_number: int):
        if pass_number > 0:
            return

        def fix_unpack_int():
            if isinstance(self.value, int):
                return [TensorGuess.default_size] * 2

        def fix_too_many(want):
            try:
                if len(self.value) > want:
                    return [TensorGuess.default_size] * want
            except TypeError:
                pass

        def fix_too_few(want, got):
            try:
                if len(self.value) == got:
                    return [TensorGuess.default_size] * want
            except TypeError:
                pass

        fixors = [
            (r"TypeError: cannot unpack non-iterable int object", fix_unpack_int),
            (r"ValueError: too many values to unpack \(expected (?P<want>\d+)\)", fix_too_many),
            (r"ValueError: not enough values to unpack \(expected (?P<want>\d+), got (?P<got>\d+)\)", fix_too_few),
        ]

        new_value = self.apply_fixors(fixors, error_message)
        if new_value is not None:
            return LiteralGuess(new_value)


class LiteralGuess(Guess):
    def get_fix(self, error_message: str, pass_number: int):
        fix = super(LiteralGuess, self).get_fix(error_message, pass_number)
        if fix:
            return fix

        if pass_number > 0:
            return

        fixors = [
            ("TypeError: (?P<typename>'[^']+') object is not subscriptable",
             self.fix_not_subscriptable),
        ]

        new_value = self.apply_fixors(fixors, error_message)
        if new_value is not None:
            return LiteralGuess(new_value)

    def fix_not_subscriptable(self, typename):
        if typename == 'int' and isinstance(self.value, int):
            return [TensorGuess.default_size] * 2


class ConfigGuess(Guess):
    def __init__(self):
        super(ConfigGuess, self).__init__(value=MockConfig())

    def get_fix(self, error_message: str, pass_number: int):
        pass


class TensorGuess(Guess):
    default_size = DeduceParameters.default_size

    def __init__(self, shape, dtype=torch.float32):
        super(TensorGuess, self).__init__()
        assert isinstance(shape, list)
        assert all(isinstance(x, int) for x in shape)
        self.shape = shape
        self.dtype = dtype
        self.value = torch.randn(self.shape, dtype=self.dtype)

    def clone(self):
        return self.__class__(self.shape, self.dtype)

    def __str__(self):
        return f"shape({self.shape})"

    def get_fix(self, error_message: str, pass_number: int):
        fix = super(TensorGuess, self).get_fix(error_message, pass_number)
        if fix:
            return fix

        new_shape = self.apply_fixors(self.shape_fixors(pass_number), error_message)
        if new_shape is not None:
            return self.__class__(new_shape, self.dtype)

    def shape_fixors(self, pass_number: int):
        if pass_number == 0:
            return [
                (r"Given groups=(?P<groups>\d+).*(?P<weight>\[[\d, ]+\]), expected input(?P<got>\[[\d, ]+\])",
                 self.fix_convolution_if_matching),
                (r"Expected \d+-dimensional.*for.*(?P<weight>\[[\d, ]+\]).*got.*(?P<got>\[[\d, ]+\])",
                 self.fix_convolution_if_matching),
                (r"(?P<want>\d+) channels, but got (?P<got>\d+) channels",
                 self.fix_num_channels),
                (r"same number of dimensions: got (?P<want>\d+) and (?P<got>\d+)",
                 self.fix_dimensions),
                (r"Got (?P<got>\d+)D .*needs (?P<want>\d+)D",
                 self.fix_dimensions),
                (r"input must have (?P<want>\d+) dimensions, got (?P<got>\d+)",
                 self.fix_dimensions),
                (r"The size.*[(](?P<want>\d+)[)] must match.*[(](?P<got>\d+)[)] at.*dimension (?P<dim>\d+)",
                 self.fix_dimensions_at),
                (r"must match except in dimension \d+. Got (?P<want>\d+) and (?P<got>\d+) in dimension (?P<dim>\d+)",
                 self.fix_dimensions_at),
                (r"matrices expected, got (?P<got>\d+)D, (?P<want>\d+)D ",
                 self.fix_dimensions),
                (r"expected \d+D or (?P<want>\d+)D input \(got (?P<got>\d+)D input\)",
                 self.fix_dimensions),
                (r"Expected.*size (?P<want>[\d, ()]+), got (?P<got>[\d, ()]+)",
                 self.fix_shape),
            ]
        if pass_number == 1:
            return [
                (r"Given groups=(?P<groups>\d+).*(?P<weight>\[[\d, ]+\]), expected input\[[\d, ]+\]",
                 self.fix_convolution),
                (r"Expected \d+-dimensional.*for.*(?P<weight>\[[\d, ]+\]).*got.*\[[\d, ]+\]",
                 self.fix_convolution),
                (r"same number of dimensions: got (?P<got>\d+) and (?P<want>\d+)",
                 self.fix_dimensions),
                (r"Got \d+D .*needs (?P<want>\d+)D",
                 self.fix_dimensions),
                (r"The size.*[(](?P<got>\d+)[)] must match.*[(](?P<want>\d+)[)] at.*dimension (?P<dim>\d+)",
                 self.fix_dimensions_at),
                (r"must match except in dimension \d+. Got (?P<got>\d+) and (?P<want>\d+) in dimension (?P<dim>\d+)",
                 self.fix_dimensions_at),
                (r"expected (?P<want>\d+)D or \d+D input \(got (?P<got>\d+)D input\)",
                 self.fix_dimensions),
            ]

    def fix_shape(self, want, got):
        if self.shape == list(got):
            return list(want)

    def fix_convolution(self, weight: List[int], groups: int = 1):
        return [self.default_size, weight[1] * groups] + [x * self.default_size for x in weight[2:]]

    def fix_convolution_if_matching(self, weight, got, groups=1):
        if got == self.shape:
            return self.fix_convolution(weight, groups)

    def fix_num_channels(self, want, got):
        guess = list(self.shape)
        if len(guess) > 1:
            guess[1] = guess[1] * want // got
            if guess[1] > 0:
                return guess

    def fix_dimensions(self, want, got=None):
        shape = list(self.shape)
        if got is None or len(shape) == got:
            shape.extend([self.default_size] * want)
            return shape[:want]

    def fix_dimensions_at(self, want, got, dim):
        shape = list(self.shape)
        if dim < len(shape) and shape[dim] == got:
            shape[dim] = want
            return shape


class MockConfig(object):
    def __init__(self):
        super(MockConfig, self).__init__()
        self._guesses = dict()
        self._values = dict()

    def __getitem__(self, item):
        if item not in self._guesses:
            self._guesses[item] = DeduceParameter.initial_arg_init(name=item, position=None)
        return self._values[item].guess()

    __getattr__ = __getitem__


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
    group.add_argument("--download", action="store_true")
    group.add_argument("--run")
    parser.add_argument("--download-dir", "-d", default="../paritybench_download")
    parser.add_argument("--limit", "-l", type=int)
    args = parser.parse_args()

    if args.download:
        CrawlGitHub(args.download_dir).download()
        return

    if args.run:
        assert os.path.isfile(args.run)
        errors, stats = test_zipfile(args.run)
        errors.print_report()
        log.info(f"Stats: {stats}")
        return

    test_all(args.download_dir, args.limit)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
