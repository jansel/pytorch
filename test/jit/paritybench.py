#!/usr/bin/env python3
import argparse
import ast
import astor
import inspect
import logging
import os
import re
import torch
import types

from itertools import chain
from typing import List

log = logging.getLogger(__name__)

NN_MODULE_RE = re.compile(r"(torch\.)?(nn\.Module|autograd\.Function)\b", re.MULTILINE)
IMPORT_WHITELIST = ("torch",
                    # TODO: "torchvision",
                    "math", "collections")


class PyTorchModuleExtractor(object):
    """
    Walk through a filesystem and extract all `torch.nn.Module`,
    then test if they function correctly with the JIT.
    """

    def __init__(self):
        super().__init__()
        self.output_module = types.ModuleType(f"{__name__}.output")
        self.module_names = set()
        self.module_statements = []
        self.imports = dict()

    def add_module_alias(self, name: str):
        if "name" in self.output_module.__dict__:
            log.warning("Skipping module name that would clash with existing symbol: %s", name)
        self.output_module.__dict__[name] = self.output_module
        self.module_names.add(name)

    def search_file(self, filename: str):
        if not filename.endswith(".py") or re.search(r"output\.py$", filename):
            return

        m = re.match(r"([a-z0-9_]+)/__init__.py$", filename, re.I)
        if m:
            self.add_module_alias(m.group(1))

        source = open(filename).read()

        if not NN_MODULE_RE.search(source):
            return  # fast exit path

        self.add_module_alias(os.path.splitext(os.path.basename(filename))[0])

        log.info("Searching %s", filename)
        tree = ast.parse(source, filename)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                bases = [astor.to_source(x).strip() for x in node.bases]
                if any(NN_MODULE_RE.match(x) for x in bases):
                    self.module_statements.append(node)
                elif "torch" in str(bases):
                    log.warning("Maybe need to add a base: %s", bases)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                node_str = astor.to_source(node)
                if re.match(r"(from ({0})\b)|(import ({0})\b)".format("|".join(IMPORT_WHITELIST)),
                            node_str):
                    self.imports[node_str] = node

    def search_directory(self, filename: str):
        for root, _, files in os.walk(filename, topdown=False):
            for name in files:
                self.search_file(os.path.join(root, name))

    def main(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("search_dir")
        args = parser.parse_args()
        self.search_directory(args.search_dir)

        # self.debug_output()
        self.construct_module()
        self.test_modules()

    def construct_module(self):
        code = compile(ast.Module(list(chain(self.imports.values(), self.module_statements))),
                       "<string>", "exec")
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
        for name, value in self.output_module.__dict__.items():
            if isinstance(value, type) and issubclass(value, torch.nn.Module):
                self.test_nn_module(name, value)

    def test_nn_module(self, name: str, nn_cls: type):
        try:
            nn_module = self.instantiate_nn_module(nn_cls)
        except Exception as e:
            log.warning(f"{name} {e.__class__.__name__}: {e}", exc_info=False)
            return

        try:
            signature = inspect.signature(nn_module.forward)
            num_args = len(list(self.needed_args(signature)))
            DeduceArgShapes(nn_module, num_args).search()
        except Exception as e:
            log.warning(f"{name} {e.__class__.__name__}: {e}", exc_info=True)

    def debug_output(self, filename: str = "output.py"):
        with open(filename, "w") as out:
            out.writelines(["import sys\n",
                            "_module = sys.modules[__name__]\n"])
            out.writelines([f"{module_name} = _module\n" for module_name in self.module_names])

            for import_line in self.imports.keys():
                out.write(import_line)
            out.write("\n" * 2)

            for node in self.module_statements:
                out.write(astor.to_source(node))
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    PyTorchModuleExtractor().main()
