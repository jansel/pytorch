#!/usr/bin/env python3
import ast
import astor
import logging
import os
import re

log = logging.getLogger(__name__)

NN_MODULE_RE = re.compile(r"(torch\.)?nn\.Module\b", re.MULTILINE)
EXTRA_INCLUDE_RE = re.compile(r"(torch\.)?autograd\.Function\b", re.MULTILINE)
IMPORT_WHITELIST = ('torch', 'torchvision', 'math', 'collections')


class PyTorchModuleExtractor(object):
    """
    Walks through a filesystem and extracts `torch.nn.Module`s
    """

    def __init__(self):
        super().__init__()
        self.module_names = set()
        self.import_lines = set()
        self.pytorch_modules = []
        self.extras = []
        self.output_symbols = set()

    def search_file(self, filename):
        if not filename.endswith(".py") or re.search(r"output\.py$", filename):
            return

        m = re.match(r'([a-z0-9_]+)/__init__.py$', filename, re.I)
        if m:
            self.module_names.add(m.group(1))

        source = open(filename).read()
        if not NN_MODULE_RE.search(source):
            return  # fast exit path

        self.module_names.add(os.path.splitext(os.path.basename(filename))[0])

        log.info("Searching %s", filename)
        tree = ast.parse(source, filename)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                bases = [astor.to_source(x).strip() for x in node.bases]
                if node.name in self.output_symbols:
                    log.warning("Skipping duplicate symbol %s", node.name)
                    continue
                if any(NN_MODULE_RE.match(x) for x in bases):
                    self.pytorch_modules.append(node)
                    self.output_symbols.add(node.name)
                elif any(EXTRA_INCLUDE_RE.match(x) for x in bases):
                    self.extras.append(node)
                    self.output_symbols.add(node.name)
                elif 'torch' in str(bases):
                    log.warning('Maybe need to add a base: %s', bases)

            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self.import_lines.add(astor.to_source(node))

    def search_directory(self, filename):
        for root, _, files in os.walk(filename, topdown=False):
            for name in files:
                self.search_file(os.path.join(root, name))

    def main(self):
        self.search_directory(".")
        log.info("Found %s modules %s imports", len(self.pytorch_modules), len(self.import_lines))
        self.write_output()

    def write_output(self, filename="output.py"):
        with open(filename, "w") as out:
            out.writelines(["import sys\n",
                            "_module = sys.modules[__name__]\n"])
            out.writelines([f"{module_name} = _module\n" for module_name in self.module_names])



            for import_line in self.import_lines:
                if re.match(r"(from ({0})\b)|(import ({0})\b)".format("|".join(IMPORT_WHITELIST)), import_line):
                    out.write(import_line)
            out.write("\n" * 2)

            for node in self.extras:
                # node = CleanupAST(self).visit(node)
                out.write(astor.to_source(node))
                out.write("\n")

            for node in self.pytorch_modules:
                # node = CleanupAST(self).visit(node)
                out.write(astor.to_source(node))
                out.write("\n")

class CleanupAST(ast.NodeTransformer):
    def __init__(self, extractor):
        super(CleanupAST, self).__init__()
        self.extractor = extractor

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr in self.extractor.output_symbols:
            log.info('Rewrote %s', node.attr)
            return ast.Name(id=node.attr, ctx=node.ctx)
        return node




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    PyTorchModuleExtractor().main()
