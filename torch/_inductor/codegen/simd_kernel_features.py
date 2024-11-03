from __future__ import annotations

import collections
import dataclasses
import functools
import itertools
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Type, Union

import sympy

import torch

from ...utils._ordered_set import OrderedSet
from ...utils._sympy.functions import FloorDiv, ModularIndexing
from ..dependencies import Dep, MemoryDep
from ..runtime.hints import ReductionHint
from ..scheduler import SchedulerNode
from ..utils import cache_on_self, sympy_subs
from ..virtualized import V


class NodeScheduleMarker:
    @staticmethod
    def only_nodes(it: Iterable[NodeScheduleEntry]) -> Iterable[SchedulerNode]:
        for item in it:
            if not (item is DisableReduction or item is EnableReduction):
                yield item  # type: ignore[misc]

    @staticmethod
    def is_reduction() -> bool:
        return False


NodeScheduleEntry = Union[SchedulerNode, Type[NodeScheduleMarker]]


class DisableReduction(NodeScheduleMarker):
    """
    Marker to invoke `kernel.disable_reduction()`.  This closes a
    reduction loop and allows for pointwise ops to occur on the output
    of a reduction.
    """


class EnableReduction(NodeScheduleMarker):
    """
    Marker to end a DisableReduction block.
    """

    @staticmethod
    def filter(node_schedule: List[NodeScheduleEntry]) -> Iterable[SchedulerNode]:
        """
        Get the nodes from node_schedule skipping those in a
        DisableReduction block.
        """
        disabled = False
        for node in node_schedule:
            if node in (EnableReduction, DisableReduction):
                # Don't tile stuff outside the main reduction loop
                disabled = node is DisableReduction
            elif disabled:
                pass
            else:
                yield node  # type: ignore[misc]


class SIMDKernelFeatures:
    """
    An ordered schedule of nodes that will become a single kernel.
    """

    def __init__(
        self,
        node_schedule: List[NodeScheduleEntry],
        numel: sympy.Expr,
        reduction_numel: sympy.Expr = sympy.S.One,
    ):
        self.node_schedule = node_schedule
        self.numel = V.graph.sizevars.simplify(numel)  # numel excludes reduction_numel
        self.reduction_numel = V.graph.sizevars.simplify(reduction_numel)

    @cache_on_self
    def is_reduction(self) -> bool:
        return self.reduction_numel != 1

    @cache_on_self
    def scheduler_nodes(self) -> Iterable[SchedulerNode]:
        return tuple(NodeScheduleMarker.only_nodes(self.node_schedule))

    def reduction_nodes(self) -> List[SchedulerNode]:
        return [n for n in self.scheduler_nodes() if n.is_reduction()]

    @cache_on_self
    def buf_accesses(self) -> Dict[str, List[Dep]]:
        """only needed for config.benchmark_kernel"""
        buf_accesses = collections.defaultdict(list)
        for node in self.scheduler_nodes():
            for access in node.read_writes.reads | node.read_writes.writes:
                buf_accesses[access.name].append(access)
        return buf_accesses

    @cache_on_self
    def op_counts(self) -> collections.Counter[str]:
        counts: collections.Counter[str] = collections.Counter()
        for node in self.scheduler_nodes():
            counts.update(node._body.op_counts)
        return counts

    def contains_op(self, op_name: str) -> bool:
        """True if V.ops.{op_name} is used in node_schedule"""
        return bool(self.op_counts().get(op_name))

    def get_mutations(self) -> OrderedSet[str]:
        mutations: OrderedSet[str] = OrderedSet()
        for node in self.scheduler_nodes():
            for buf in node.get_outputs():
                mutations.update(buf.get_mutations())
        return mutations

    @cache_on_self
    def select_index_dtype(self) -> torch.dtype:
        # Gather all used buffer names
        buffer_names: OrderedSet[str] = OrderedSet()
        for node in self.scheduler_nodes():
            buffer_names.update(node.get_buffer_names())
            buffer_names.update(node.used_buffer_names())
        buffers = [V.graph.get_buffer(name) for name in buffer_names]

        # In theory we can separately check xnumel and rnumel are <= int_max
        # but some indexers do use the full linear index so we need to be
        # conservative here.
        total_numel = self.numel * self.reduction_numel

        from .simd import SIMDScheduling

        if SIMDScheduling.can_use_32bit_indexing(total_numel, buffers):
            return torch.int32
        return torch.int64

    @cache_on_self
    def get_reduction_hint(self) -> ReductionHint:
        reductions = self.reduction_nodes()
        if len(reductions) > 0:
            hints = [self.reduction_hint(n) for n in reductions]
            if hints.count(hints[0]) == len(hints):
                reduction_hint_val = hints[0]
            else:
                reduction_hint_val = ReductionHint.DEFAULT

            if (
                reduction_hint_val == ReductionHint.INNER
                and self.has_non_contiguous_pw_in_reduction_kernel()
            ):
                reduction_hint_val = ReductionHint.DEFAULT
        else:
            reduction_hint_val = ReductionHint.DEFAULT
        return reduction_hint_val

    def has_non_contiguous_pw_in_reduction_kernel(self) -> bool:
        pointwise_nodes = [
            n
            for n in self.scheduler_nodes()
            if not n.is_reduction()
            and n.group[1][0] == self.numel * self.reduction_numel
        ]
        for node in pointwise_nodes:
            # An index can be an integer when loading a random seed.
            if not all(
                not isinstance(dep, MemoryDep)
                or dep.is_contiguous()
                or isinstance(dep.index, (sympy.Integer, int))
                or dep.stride1_for_last_dim()
                for dep in itertools.chain(
                    node.read_writes.reads, node.read_writes.writes
                )
            ):
                return True
        return False

    @staticmethod
    def reduction_hint(node: Any) -> ReductionHint:
        assert node.is_reduction()
        if node.node.data.reduction_hint != ReductionHint.INNER and all(
            dep.is_contiguous()
            for dep in itertools.chain(node.read_writes.reads, node.read_writes.writes)
        ):
            return ReductionHint.INNER
        else:
            return node.node.data.reduction_hint


class MemoryEstimator:
    """
    Estimate various properties of the kernel for use in heuristics.
    We simulate the memory effects of CSE/buffer elimination in codegen.
    """

    symbols: Tuple[sympy.Symbol, ...]
    kernel_sizes: Tuple[sympy.Expr, ...]
    node_index_vars: List[sympy.Expr]

    def __init__(self, features: SIMDKernelFeatures, groups: Sequence[sympy.Expr]):
        self.features = features
        self.inside_reduction = features.is_reduction()
        self.outside_loop = MemoryEstimate()
        self.loops = [MemoryEstimate()]
        self.persistent = MemoryEstimate()
        self.store_buffer_names: OrderedSet[str] = OrderedSet()
        self.must_keep_buffers: OrderedSet[str] = OrderedSet()
        self.groups = groups

        if len(groups) == 2:
            self.symbols = (sympy.Symbol("x"), sympy.Symbol("r"))
        elif len(groups) == 3:
            self.symbols = (sympy.Symbol("x"), sympy.Symbol("y"), sympy.Symbol("r"))
        else:
            raise NotImplementedError(len(groups))

        self.simulate_codegen()
        self.remove_kernel_local()

    def simulate_codegen(self) -> None:
        from .simd import SIMDKernel

        kernel_size_outside_loop = (*self.groups[:-1], sympy.S.One)
        kernel_size_inside_loop = tuple(self.groups)
        self.kernel_sizes = kernel_size_inside_loop

        for node in self.features.node_schedule:
            if node is DisableReduction:
                self.inside_reduction = False
                self.kernel_sizes = kernel_size_outside_loop
                continue
            elif node is EnableReduction:
                self.inside_reduction = True
                self.kernel_sizes = kernel_size_inside_loop
                self.loops.append(MemoryEstimate())
                continue
            assert isinstance(node, SchedulerNode)
            self.node_index_vars = [
                *itertools.chain.from_iterable(
                    SIMDKernel.map_kernel_groups_to_node_sizes(
                        self.kernel_sizes, node.get_ranges(), self.set_ranges
                    )
                )
            ]

            for dep in node.read_writes.reads:
                name, dep = self.process_dep(dep)
                if not self.persistent.writes.get(name):  # cache miss?
                    self.persistent.reads[name].add(dep)
                if not (
                    self.outside_loop.writes.get(name)
                    or self.loops[-1].writes.get(name)
                ):
                    self.scope(dep).reads[name].add(dep)
                    if name in self.store_buffer_names and self.loops[-1].reads.get(
                        name
                    ):
                        self.must_keep_buffers.add(name)

            for dep in node.read_writes.writes:
                name, dep = self.process_dep(dep)
                self.store_buffer_names.add(name)
                self.persistent.writes[name].add(dep)
                self.scope(dep).writes[name].add(dep)

    def remove_kernel_local(self) -> None:
        # Remove any kernel-local buffers
        for name in self.store_buffer_names:
            if not self.persistent.reads.get(
                name
            ) and V.graph.scheduler.can_buffer_be_removed_through_fusion(
                name, self.store_buffer_names
            ):
                self.persistent.remove(name)
                if name not in self.must_keep_buffers:
                    # we can also remove this from the looped kernel
                    self.outside_loop.remove(name)
                    for loop in self.loops:
                        loop.remove(name)

        if not self.loops[-1]:
            self.loops.pop()  # for pointwise ops

    def scope(self, dep: MemoryDep) -> MemoryEstimate:
        """Determine how a read/write should be categorized"""
        if self.inside_reduction and (
            self.symbols[-1] in dep.index.free_symbols or dep.is_indirect()
        ):
            return self.loops[-1]
        return self.outside_loop

    def set_ranges(self, *lengths: List[List[sympy.Expr]]) -> List[List[sympy.Expr]]:
        assert len(self.kernel_sizes) == len(lengths)
        return [
            self.make_flat_range(sym, numel, length)
            for sym, numel, length in zip(self.symbols, self.kernel_sizes, lengths)
        ]

    def process_dep(self, dep: Dep) -> Tuple[str, MemoryDep]:
        assert isinstance(dep, MemoryDep)
        assert len(dep.var_names) == len(self.node_index_vars)
        index = sympy_subs(dep.index, dict(zip(dep.var_names, self.node_index_vars)))
        index = V.graph.sizevars.simplify_with_ranges(
            index, dict(zip(self.symbols, self.kernel_sizes))
        )
        return dep.name, MemoryDep(
            name=dep.name,
            index=index,
            var_names=self.symbols,
            size=self.kernel_sizes,
            mode=dep.mode,
        )

    @staticmethod
    def make_flat_range(
        sym: sympy.Symbol, numel: sympy.Expr, lengths: List[sympy.Expr]
    ) -> List[sympy.Expr]:
        if len(lengths) == 1 and numel == lengths[0]:
            return [sym]
        divisor = sympy.S.One
        itervars = []
        for length in reversed(lengths):
            if V.graph.sizevars.statically_known_equals(divisor * length, numel):
                expr = FloorDiv(sym, divisor)
            else:
                expr = ModularIndexing(sym, divisor, length)
            itervars.append(expr)
            divisor = divisor * length
        return [*reversed(itervars)]


@dataclasses.dataclass
class MemoryEstimate:
    reads: Dict[str, OrderedSet[MemoryDep]] = dataclasses.field(
        default_factory=functools.partial(collections.defaultdict, OrderedSet)
    )
    writes: Dict[str, OrderedSet[MemoryDep]] = dataclasses.field(
        default_factory=functools.partial(collections.defaultdict, OrderedSet)
    )

    def remove(self, name: str) -> None:
        self.reads.pop(name, None)
        self.writes.pop(name, None)

    def __bool__(self) -> bool:
        return bool(self.reads or self.writes)
