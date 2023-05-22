from typing import List, Union

import torch

_tensor_from_blob = torch.ops.inductor._tensor_from_blob
ALIGN_BYTES = 64
assert (ALIGN_BYTES & (ALIGN_BYTES - 1)) == 0, "must be power of 2"


class MicroTensor:
    """
    A cheaper to allocate tensor object that can be passed directly to
    both Triton and aten ops.
    """

    def __init__(self, data_ptr, size, stride, dtype, device):
        self._data_ptr = data_ptr
        self._size = size
        self._stride = stride
        self.dtype = dtype
        self.device = device

    def data_ptr(self):
        return self._data_ptr

    def __repr__(self):
        return f"MicroTensor({self._data_ptr!r}, {self._size!r}, {self._stride!r}, {self.dtype!r}, {self.device!r})"

    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    def as_strided(self, size, stride, offset=0):
        return MicroTensor(
            self._data_ptr + offset, size, stride, self.dtype, self.device
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {k: promote_to_tensor(v) for k, v in kwargs.items()} if kwargs else {}
        return func(*(promote_to_tensor(a) for a in args), **kwargs)


def promote_to_tensor(x: Union[torch.Tensor, MicroTensor]):
    """Upcast MicroTensors to Tensor"""
    if isinstance(x, MicroTensor):
        return _tensor_from_blob(x._data_ptr, x._size, x._stride, x.dtype, x.device)
    return x


def needed_numel(sizes, strides):
    """Compute number of elements needed to allocate a tensor"""
    numel = 1
    for size, stride in zip(sizes, strides):
        if size == 0:
            return 0
        numel += (size - 1) * stride
    return numel


def roundup(nbytes):
    """Round up to the nearest multiple of ALIGN_BYTES"""
    return (nbytes + ALIGN_BYTES - 1) & -ALIGN_BYTES


def needed_nbytes(sizes, strides, dtype):
    """Compute number of bytes needed to allocate a tensor"""
    return roundup(needed_numel(sizes, strides) * dtype.itemsize)


class TreeAllocator(object):
    """
    A minimal tensor memory allocator using a "stack of stacks" memory model.
    """

    def __init__(self, device: torch.device, nbytes: int, parent=None):
        if parent:
            self.parent: TreeAllocator = parent
            self.buffer = parent.alloc((nbytes,), (1,), torch.uint8)
        else:
            self.parent = None
            self.buffer = torch.empty(nbytes, dtype=torch.uint8, device=device)
        self.tos = self._buffer.data_ptr()
        self.allocations = []
        self.device = device
        self.end = self.tos + nbytes

    def alloc(self, size: List[int], stride: List[int], dtype: torch.dtype):
        result = MicroTensor(self.tos, size, stride, dtype, self.device)
        self.tos += needed_nbytes(size, stride, dtype)
        assert self.tos <= self.end, "overflow"
        self.allocations.append(result)
        return result

    def dealloc(self, obj: MicroTensor):
        assert (
            obj is self.allocations.pop()
        ), "must deallocate in last-in-first-out order"
        self.tos = obj._data_ptr

    def branch(self, nbytes):
        """Create a new allocator whose storage goes on this allocator's stack"""
        return TreeAllocator(self.device, nbytes, self)

    def release(self):
        assert not self.allocations
        if self.parent:
            self.parent.dealloc(self.buffer)
        self.tos = None
        self.buffer = None
