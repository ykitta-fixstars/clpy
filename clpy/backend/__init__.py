import contextlib

# from clpy.backend import compiler  # NOQA
from clpy.backend import device  # NOQA
from clpy.backend import function  # NOQA
from clpy.backend import memory  # NOQA
# from clpy.backend import memory_hook  # NOQA
# from clpy.backend import memory_hooks  # NOQA
from clpy.backend import pinned_memory  # NOQA
# from clpy.backend import profiler  # NOQA
# from clpy.backend import runtime  # NOQA
from clpy.backend import stream  # NOQA


_available = None


# import class and function
# from clpy.backend.compiler import compile_with_cache  # NOQA
from clpy.backend.device import Device  # NOQA
from clpy.backend.device import get_cublas_handle  # NOQA
from clpy.backend.device import get_device_id  # NOQA
from clpy.backend.function import Function  # NOQA
from clpy.backend.function import Module  # NOQA
from clpy.backend.memory import alloc  # NOQA
from clpy.backend.memory import malloc_managed  # NOQA
from clpy.backend.memory import ManagedMemory  # NOQA
from clpy.backend.memory import Memory  # NOQA
from clpy.backend.memory import MemoryPointer  # NOQA
from clpy.backend.memory import MemoryPool  # NOQA
from clpy.backend.memory import set_allocator  # NOQA
# from clpy.backend.memory_hook import MemoryHook  # NOQA
from clpy.backend.pinned_memory import alloc_pinned_memory  # NOQA
from clpy.backend.pinned_memory import PinnedMemory  # NOQA
from clpy.backend.pinned_memory import PinnedMemoryPointer  # NOQA
from clpy.backend.pinned_memory import PinnedMemoryPool  # NOQA
from clpy.backend.pinned_memory import set_pinned_memory_allocator  # NOQA
from clpy.backend.stream import Event  # NOQA
# from clpy.backend.stream import get_elapsed_time  # NOQA
from clpy.backend.stream import Stream  # NOQA


@contextlib.contextmanager
def profile():
    """Enable CUDA profiling during with statement.

    This function enables profiling on entering a with statement, and disables
    profiling on leaving the statement.

    >>> with clpy.backend.profile():
    ...    # do something you want to measure
    ...    pass

    """
    raise NotImplementedError("clpy does not support this")
#    profiler.start()
#    try:
#        yield
#    finally:
#        profiler.stop()
