"""The little of COM this package needs, over ctypes.

A COM interface is a pointer to a pointer to a table of function pointers, and
calling one means indexing that table. Nothing here is a general COM
implementation: it is the few operations the Windows encoder backends make, kept
in one place so that a vtable index and a calling convention are written once
rather than at each call site.

Interfaces are reference-counted. Everything obtained from a call that returns
one has to be given back with :func:`release`, and the objects in
:mod:`~pyopengl_video.windows.d3d11` do that in their ``close``.
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import POINTER, Structure, c_int32, c_uint16, c_uint32, c_void_p
from functools import cache
from typing import Any

from pyopengl_video.encoder import EncoderError


class InteropError(EncoderError):
    """A Windows interface refused a call.

    Carries the HRESULT, so a caller that wants to tell one failure from another
    can read :attr:`result` rather than parse the message.
    """

    def __init__(self, result: int, call: str, detail: str = ''):
        self.result = result & 0xFFFFFFFF
        self.call = call
        message = f'{call} failed: 0x{self.result:08X}'
        super().__init__(f'{message} ({detail})' if detail else message)


class GUID(Structure):
    """A Windows interface or class identifier."""

    _fields_ = [('Data1', c_uint32), ('Data2', c_uint16), ('Data3', c_uint16),
                ('Data4', ctypes.c_ubyte * 8)]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GUID):
            return NotImplemented
        return bytes(self) == bytes(other)

    def __hash__(self) -> int:
        return hash(bytes(self))

    def __bytes__(self) -> bytes:
        return ctypes.string_at(ctypes.byref(self), ctypes.sizeof(self))


def guid(text: str) -> GUID:
    """Build a :class:`GUID` from its usual ``8-4-4-4-12`` spelling."""
    first, second, third, fourth, rest = text.split('-')
    return GUID(int(first, 16), int(second, 16), int(third, 16),
                (ctypes.c_ubyte * 8)(*bytes.fromhex(fourth + rest)))


@cache
def load(name: str) -> Any:
    """Load a system library, once per process.

    Raises :class:`InteropError` rather than ``OSError``, so a caller catching
    the package's own error type sees a missing library too.
    """
    if sys.platform != 'win32':          # pragma: no cover - not reachable on Windows
        raise InteropError(0, f'loading {name}', 'only Windows has this library')
    try:
        return ctypes.WinDLL(name)
    except OSError as error:
        raise InteropError(0, f'loading {name}', str(error)) from error


def method(interface: c_void_p, index: int, *argtypes: Any) -> Any:
    """Bind entry `index` of `interface`'s vtable, returning a raw HRESULT.

    The result is deliberately a plain ``c_int32`` rather than ctypes'
    ``HRESULT``: several of these calls report an ordinary outcome with a
    failure code -- enumerating adapters ends with ``DXGI_ERROR_NOT_FOUND`` --
    and an exception raised inside the loop would be the wrong shape.
    """
    vtable = ctypes.cast(interface, POINTER(POINTER(c_void_p))).contents
    return ctypes.WINFUNCTYPE(c_int32, c_void_p, *argtypes)(vtable[index])


def check(result: int, call: str, detail: str = '') -> int:
    """Raise :class:`InteropError` unless `result` is a success HRESULT."""
    if result < 0:
        raise InteropError(result, call, detail)
    return result


# IUnknown, which every interface begins with.
QUERY_INTERFACE, ADD_REF, RELEASE = 0, 1, 2


def query_interface(interface: c_void_p, iid: GUID) -> c_void_p:
    """Ask `interface` for another view of itself, as `iid`."""
    found = c_void_p()
    check(method(interface, QUERY_INTERFACE, POINTER(GUID), POINTER(c_void_p))(
        interface, ctypes.byref(iid), ctypes.byref(found)), 'QueryInterface')
    return found


def release(interface: c_void_p | None) -> None:
    """Give one reference back. Safe on a null pointer, which is what a
    closed object holds."""
    if interface:
        method(interface, RELEASE)(interface)
