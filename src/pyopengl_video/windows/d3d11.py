"""Which GPUs are present, and the Direct3D 11 textures an encoder reads.

A Windows encoder takes an ``ID3D11Texture2D``, and a D3D11 device belongs to
one adapter. Which adapter matters: a texture is only reachable without a copy
by an encoder on the GPU that holds it, so a recorder has to build its device on
the same adapter the renderer is using. :class:`Adapter` carries the LUID that
makes that comparison exact, and
:func:`~pyopengl_video.windows.interop.adapter_for_context` is what asks OpenGL
for its own.
"""
from __future__ import annotations

import ctypes
import dataclasses
from ctypes import POINTER, Structure, byref, c_int32, c_size_t, c_uint32, c_void_p, c_wchar
from typing import Any

from pyopengl_video.windows.com import GUID, InteropError, check, guid, load, method, release

#: DXGI pixel formats this package names. The two RGBA orders are the ones an
#: encoder is told about, and NV12 is what a driver allocates behind one.
DXGI_FORMAT_R8G8B8A8_UNORM = 28          # bytes R, G, B, A
DXGI_FORMAT_B8G8R8A8_UNORM = 87          # bytes B, G, R, A
DXGI_FORMAT_NV12 = 103

D3D_DRIVER_TYPE_UNKNOWN = 0
D3D_DRIVER_TYPE_HARDWARE = 1
D3D11_SDK_VERSION = 7

D3D11_CREATE_DEVICE_BGRA_SUPPORT = 0x0020
D3D11_CREATE_DEVICE_VIDEO_SUPPORT = 0x0800

D3D11_USAGE_DEFAULT = 0
D3D11_USAGE_STAGING = 3
D3D11_BIND_SHADER_RESOURCE = 0x0008
D3D11_BIND_RENDER_TARGET = 0x0020
D3D11_BIND_DECODER = 0x0200
D3D11_CPU_ACCESS_READ = 0x20000
D3D11_MAP_READ = 1

IID_IDXGIFactory1 = guid('770aae78-f26f-4dba-a829-253c83d1b387')
IID_ID3D10Multithread = guid('9b7e4e00-342c-4106-a19f-4f2704f689f0')

# Vtable slots, counted from IUnknown. Named here so that the number appears
# once, beside the interface it belongs to.
_ENUM_ADAPTERS1 = 12                     # IDXGIFactory1
_GET_DESC1 = 10                          # IDXGIAdapter1
_CREATE_TEXTURE_2D = 5                   # ID3D11Device
_MAP, _UNMAP, _COPY_RESOURCE = 14, 15, 47    # ID3D11DeviceContext
_SET_MULTITHREAD_PROTECTED = 5           # ID3D10Multithread

DXGI_ERROR_NOT_FOUND = -2005270526       # the end of an adapter enumeration

#: PCI vendor identifiers, for saying whose GPU an adapter is.
VENDORS = {0x10DE: 'NVIDIA', 0x8086: 'Intel', 0x1002: 'AMD', 0x1414: 'Microsoft'}

#: How many bytes one pixel of a format takes, for the formats read back.
_BYTES_PER_PIXEL = {DXGI_FORMAT_R8G8B8A8_UNORM: 4, DXGI_FORMAT_B8G8R8A8_UNORM: 4}


class LUID(Structure):
    _fields_ = [('LowPart', c_uint32), ('HighPart', c_int32)]


class DXGI_ADAPTER_DESC1(Structure):
    _fields_ = [
        ('Description', c_wchar * 128), ('VendorId', c_uint32), ('DeviceId', c_uint32),
        ('SubSysId', c_uint32), ('Revision', c_uint32),
        ('DedicatedVideoMemory', c_size_t), ('DedicatedSystemMemory', c_size_t),
        ('SharedSystemMemory', c_size_t), ('AdapterLuid', LUID), ('Flags', c_uint32),
    ]


class D3D11_TEXTURE2D_DESC(Structure):
    _fields_ = [
        ('Width', c_uint32), ('Height', c_uint32), ('MipLevels', c_uint32),
        ('ArraySize', c_uint32), ('Format', c_uint32),
        ('SampleDescCount', c_uint32), ('SampleDescQuality', c_uint32),
        ('Usage', c_uint32), ('BindFlags', c_uint32), ('CPUAccessFlags', c_uint32),
        ('MiscFlags', c_uint32),
    ]


class D3D11_MAPPED_SUBRESOURCE(Structure):
    _fields_ = [('pData', c_void_p), ('RowPitch', c_uint32), ('DepthPitch', c_uint32)]


def vendor_name(vendor_id: int) -> str:
    """Whose GPU a PCI vendor identifier belongs to."""
    return VENDORS.get(vendor_id, f'0x{vendor_id:04X}')


@dataclasses.dataclass(frozen=True)
class Adapter:
    """One GPU, as DXGI describes it.

    index -- where it came in the enumeration
    description -- the driver's own name for it
    vendor_id/device_id -- PCI identifiers; :attr:`vendor` names the first
    luid -- the adapter's locally unique identifier, which is the only reliable
        way to say that this is the same GPU an OpenGL context is running on
    """

    index: int
    description: str
    vendor_id: int
    device_id: int
    luid: bytes

    @property
    def vendor(self) -> str:
        """Whose hardware this is: ``'NVIDIA'``, ``'Intel'``, ``'AMD'``..."""
        return vendor_name(self.vendor_id)

    def __str__(self) -> str:
        return f'{self.description} ({self.vendor}, LUID {self.luid.hex()})'


def _factory() -> c_void_p:
    """A DXGI factory, for the caller to release."""
    dxgi = load('dxgi.dll')
    created = c_void_p()
    check(dxgi.CreateDXGIFactory1(byref(IID_IDXGIFactory1), byref(created)),
          'CreateDXGIFactory1')
    return created


def _enumerate() -> list[tuple[Adapter, c_void_p]]:
    """Every adapter, paired with its live interface. The caller releases those."""
    factory = _factory()
    try:
        enumerate_adapters = method(factory, _ENUM_ADAPTERS1, c_uint32, POINTER(c_void_p))
        found = []
        index = 0
        while True:
            interface = c_void_p()
            if enumerate_adapters(factory, index, byref(interface)) != 0:
                break
            description = DXGI_ADAPTER_DESC1()
            check(method(interface, _GET_DESC1, POINTER(DXGI_ADAPTER_DESC1))(
                interface, byref(description)), 'IDXGIAdapter1::GetDesc1')
            found.append((
                Adapter(
                    index=index,
                    description=description.Description,
                    vendor_id=description.VendorId,
                    device_id=description.DeviceId,
                    luid=ctypes.string_at(byref(description.AdapterLuid), 8),
                ),
                interface,
            ))
            index += 1
        return found
    finally:
        release(factory)


def adapters() -> list[Adapter]:
    """Every GPU DXGI can see, in its own order."""
    found = _enumerate()
    try:
        return [adapter for adapter, _interface in found]
    finally:
        for _adapter, interface in found:
            release(interface)


def adapter_for_luid(luid: bytes) -> Adapter | None:
    """The adapter with this LUID, or None when no GPU here has it."""
    for adapter in adapters():
        if adapter.luid == luid:
            return adapter
    return None


class Texture:
    """A Direct3D 11 2D texture.

    Built by :meth:`Device.create_texture`. An encoder is handed
    :attr:`pointer`; :meth:`read` brings the pixels back to the host, which is
    how a test proves the picture in it is the one that was drawn.
    """

    def __init__(self, device: Device, pointer: c_void_p, width: int, height: int,
                 format: int):
        self.device = device
        self.pointer = pointer
        self.width = int(width)
        self.height = int(height)
        self.format = int(format)

    def read(self) -> bytes:
        """The pixels, row by row with no padding.

        Copies through a staging texture, so this crosses the bus and is for
        checking a frame rather than for moving one.
        """
        if self.format not in _BYTES_PER_PIXEL:
            raise InteropError(0, 'Texture.read',
                               f'no host layout is defined for DXGI format {self.format}')
        stride = self.width * _BYTES_PER_PIXEL[self.format]
        staging = self.device.create_texture(
            self.width, self.height, self.format, bind=0,
            usage=D3D11_USAGE_STAGING, cpu_access=D3D11_CPU_ACCESS_READ)
        try:
            context = self.device.context
            method(context, _COPY_RESOURCE, c_void_p, c_void_p)(
                context, staging.pointer, self.pointer)
            mapped = D3D11_MAPPED_SUBRESOURCE()
            check(method(context, _MAP, c_void_p, c_uint32, c_uint32, c_uint32,
                         POINTER(D3D11_MAPPED_SUBRESOURCE))(
                context, staging.pointer, 0, D3D11_MAP_READ, 0, byref(mapped)),
                'ID3D11DeviceContext::Map')
            try:
                # The driver chooses the row pitch, and it is rarely the width.
                rows = [
                    ctypes.string_at(mapped.pData + row * mapped.RowPitch, stride)
                    for row in range(self.height)
                ]
            finally:
                method(context, _UNMAP, c_void_p, c_uint32)(context, staging.pointer, 0)
            return b''.join(rows)
        finally:
            staging.close()

    def close(self) -> None:
        """Give the texture back. Safe to call twice."""
        release(self.pointer)
        self.pointer = c_void_p()

    def __enter__(self) -> Texture:
        return self

    def __exit__(self, *exception: Any) -> None:
        self.close()


class Device:
    """A Direct3D 11 device on one GPU.

    adapter -- which GPU to build on; None takes whichever the system offers
        first, which is rarely what a recorder wants
    video -- ask for the video support a hardware encoder needs
    multithread -- make the device safe to use from more than one thread, which
        Intel's encoder requires of a device handed to it

    Use it as a context manager, or call :meth:`close`: it holds driver objects
    that a garbage collection will not release in any predictable order.
    """

    def __init__(self, adapter: Adapter | None = None, *, video: bool = True,
                 multithread: bool = True):
        self.adapter = adapter
        self.pointer = c_void_p()
        self.context = c_void_p()
        flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT
        if video:
            flags |= D3D11_CREATE_DEVICE_VIDEO_SUPPORT
        interface, found = self._adapter_interface(adapter)
        try:
            device, context, level = c_void_p(), c_void_p(), c_uint32()
            check(load('d3d11.dll').D3D11CreateDevice(
                interface,
                D3D_DRIVER_TYPE_UNKNOWN if interface else D3D_DRIVER_TYPE_HARDWARE,
                None, flags, None, 0, D3D11_SDK_VERSION,
                byref(device), byref(level), byref(context)), 'D3D11CreateDevice')
            self.pointer, self.context = device, context
            self.feature_level = level.value
            self.adapter = found
        finally:
            release(interface)
        if multithread:
            self._protect()

    @staticmethod
    def _adapter_interface(adapter: Adapter | None) -> tuple[c_void_p, Adapter | None]:
        """The live interface for `adapter`, which the caller releases."""
        if adapter is None:
            return c_void_p(), None
        found = _enumerate()
        wanted, keep = c_void_p(), None
        for described, interface in found:
            if described.luid == adapter.luid:
                wanted, keep = interface, described
            else:
                release(interface)
        if not wanted:
            raise InteropError(0, 'D3D11CreateDevice',
                               f'no adapter here has LUID {adapter.luid.hex()}')
        return wanted, keep

    def _protect(self) -> None:
        """Turn on the device's own locking.

        Intel's encoder is handed this device and calls it from its own threads;
        without this the two race in the driver.
        """
        multithread = query_interface_or_none(self.context, IID_ID3D10Multithread)
        if multithread is None:              # pragma: no cover - every D3D11 has it
            return
        try:
            method(multithread, _SET_MULTITHREAD_PROTECTED, c_int32)(multithread, 1)
        finally:
            release(multithread)

    def create_texture(self, width: int, height: int,
                       format: int = DXGI_FORMAT_B8G8R8A8_UNORM, *,
                       bind: int | None = None, usage: int = D3D11_USAGE_DEFAULT,
                       cpu_access: int = 0, misc: int = 0) -> Texture:
        """Allocate a texture on this device.

        The default binding is render target plus shader resource, which is what
        both the OpenGL interop and the encoders accept for a colour surface. An
        NV12 surface a driver asks for gets the decoder binding instead, since
        that is the only one D3D11 allows for it.
        """
        if bind is None:
            bind = (D3D11_BIND_DECODER if format == DXGI_FORMAT_NV12
                    else D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE)
        description = D3D11_TEXTURE2D_DESC(
            Width=int(width), Height=int(height), MipLevels=1, ArraySize=1,
            Format=int(format), SampleDescCount=1, SampleDescQuality=0,
            Usage=usage, BindFlags=bind, CPUAccessFlags=cpu_access, MiscFlags=misc)
        created = c_void_p()
        check(method(self.pointer, _CREATE_TEXTURE_2D, POINTER(D3D11_TEXTURE2D_DESC),
                     c_void_p, POINTER(c_void_p))(
            self.pointer, byref(description), None, byref(created)),
            'ID3D11Device::CreateTexture2D',
            f'{width}x{height} DXGI format {format}')
        return Texture(self, created, width, height, format)

    def close(self) -> None:
        """Release the device and its context. Safe to call twice."""
        release(self.context)
        release(self.pointer)
        self.context = c_void_p()
        self.pointer = c_void_p()

    def __enter__(self) -> Device:
        return self

    def __exit__(self, *exception: Any) -> None:
        self.close()


def query_interface_or_none(interface: c_void_p, iid: GUID) -> c_void_p | None:
    """Another view of `interface`, or None when it does not offer one."""
    found = c_void_p()
    result = method(interface, 0, POINTER(GUID), POINTER(c_void_p))(
        interface, byref(iid), byref(found))
    return found if result >= 0 and found else None
