"""Stand-ins for the oneVPL runtime, Direct3D and the WGL interop.

The oneVPL backend runs on Windows with Intel hardware. What it decides --
which surface a picture is submitted in, what a status means, the order things
are released in -- is ordinary Python, and :func:`machine` lets that run on any
platform: the encoder under test is the real
:class:`~pyopengl_video.vpl.encoder.VPLEncoder`, over the real
:class:`~pyopengl_video.windows.interop.SharedTexture`, and only the calls that
leave the process are replaced. Every one of those is recorded in
:attr:`Machine.calls`, in order.
"""
import ctypes
from ctypes import c_void_p

from pyopengl_video.vpl import api
from pyopengl_video.vpl import encoder as vpl_encoder
from pyopengl_video.windows import d3d11, interop

INTEL = d3d11.Adapter(index=0, description='Intel(R) Arc(TM)', vendor_id=0x8086,
                      device_id=0x56A0, luid=bytes(8))
NVIDIA = d3d11.Adapter(index=1, description='NVIDIA GeForce', vendor_id=0x10DE,
                       device_id=0x2489, luid=bytes([1, *bytes(7)]))


class Texture:
    """Stands in for :class:`pyopengl_video.windows.d3d11.Texture`."""

    def __init__(self, machine, address, width, height, format):
        self.machine = machine
        self.pointer = c_void_p(address)
        self.width, self.height, self.format = width, height, format

    def close(self):
        if self.pointer:
            self.machine.calls.append(('Texture.Release', self.pointer.value))
        self.pointer = c_void_p()


class Device:
    """Stands in for :class:`pyopengl_video.windows.d3d11.Device`."""

    def __init__(self, machine, adapter):
        self.machine = machine
        self.adapter = adapter
        self.pointer = c_void_p(0xD3D0)
        self.context = c_void_p(0xD3D8)

    def create_texture(self, width, height, format=d3d11.DXGI_FORMAT_B8G8R8A8_UNORM,
                       **_options):
        if self.machine.fail_textures_after == len(self.machine.textures):
            raise interop.InteropError(-2147024882, 'ID3D11Device::CreateTexture2D')
        texture = Texture(self.machine, 0x10000 + 0x100 * len(self.machine.textures),
                          width, height, format)
        self.machine.textures.append(texture)
        return texture

    def close(self):
        if self.pointer:
            self.machine.calls.append(('Device.Release',))
        self.pointer = c_void_p()


class Runtime:
    """Stands in for :class:`pyopengl_video.vpl.api.VPL`.

    statuses -- what ``EncodeFrameAsync`` answers, call by call; ``MFX_ERR_NONE``
        once they run out
    sync_status -- what ``SyncOperation`` answers
    payload -- the compressed picture a successful submission writes
    stated_length -- the ``DataLength`` it reports, where that is not the
        payload's own
    """

    check = staticmethod(api.VPL.check)

    def __init__(self, machine):
        self.machine = machine
        self.statuses = []
        self.sync_status = api.MFX_ERR_NONE
        self.payload = b'\x00\x00\x00\x01\x65picture'
        self.stated_length = None
        self.buffer_kb = 64
        self.surfaces = []
        self.configured = {}
        self.allocator = None

    def _note(self, name):
        self.machine.calls.append((name,))
        return api.MFX_ERR_NONE

    def MFXInit(self, _implementation, _version, session):
        session._obj.value = 0x5E55  # noqa: SLF001 - what a byref argument points at
        return self._note('MFXInit')

    def MFXVideoCORE_SetHandle(self, *_arguments):
        return self._note('MFXVideoCORE_SetHandle')

    def MFXVideoCORE_SetFrameAllocator(self, _session, allocator):
        self.allocator = allocator._obj  # noqa: SLF001 - what a byref argument points at
        return self._note('MFXVideoCORE_SetFrameAllocator')

    def MFXVideoENCODE_QueryIOSurf(self, _session, _parameters, request):
        request._obj.NumFrameSuggested = 3  # noqa: SLF001 - what a byref argument points at
        return self._note('MFXVideoENCODE_QueryIOSurf')

    def MFXVideoENCODE_Init(self, _session, parameters):
        mfx = parameters._obj.mfx  # noqa: SLF001 - what a byref argument points at
        info = mfx.FrameInfo
        self.configured = dict(
            width=info.Width, height=info.Height, crop=(info.CropW, info.CropH),
            rate=(info.FrameRateExtN, info.FrameRateExtD),
            gop=mfx.GopPicSize, distance=mfx.GopRefDist,
            method=mfx.RateControlMethod, multiplier=mfx.BRCParamMultiplier,
            target=mfx.TargetKbps, most=mfx.MaxKbps, initial=mfx.InitialDelayInKB)
        return self._note('MFXVideoENCODE_Init')

    def MFXVideoENCODE_GetVideoParam(self, _session, parameters):
        mfx = parameters._obj.mfx  # noqa: SLF001 - what a byref argument points at
        mfx.BufferSizeInKB = self.buffer_kb
        mfx.GopRefDist = self.configured.get('distance', 1)
        return api.MFX_ERR_NONE

    def MFXVideoENCODE_EncodeFrameAsync(self, _session, _control, surface,
                                        bitstream, sync_point):
        self.machine.calls.append(('EncodeFrameAsync',))
        if surface is not None:
            submitted = surface._obj  # noqa: SLF001 - what a byref argument points at
            self.surfaces.append((ctypes.addressof(submitted), submitted.Data.MemId,
                                  submitted.Data.TimeStamp))
        status = self.statuses.pop(0) if self.statuses else api.MFX_ERR_NONE
        if status == api.MFX_ERR_NONE:
            if surface is None:
                return api.MFX_ERR_MORE_DATA
            stream = bitstream._obj  # noqa: SLF001 - what a byref argument points at
            ctypes.memmove(stream.Data, self.payload, len(self.payload))
            stream.DataLength = (len(self.payload) if self.stated_length is None
                                 else self.stated_length)
            stream.FrameType = api.MFX_FRAMETYPE_I | api.MFX_FRAMETYPE_IDR
            stream.TimeStamp = submitted.Data.TimeStamp
            sync_point._obj.value = 0x51C  # noqa: SLF001 - what a byref argument points at
        return status

    def MFXVideoCORE_SyncOperation(self, *_arguments):
        self.machine.calls.append(('SyncOperation',))
        return self.sync_status

    def MFXVideoENCODE_Close(self, _session):
        return self._note('MFXVideoENCODE_Close')

    def MFXClose(self, _session):
        return self._note('MFXClose')


class Machine:
    """A Windows machine with an Intel encoder, as far as the backend can tell.

    calls -- every call that would have left the process, in order
    context_adapter -- the adapter the OpenGL context is said to be on
    fail -- names of the calls in :attr:`calls` that raise or answer failure
    fail_textures_after -- how many Direct3D textures are made before
        ``CreateTexture2D`` fails; None for never
    """

    def __init__(self):
        self.calls = []
        self.textures = []
        self.context_adapter = INTEL
        self.fail = set()
        self.fail_textures_after = None
        self.runtime = Runtime(self)
        self.devices = []
        self._names = 0

    def named(self, name):
        """The calls of this name, in order."""
        return [call for call in self.calls if call[0] == name]

    def _next(self):
        self._names += 1
        return self._names

    def _gl(self, name, *arguments):
        """Record an OpenGL or WGL call, raising where the machine says it fails."""
        self.calls.append((name, *arguments))
        if name in self.fail:
            raise RuntimeError(f'{name}: no OpenGL context is current')

    def install(self, monkeypatch):
        """Put the stand-ins where the backend looks for the real things."""
        monkeypatch.setattr(api.VPL, 'load', classmethod(lambda _cls: self.runtime))
        monkeypatch.setattr(vpl_encoder.d3d11, 'Device', self._device)
        monkeypatch.setattr(interop, 'adapter_for_context', lambda: self.context_adapter)
        monkeypatch.setattr(interop, 'available', lambda: True)
        monkeypatch.setattr(interop, 'wglDXOpenDeviceNV', lambda _device: 0xDEB1CE)
        monkeypatch.setattr(interop, 'wglDXCloseDeviceNV',
                            lambda _device: self._gl('wglDXCloseDeviceNV'))
        monkeypatch.setattr(interop, 'wglDXRegisterObjectNV', self._register)
        monkeypatch.setattr(interop, 'wglDXUnregisterObjectNV',
                            lambda _device, handle: self._gl('wglDXUnregisterObjectNV',
                                                             handle))
        monkeypatch.setattr(interop, 'wglDXLockObjectsNV', self._lock('wglDXLockObjectsNV'))
        monkeypatch.setattr(interop, 'wglDXUnlockObjectsNV',
                            self._lock('wglDXUnlockObjectsNV'))
        monkeypatch.setattr(interop, 'glGenTextures', lambda _count: self._next())
        monkeypatch.setattr(interop, 'glDeleteTextures',
                            lambda names: self._gl('glDeleteTextures', *names))
        monkeypatch.setattr(interop, 'create_framebuffer',
                            self._framebuffer)
        monkeypatch.setattr(interop, 'delete_framebuffer',
                            lambda name: self._gl('glDeleteFramebuffers', name))
        return self

    def _device(self, adapter=None, **_options):
        device = Device(self, adapter)
        self.devices.append(device)
        return device

    def _framebuffer(self, _texture, _target=None):
        if 'glGenFramebuffers' in self.fail:
            raise RuntimeError('glGenFramebuffers: the attachment was refused')
        return self._next()

    def _register(self, _device, _resource, name, _target, _access):
        if 'wglDXRegisterObjectNV' in self.fail:
            return 0
        self.calls.append(('wglDXRegisterObjectNV', name))
        return 0x4000 + name

    def _lock(self, name):
        def call(_device, _count, handles):
            self.calls.append((name, handles[0]))
            return name not in self.fail
        return call
