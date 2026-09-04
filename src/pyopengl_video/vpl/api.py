"""ctypes declarations for oneVPL, Intel's video encoding interface.

``libvpl.dll`` is a dispatcher: it finds the runtime the installed driver
provides -- on older parts the Media SDK, on newer ones the oneVPL runtime --
and forwards to it. The interface is flat C, so unlike NvEncodeAPI there is no
function table to fill in: the entry points are ordinary exports, bound here
with their argument types.

**The structures are packed, and getting that wrong is silent.** The header
wraps each in a ``MFX_PACK_BEGIN_*`` directive giving it 4- or 8-byte packing,
which changes the layout wherever a wide field would otherwise be padded:
``mfxFrameInfo`` holds a ``mfxU64`` in one arm of its geometry union and is 68
bytes packed, 80 unpacked. A runtime handed the unpacked layout answers
``MFX_ERR_INVALID_VIDEO_PARAM`` to every call and names nothing, so
:data:`EXPECTED_SIZES` records what each structure must measure and
``tests/test_vpl_abi.py`` holds the declarations to it.

The layouts, enumerations and packing describe the API as published in the
oneVPL headers distributed by `intel/libvpl <https://github.com/intel/libvpl>`_
under the MIT licence -- see NOTICES.md.
"""
from __future__ import annotations

import ctypes
import sys
from collections.abc import Callable
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    Union,
    c_int32,
    c_int64,
    c_uint8,
    c_uint16,
    c_uint32,
    c_uint64,
    c_void_p,
)
from typing import TYPE_CHECKING, Any

from pyopengl_video.encoder import EncoderError

u8, u16, u32, u64, i64 = c_uint8, c_uint16, c_uint32, c_uint64, c_int64


def library_name() -> str:
    """The oneVPL dispatcher on this platform."""
    return 'libvpl.dll' if sys.platform == 'win32' else 'libvpl.so.2'


LIBRARY_NAME = library_name()


def fourcc(text: str) -> int:
    """The four-character code the interface spells a format with."""
    return int.from_bytes(text.encode('ascii'), 'little')


def fourcc_name(code: int) -> str:
    """A four-character code, back as the text it stands for."""
    return code.to_bytes(4, 'little').decode('ascii', 'replace')


# --------------------------------------------------------------- constants

MFX_CODEC_AVC = fourcc('AVC ')

#: RGB4 is BGRA in memory and RGB4's opposite, BGR4, is RGBA. The comments in
#: the header name the byte that comes first, and these are what a Direct3D
#: format maps onto.
MFX_FOURCC_NV12 = fourcc('NV12')
MFX_FOURCC_RGB4 = fourcc('RGB4')       # bytes B, G, R, A
MFX_FOURCC_BGR4 = fourcc('BGR4')       # bytes R, G, B, A

MFX_IMPL_SOFTWARE = 0x0001
MFX_IMPL_HARDWARE = 0x0002
MFX_IMPL_HARDWARE_ANY = 0x0004
MFX_IMPL_VIA_D3D9 = 0x0200
MFX_IMPL_VIA_D3D11 = 0x0300
MFX_IMPL_VIA_VAAPI = 0x0400

MFX_IOPATTERN_IN_VIDEO_MEMORY = 0x01
MFX_IOPATTERN_IN_SYSTEM_MEMORY = 0x02

MFX_PICSTRUCT_PROGRESSIVE = 0x01
MFX_CHROMAFORMAT_YUV420 = 1
MFX_CHROMAFORMAT_YUV444 = 3

MFX_RATECONTROL_CBR = 1
MFX_RATECONTROL_VBR = 2
MFX_RATECONTROL_CQP = 3

MFX_PROFILE_AVC_BASELINE = 66
MFX_PROFILE_AVC_MAIN = 77
MFX_PROFILE_AVC_HIGH = 100

MFX_FRAMETYPE_I = 0x0001
MFX_FRAMETYPE_P = 0x0002
MFX_FRAMETYPE_B = 0x0004
MFX_FRAMETYPE_REF = 0x0040
MFX_FRAMETYPE_IDR = 0x0080

MFX_HANDLE_D3D11_DEVICE = 3
MFX_HANDLE_VA_DISPLAY = 4

MFX_EXTBUFF_CODING_OPTION_SPSPPS = fourcc('COSP')
MFX_EXTBUFF_VIDEO_SIGNAL_INFO = fourcc('VSIN')

MFX_TIMESTAMP_UNKNOWN = 0xFFFFFFFFFFFFFFFF

#: Memory type flags an allocation request carries, for reading its purpose.
MFX_MEMTYPE_DXVA2_DECODER_TARGET = 0x0010
MFX_MEMTYPE_SYSTEM_MEMORY = 0x0040
MFX_MEMTYPE_FROM_ENCODE = 0x0100
MFX_MEMTYPE_INTERNAL_FRAME = 0x0001
MFX_MEMTYPE_EXTERNAL_FRAME = 0x0002

#: Status codes. Negative is a failure, positive a warning that the call
#: nonetheless completed.
MFX_ERR_NONE = 0
STATUS_NAMES = {
    0: 'MFX_ERR_NONE', -1: 'MFX_ERR_UNKNOWN', -2: 'MFX_ERR_NULL_PTR',
    -3: 'MFX_ERR_UNSUPPORTED', -4: 'MFX_ERR_MEMORY_ALLOC',
    -5: 'MFX_ERR_NOT_ENOUGH_BUFFER', -6: 'MFX_ERR_INVALID_HANDLE',
    -7: 'MFX_ERR_LOCK_MEMORY', -8: 'MFX_ERR_NOT_INITIALIZED',
    -9: 'MFX_ERR_NOT_FOUND', -10: 'MFX_ERR_MORE_DATA', -11: 'MFX_ERR_MORE_SURFACE',
    -12: 'MFX_ERR_ABORTED', -13: 'MFX_ERR_DEVICE_LOST',
    -14: 'MFX_ERR_INCOMPATIBLE_VIDEO_PARAM', -15: 'MFX_ERR_INVALID_VIDEO_PARAM',
    -16: 'MFX_ERR_UNDEFINED_BEHAVIOR', -17: 'MFX_ERR_DEVICE_FAILED',
    -18: 'MFX_ERR_MORE_BITSTREAM', -21: 'MFX_ERR_GPU_HANG',
    -22: 'MFX_ERR_REALLOC_SURFACE', -23: 'MFX_ERR_RESOURCE_MAPPED',
    -24: 'MFX_ERR_NOT_IMPLEMENTED', -25: 'MFX_ERR_MORE_EXTBUFFER',
    1: 'MFX_WRN_IN_EXECUTION', 2: 'MFX_WRN_DEVICE_BUSY',
    3: 'MFX_WRN_VIDEO_PARAM_CHANGED', 4: 'MFX_WRN_PARTIAL_ACCELERATION',
    5: 'MFX_WRN_INCOMPATIBLE_VIDEO_PARAM', 6: 'MFX_WRN_VALUE_NOT_CHANGED',
    7: 'MFX_WRN_OUT_OF_RANGE', 10: 'MFX_WRN_FILTER_SKIPPED',
}
MFX_ERR_MORE_DATA = -10
MFX_ERR_NOT_FOUND = -9
MFX_WRN_DEVICE_BUSY = 2
MFX_WRN_PARTIAL_ACCELERATION = 4
MFX_WRN_INCOMPATIBLE_VIDEO_PARAM = 5


class VPLError(EncoderError):
    """oneVPL refused a call. Carries the status code and its name."""

    def __init__(self, status: int, call: str, detail: str = ''):
        self.status = status
        self.name = STATUS_NAMES.get(status, f'status {status}')
        message = f'{call} failed: {self.name}'
        super().__init__(f'{message} ({detail})' if detail else message)


# -------------------------------------------------------------- structures
# Each carries the packing its MFX_PACK_BEGIN_* directive gives it in the
# header. The value is load-bearing: see this module's docstring.


class mfxVersion(Structure):
    _pack_ = 4
    _fields_ = [('Minor', u16), ('Major', u16)]

    def __str__(self) -> str:
        return f'{self.Major}.{self.Minor}'


class _Geometry(Structure):
    _pack_ = 4
    _fields_ = [('Width', u16), ('Height', u16), ('CropX', u16), ('CropY', u16),
                ('CropW', u16), ('CropH', u16)]


class _BufferGeometry(Structure):
    _pack_ = 4
    _fields_ = [('BufferSize', u64), ('reserved5', u32)]


class _GeometryUnion(Union):
    _pack_ = 4
    _anonymous_ = ('frame',)
    _fields_ = [('frame', _Geometry), ('buffer', _BufferGeometry)]


class mfxFrameInfo(Structure):
    """Format and geometry of a surface. pack(4), and 68 bytes because of it."""

    _pack_ = 4
    _anonymous_ = ('geometry',)
    _fields_ = [
        ('reserved', u32 * 4), ('ChannelId', u16), ('BitDepthLuma', u16),
        ('BitDepthChroma', u16), ('Shift', u16),
        # mfxFrameId, flattened: its ViewId arm aliases DependencyId exactly.
        ('TemporalId', u16), ('PriorityId', u16), ('DependencyId', u16),
        ('QualityId', u16),
        ('FourCC', u32),
        ('geometry', _GeometryUnion),
        ('FrameRateExtN', u32), ('FrameRateExtD', u32),
        ('reserved3', u16), ('AspectRatioW', u16), ('AspectRatioH', u16),
        ('PicStruct', u16), ('ChromaFormat', u16), ('reserved2', u16),
    ]


class mfxInfoMFX(Structure):
    """Codec configuration. pack(4).

    Every arm of the trailing union is thirteen ``mfxU16`` wide, so the encode
    arm's fields are declared inline: the decode and JPEG arms alias them
    exactly, and nothing here reads those.
    """

    _pack_ = 4
    _fields_ = [
        ('reserved', u32 * 7), ('LowPower', u16), ('BRCParamMultiplier', u16),
        ('FrameInfo', mfxFrameInfo),
        ('CodecId', u32), ('CodecProfile', u16), ('CodecLevel', u16),
        ('NumThread', u16),
        ('TargetUsage', u16), ('GopPicSize', u16), ('GopRefDist', u16),
        ('GopOptFlag', u16), ('IdrInterval', u16), ('RateControlMethod', u16),
        ('InitialDelayInKB', u16), ('BufferSizeInKB', u16), ('TargetKbps', u16),
        ('MaxKbps', u16), ('NumSlice', u16), ('NumRefFrame', u16),
        ('EncodedOrder', u16),
    ]


class mfxInfoVPP(Structure):
    _pack_ = 4
    _fields_ = [('reserved', u32 * 8), ('In', mfxFrameInfo), ('Out', mfxFrameInfo)]


class _CodecUnion(Union):
    _pack_ = 4
    _fields_ = [('mfx', mfxInfoMFX), ('vpp', mfxInfoVPP)]


class mfxVideoParam(Structure):
    """What an encoder is configured with. pack(8)."""

    _pack_ = 8
    _fields_ = [
        ('AllocId', u32), ('reserved', u32 * 2), ('reserved3', u16),
        ('AsyncDepth', u16),
        ('codec', _CodecUnion),
        ('Protected', u16), ('IOPattern', u16),
        ('ExtParam', POINTER(c_void_p)), ('NumExtParam', u16), ('reserved2', u16),
    ]

    @property
    def mfx(self) -> mfxInfoMFX:
        """The codec arm of the configuration union."""
        return self.codec.mfx

    @property
    def vpp(self) -> mfxInfoVPP:
        """The video-processing arm of the configuration union."""
        return self.codec.vpp


class _ExtParamUnion(Union):
    _pack_ = 8
    _fields_ = [('ExtParam', c_void_p), ('reserved2', u64)]


class mfxFrameData(Structure):
    """Where a surface's pixels are. pack(8).

    For a surface in video memory only :attr:`MemId` matters: it is the token
    the frame allocator turns back into a Direct3D texture.
    """

    _pack_ = 8
    _anonymous_ = ('external',)
    _fields_ = [
        ('external', _ExtParamUnion), ('NumExtParam', u16), ('reserved', u16 * 9),
        ('MemType', u16), ('PitchHigh', u16),
        ('TimeStamp', u64), ('FrameOrder', u32), ('Locked', u16), ('Pitch', u16),
        ('Y', c_void_p), ('UV', c_void_p), ('Cr', c_void_p), ('A', c_void_p),
        ('MemId', c_void_p), ('Corrupted', u16), ('DataFlag', u16),
    ]


class mfxStructVersion(Structure):
    _pack_ = 4
    _fields_ = [('Minor', u8), ('Major', u8)]


class _SurfaceInterfaceUnion(Union):
    _pack_ = 8
    _fields_ = [('FrameInterface', c_void_p), ('reserved', u32 * 2)]


class mfxFrameSurface1(Structure):
    """One picture handed to the encoder. pack(8)."""

    _pack_ = 8
    _fields_ = [
        ('interface', _SurfaceInterfaceUnion), ('Version', mfxStructVersion),
        ('reserved1', u16 * 3), ('Info', mfxFrameInfo), ('Data', mfxFrameData),
    ]


class _BitstreamHeader(Structure):
    _pack_ = 8
    _fields_ = [('EncryptedData', c_void_p), ('ExtParam', c_void_p),
                ('NumExtParam', u16), ('reserved1', u16), ('CodecId', u32)]


class _BitstreamHeaderUnion(Union):
    _pack_ = 8
    _fields_ = [('header', _BitstreamHeader), ('reserved', u32 * 6)]


class mfxBitstream(Structure):
    """The compressed output buffer. pack(8).

    ``Data`` is the caller's; ``DataOffset`` and ``DataLength`` say which part of
    it the encoder wrote. Time stamps are in 90 kHz units, which is this
    package's timescale.
    """

    _pack_ = 8
    _fields_ = [
        ('header', _BitstreamHeaderUnion), ('DecodeTimeStamp', i64),
        ('TimeStamp', u64), ('Data', POINTER(u8)), ('DataOffset', u32),
        ('DataLength', u32), ('MaxLength', u32), ('PicStruct', u16),
        ('FrameType', u16), ('DataFlag', u16), ('reserved2', u16),
    ]


class mfxExtBuffer(Structure):
    _pack_ = 4
    _fields_ = [('BufferId', u32), ('BufferSz', u32)]


class mfxExtCodingOptionSPSPPS(Structure):
    """Where the encoder writes its parameter sets. pack(8)."""

    _pack_ = 8
    _fields_ = [
        ('Header', mfxExtBuffer), ('SPSBuffer', POINTER(u8)), ('PPSBuffer', POINTER(u8)),
        ('SPSBufSize', u16), ('PPSBufSize', u16), ('SPSId', u16), ('PPSId', u16),
    ]


class mfxExtVideoSignalInfo(Structure):
    """What the stream says its colour is. pack(4).

    The hardware picks the matrix it converts RGB with, so the stream has to
    declare which one, or a player is left guessing and the picture comes back
    with the wrong saturation.
    """

    _pack_ = 4
    _fields_ = [
        ('Header', mfxExtBuffer), ('VideoFormat', u16), ('VideoFullRange', u16),
        ('ColourDescriptionPresent', u16), ('ColourPrimaries', u16),
        ('TransferCharacteristics', u16), ('MatrixCoefficients', u16),
    ]


class mfxEncodeCtrl(Structure):
    """Per-picture instructions, such as demanding a key frame. pack(8)."""

    _pack_ = 8
    _fields_ = [
        ('Header', mfxExtBuffer), ('reserved', u32 * 4), ('reserved1', u16),
        ('MfxNalUnitType', u16), ('SkipFrame', u16), ('QP', u16), ('FrameType', u16),
        ('NumExtParam', u16), ('NumPayload', u16), ('reserved2', u16),
        ('ExtParam', c_void_p), ('Payload', c_void_p),
    ]


class mfxFrameAllocRequest(Structure):
    """What the runtime is asking the allocator for. pack(4)."""

    _pack_ = 4
    _fields_ = [
        ('AllocId', u32), ('reserved3', u32 * 3), ('Info', mfxFrameInfo),
        ('Type', u16), ('NumFrameMin', u16), ('NumFrameSuggested', u16),
        ('reserved2', u16),
    ]


class mfxFrameAllocResponse(Structure):
    """What the allocator gave it. pack(8)."""

    _pack_ = 8
    _fields_ = [
        ('AllocId', u32), ('reserved', u32 * 3), ('mids', POINTER(c_void_p)),
        ('NumFrameActual', u16), ('reserved2', u16),
    ]


# The allocator's callbacks are __cdecl, as MFX_CDECL resolves to on Windows.
ALLOC_FUNC = CFUNCTYPE(c_int32, c_void_p, POINTER(mfxFrameAllocRequest),
                       POINTER(mfxFrameAllocResponse))
LOCK_FUNC = CFUNCTYPE(c_int32, c_void_p, c_void_p, POINTER(mfxFrameData))
GETHDL_FUNC = CFUNCTYPE(c_int32, c_void_p, c_void_p, POINTER(c_void_p))
FREE_FUNC = CFUNCTYPE(c_int32, c_void_p, POINTER(mfxFrameAllocResponse))


class mfxFrameAllocator(Structure):
    """The five callbacks the runtime allocates surfaces through. pack(8)."""

    _pack_ = 8
    _fields_ = [
        ('reserved', u32 * 4), ('pthis', c_void_p), ('Alloc', ALLOC_FUNC),
        ('Lock', LOCK_FUNC), ('Unlock', LOCK_FUNC), ('GetHDL', GETHDL_FUNC),
        ('Free', FREE_FUNC),
    ]


#: What each structure must measure on a 64-bit build. A mismatch means the
#: packing above is wrong, which a runtime reports only as an invalid parameter.
EXPECTED_SIZES = {
    'mfxVersion': 4,
    'mfxFrameInfo': 68,
    'mfxInfoMFX': 136,
    'mfxInfoVPP': 168,
    'mfxVideoParam': 208,
    'mfxFrameData': 96,
    'mfxFrameSurface1': 184,
    'mfxBitstream': 72,
    'mfxExtBuffer': 8,
    'mfxExtCodingOptionSPSPPS': 32,
    'mfxExtVideoSignalInfo': 20,
    'mfxEncodeCtrl': 56,
    'mfxFrameAllocRequest': 92,
    'mfxFrameAllocResponse': 32,
    'mfxFrameAllocator': 64,
}

#: Field offsets that a wrong union or a missed reserved array would move, and
#: that no runtime error would name.
EXPECTED_OFFSETS = {
    ('mfxFrameInfo', 'FourCC'): 32,
    ('mfxFrameInfo', 'Width'): 36,
    ('mfxFrameInfo', 'FrameRateExtN'): 48,
    ('mfxFrameInfo', 'PicStruct'): 62,
    ('mfxInfoMFX', 'FrameInfo'): 32,
    ('mfxInfoMFX', 'CodecId'): 100,
    ('mfxInfoMFX', 'RateControlMethod'): 120,
    ('mfxInfoMFX', 'BufferSizeInKB'): 124,
    ('mfxInfoMFX', 'TargetKbps'): 126,
    ('mfxVideoParam', 'codec'): 16,
    ('mfxVideoParam', 'IOPattern'): 186,
    ('mfxFrameData', 'TimeStamp'): 32,
    ('mfxFrameData', 'MemId'): 80,
    ('mfxFrameSurface1', 'Info'): 16,
    ('mfxFrameSurface1', 'Data'): 88,
    ('mfxBitstream', 'Data'): 40,
    ('mfxBitstream', 'DataLength'): 52,
}

#: Every entry point this package calls, with its argument types.
PROTOTYPES: dict[str, list[Any]] = {
    'MFXInit': [c_int32, POINTER(mfxVersion), POINTER(c_void_p)],
    'MFXClose': [c_void_p],
    'MFXQueryIMPL': [c_void_p, POINTER(c_int32)],
    'MFXQueryVersion': [c_void_p, POINTER(mfxVersion)],
    'MFXVideoCORE_SetHandle': [c_void_p, c_int32, c_void_p],
    'MFXVideoCORE_SetFrameAllocator': [c_void_p, POINTER(mfxFrameAllocator)],
    'MFXVideoCORE_SyncOperation': [c_void_p, c_void_p, u32],
    'MFXVideoENCODE_Query': [c_void_p, POINTER(mfxVideoParam), POINTER(mfxVideoParam)],
    'MFXVideoENCODE_QueryIOSurf': [c_void_p, POINTER(mfxVideoParam),
                                   POINTER(mfxFrameAllocRequest)],
    'MFXVideoENCODE_Init': [c_void_p, POINTER(mfxVideoParam)],
    'MFXVideoENCODE_Close': [c_void_p],
    'MFXVideoENCODE_GetVideoParam': [c_void_p, POINTER(mfxVideoParam)],
    'MFXVideoENCODE_EncodeFrameAsync': [c_void_p, POINTER(mfxEncodeCtrl),
                                        POINTER(mfxFrameSurface1),
                                        POINTER(mfxBitstream), POINTER(c_void_p)],
}


class VPL:
    """The oneVPL dispatcher, with the calls this package makes bound.

    Attribute access gives the bound entry points, so a caller writes
    ``vpl.MFXVideoENCODE_Init(session, params)``. :meth:`check` turns a status
    into a :class:`VPLError` naming the code.
    """

    _instance: VPL | None = None

    if TYPE_CHECKING:
        # Bound in __init__ from PROTOTYPES. Naming them here is what lets a
        # reader, and a type checker, see the interface this class offers.
        MFXInit: Callable[..., int]
        MFXClose: Callable[..., int]
        MFXQueryIMPL: Callable[..., int]
        MFXQueryVersion: Callable[..., int]
        MFXVideoCORE_SetHandle: Callable[..., int]
        MFXVideoCORE_SetFrameAllocator: Callable[..., int]
        MFXVideoCORE_SyncOperation: Callable[..., int]
        MFXVideoENCODE_Query: Callable[..., int]
        MFXVideoENCODE_QueryIOSurf: Callable[..., int]
        MFXVideoENCODE_Init: Callable[..., int]
        MFXVideoENCODE_Close: Callable[..., int]
        MFXVideoENCODE_GetVideoParam: Callable[..., int]
        MFXVideoENCODE_EncodeFrameAsync: Callable[..., int]

    def __init__(self, library_name: str = LIBRARY_NAME):
        # MFX_CDECL is __cdecl, which is what CDLL calls with on every platform.
        # The two conventions coincide on 64-bit Windows and differ on 32-bit.
        self.library = ctypes.CDLL(library_name)
        for name, argtypes in PROTOTYPES.items():
            function = getattr(self.library, name)
            function.argtypes = argtypes
            function.restype = c_int32
            setattr(self, name, function)

    @classmethod
    def load(cls) -> VPL:
        """The process-wide interface, loading the dispatcher on first use."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def check(status: int, call: str, detail: str = '') -> int:
        """Raise :class:`VPLError` on a failure; return the status otherwise.

        A positive status is a warning -- the call did what was asked and
        changed something on the way -- so it comes back for the caller to read.
        """
        if status < 0:
            raise VPLError(status, call, detail)
        return status
