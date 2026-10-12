"""ctypes declarations for NVDEC and the CUDA driver API it decodes into.

Two libraries, both part of the NVIDIA display driver:

- ``libnvcuvid.so.1`` (``nvcuvid.dll``) is the decoder. Its parser reads the
  bitstream and calls back with each sequence, each picture to decode and
  each picture to display; its decoder turns a picture into NV12 in CUDA
  device memory.
- ``libcuda.so.1`` (``nvcuda.dll``) is the CUDA driver API: contexts, copies
  between device memory and OpenGL textures, and the registration that lets
  CUDA write a texture. The CUDA toolkit and runtime are not involved.

:class:`Cuda` and :class:`Cuvid` load them and bind the calls this package
makes. Every call returns a ``CUresult``; :meth:`Cuda.check` turns one that is
not success into a :class:`CUDAError` carrying the driver's name for it.

The layouts and enumerations describe the ABI published in
``dynlink_cuviddec.h``, ``dynlink_nvcuvid.h`` and ``dynlink_cuda.h``, as
distributed in nv-codec-headers under the MIT licence -- see NOTICES.md.
``tests/test_nvdec_abi.py`` checks the declarations here against sizes,
offsets and values recorded from those headers.
"""
from __future__ import annotations

import ctypes
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from ctypes import (
    POINTER,
    Structure,
    c_char_p,
    c_int,
    c_longlong,
    c_short,
    c_size_t,
    c_ubyte,
    c_uint,
    c_ulong,
    c_ulonglong,
    c_ushort,
    c_void_p,
)
from typing import Any

from pyopengl_video.decoder import DecoderError


def library_names() -> tuple[str, str]:
    """The driver's CUDA and decoder libraries on this platform."""
    if sys.platform == 'win32':
        return 'nvcuda.dll', 'nvcuvid.dll'
    return 'libcuda.so.1', 'libnvcuvid.so.1'


CUDA_LIBRARY_NAME, CUVID_LIBRARY_NAME = library_names()

# Both interfaces are __stdcall on Windows and cdecl everywhere else, which
# are one ABI on 64-bit Windows. The branch is written out because ctypes
# offers the Windows spellings on Windows alone.
if sys.platform == 'win32':
    _LOADER = ctypes.WinDLL
    FUNCTYPE = ctypes.WINFUNCTYPE
else:
    _LOADER = ctypes.CDLL
    FUNCTYPE = ctypes.CFUNCTYPE

#: Interface version these declarations describe.
HEADER_VERSION = (13, 1)

#: A mapped picture's address is 64 bits wide in the calls bound here, which
#: is the ABI of a 64-bit process.
SUPPORTED_POINTER_SIZE = ctypes.sizeof(c_void_p) == 8

# --------------------------------------------------------------------------
# enumerations, as plain constants: they cross the ABI as int
# --------------------------------------------------------------------------

CUDA_SUCCESS = 0

CU_MEMORYTYPE_HOST = 1
CU_MEMORYTYPE_DEVICE = 2
CU_MEMORYTYPE_ARRAY = 3

CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 2

CU_GL_DEVICE_LIST_ALL = 1

cudaVideoCodec_H264 = 4
cudaVideoCodec_HEVC = 8
cudaVideoCodec_VP9 = 10
cudaVideoCodec_AV1 = 11

cudaVideoSurfaceFormat_NV12 = 0
cudaVideoSurfaceFormat_P016 = 1

cudaVideoChromaFormat_Monochrome = 0
cudaVideoChromaFormat_420 = 1
cudaVideoChromaFormat_422 = 2
cudaVideoChromaFormat_444 = 3

cudaVideoDeinterlaceMode_Weave = 0
cudaVideoDeinterlaceMode_Bob = 1
cudaVideoDeinterlaceMode_Adaptive = 2

cudaVideoCreate_PreferCUVID = 0x04

CUVID_PKT_ENDOFSTREAM = 0x01
CUVID_PKT_TIMESTAMP = 0x02
CUVID_PKT_DISCONTINUITY = 0x04
CUVID_PKT_ENDOFPICTURE = 0x08

#: The codec names this package uses, as the decoder's enumeration.
CODECS = {'h264': cudaVideoCodec_H264}

CHROMA_FORMAT_NAMES = {
    cudaVideoChromaFormat_Monochrome: 'monochrome',
    cudaVideoChromaFormat_420: '4:2:0',
    cudaVideoChromaFormat_422: '4:2:2',
    cudaVideoChromaFormat_444: '4:4:4',
}


class CUDAError(DecoderError):
    """A CUDA driver or NVDEC call failed.

    Carries the numeric ``CUresult``, the driver's name for it and the call
    that returned it.
    """

    def __init__(self, status: int, call: str, name: str = '', detail: str = '') -> None:
        self.status = status
        self.call = call
        self.name = name or 'CUDA_ERROR'
        message = f'{call} failed: {self.name} ({status})'
        if detail:
            message = f'{message}: {detail}'
        super().__init__(message)


# --------------------------------------------------------------------------
# structures
# --------------------------------------------------------------------------

class CUDA_MEMCPY2D(Structure):
    """One two-dimensional copy between host memory, device memory and arrays."""

    _fields_ = [
        ('srcXInBytes', c_size_t),
        ('srcY', c_size_t),
        ('srcMemoryType', c_int),
        ('srcHost', c_void_p),
        ('srcDevice', c_ulonglong),
        ('srcArray', c_void_p),
        ('srcPitch', c_size_t),
        ('dstXInBytes', c_size_t),
        ('dstY', c_size_t),
        ('dstMemoryType', c_int),
        ('dstHost', c_void_p),
        ('dstDevice', c_ulonglong),
        ('dstArray', c_void_p),
        ('dstPitch', c_size_t),
        ('WidthInBytes', c_size_t),
        ('Height', c_size_t),
    ]


class _FrameRate(Structure):
    _fields_ = [('numerator', c_uint), ('denominator', c_uint)]


class _Rectangle(Structure):
    _fields_ = [('left', c_int), ('top', c_int), ('right', c_int), ('bottom', c_int)]


class _ShortRectangle(Structure):
    _fields_ = [('left', c_short), ('top', c_short), ('right', c_short),
                ('bottom', c_short)]


class _AspectRatio(Structure):
    _fields_ = [('x', c_int), ('y', c_int)]


class _SignalDescription(Structure):
    """The stream's colour description, as H.273 code points."""

    _fields_ = [
        ('video_format', c_ubyte, 3),
        ('video_full_range_flag', c_ubyte, 1),
        ('reserved_zero_bits', c_ubyte, 4),
        ('color_primaries', c_ubyte),
        ('transfer_characteristics', c_ubyte),
        ('matrix_coefficients', c_ubyte),
    ]


class CUVIDEOFORMAT(Structure):
    """What the parser found in a sequence header."""

    _fields_ = [
        ('codec', c_int),
        ('frame_rate', _FrameRate),
        ('progressive_sequence', c_ubyte),
        ('bit_depth_luma_minus8', c_ubyte),
        ('bit_depth_chroma_minus8', c_ubyte),
        ('min_num_decode_surfaces', c_ubyte),
        ('coded_width', c_uint),
        ('coded_height', c_uint),
        ('display_area', _Rectangle),
        ('chroma_format', c_int),
        ('bitrate', c_uint),
        ('display_aspect_ratio', _AspectRatio),
        ('video_signal_description', _SignalDescription),
        ('seqhdr_data_length', c_uint),
    ]


class CUVIDSOURCEDATAPACKET(Structure):
    """A piece of bitstream handed to the parser."""

    _fields_ = [
        ('flags', c_ulong),
        ('payload_size', c_ulong),
        ('payload', c_void_p),
        ('timestamp', c_longlong),
    ]


class CUVIDPARSERDISPINFO(Structure):
    """A picture ready to be shown, in display order."""

    _fields_ = [
        ('picture_index', c_int),
        ('progressive_frame', c_int),
        ('top_field_first', c_int),
        ('repeat_first_field', c_int),
        ('timestamp', c_longlong),
    ]


#: The parser's callbacks. Each returns 0 to fail the parse; the sequence
#: callback returns the number of decode surfaces the parser may cycle through.
#: The picture parameters are passed on to the decoder unread, so they arrive
#: as an address.
SEQUENCE_CALLBACK = FUNCTYPE(c_int, c_void_p, POINTER(CUVIDEOFORMAT))
DECODE_CALLBACK = FUNCTYPE(c_int, c_void_p, c_void_p)
DISPLAY_CALLBACK = FUNCTYPE(c_int, c_void_p, POINTER(CUVIDPARSERDISPINFO))


class CUVIDPARSERPARAMS(Structure):
    _fields_ = [
        ('CodecType', c_int),
        ('ulMaxNumDecodeSurfaces', c_uint),
        ('ulClockRate', c_uint),
        ('ulErrorThreshold', c_uint),
        ('ulMaxDisplayDelay', c_uint),
        ('bAnnexb', c_uint, 1),
        ('bMemoryOptimize', c_uint, 1),
        ('uReserved', c_uint, 30),
        ('uReserved1', c_uint * 4),
        ('pUserData', c_void_p),
        ('pfnSequenceCallback', SEQUENCE_CALLBACK),
        ('pfnDecodePicture', DECODE_CALLBACK),
        ('pfnDisplayPicture', DISPLAY_CALLBACK),
        ('pfnGetOperatingPoint', c_void_p),
        ('pfnGetSEIMsg', c_void_p),
        ('pvReserved2', c_void_p * 5),
        ('pExtVideoInfo', c_void_p),
    ]


class CUVIDDECODECAPS(Structure):
    """What the hardware decodes for one codec, chroma format and bit depth."""

    _fields_ = [
        ('eCodecType', c_int),
        ('eChromaFormat', c_int),
        ('nBitDepthMinus8', c_uint),
        ('reserved1', c_uint * 3),
        ('bIsSupported', c_ubyte),
        ('nNumNVDECs', c_ubyte),
        ('nOutputFormatMask', c_ushort),
        ('nMaxWidth', c_uint),
        ('nMaxHeight', c_uint),
        ('nMaxMBCount', c_uint),
        ('nMinWidth', c_ushort),
        ('nMinHeight', c_ushort),
        ('bIsHistogramSupported', c_ubyte),
        ('nCounterBitDepth', c_ubyte),
        ('nMaxHistogramBins', c_ushort),
        ('bIsDecodeStatsSupported', c_ubyte),
        ('reserved4', c_ubyte * 3),
        ('reserved3', c_uint * 9),
    ]


class CUVIDDECODECREATEINFO(Structure):
    _fields_ = [
        ('ulWidth', c_ulong),
        ('ulHeight', c_ulong),
        ('ulNumDecodeSurfaces', c_ulong),
        ('CodecType', c_int),
        ('ChromaFormat', c_int),
        ('ulCreationFlags', c_ulong),
        ('bitDepthMinus8', c_ulong),
        ('ulIntraDecodeOnly', c_ulong),
        ('ulMaxWidth', c_ulong),
        ('ulMaxHeight', c_ulong),
        ('Reserved1', c_ulong),
        ('display_area', _ShortRectangle),
        ('OutputFormat', c_int),
        ('DeinterlaceMode', c_int),
        ('ulTargetWidth', c_ulong),
        ('ulTargetHeight', c_ulong),
        ('ulNumOutputSurfaces', c_ulong),
        ('vidLock', c_void_p),
        ('target_rect', _ShortRectangle),
        ('enableHistogram', c_ulong),
        ('enableDecodeFeatures', c_ulong),
        ('Reserved2', c_ulong * 3),
    ]


class CUVIDPROCPARAMS(Structure):
    """How a decoded picture is post-processed when it is mapped."""

    _fields_ = [
        ('progressive_frame', c_int),
        ('second_field', c_int),
        ('top_field_first', c_int),
        ('unpaired_field', c_int),
        ('reserved_flags', c_uint),
        ('reserved_zero', c_uint),
        ('raw_input_dptr', c_ulonglong),
        ('raw_input_pitch', c_uint),
        ('raw_input_format', c_uint),
        ('raw_output_dptr', c_ulonglong),
        ('raw_output_pitch', c_uint),
        ('Reserved1', c_uint),
        ('output_stream', c_void_p),
        ('Reserved', c_uint * 46),
        ('histogram_dptr', c_void_p),
        ('pCuvidProcExt', c_void_p),
    ]


# --------------------------------------------------------------------------
# the libraries
# --------------------------------------------------------------------------

def _bound(library: Any, name: str, *arguments: Any, result: Any = c_int) -> Callable[..., Any]:
    function = getattr(library, name)
    function.argtypes = list(arguments)
    function.restype = result
    return function


class Cuda:
    """The CUDA driver API: one instance per process.

    :meth:`load` initialises the driver on first use. Attribute access gives
    the bound calls under the names the headers use, without the ``_v2``
    suffix the exported symbols carry.
    """

    _instance: Cuda | None = None

    def __init__(self, library_name: str = CUDA_LIBRARY_NAME) -> None:
        library = self.library = _LOADER(library_name)
        pointer, handles = POINTER(c_void_p), POINTER(c_void_p)
        self.cuGetErrorName = _bound(library, 'cuGetErrorName', c_int, POINTER(c_char_p))
        self.cuGetErrorString = _bound(library, 'cuGetErrorString', c_int, POINTER(c_char_p))
        self.cuInit = _bound(library, 'cuInit', c_uint)
        self.cuDeviceGetCount = _bound(library, 'cuDeviceGetCount', POINTER(c_int))
        self.cuDeviceGet = _bound(library, 'cuDeviceGet', POINTER(c_int), c_int)
        self.cuDeviceGetName = _bound(library, 'cuDeviceGetName', c_char_p, c_int, c_int)
        self.cuDevicePrimaryCtxRetain = _bound(
            library, 'cuDevicePrimaryCtxRetain', pointer, c_int)
        self.cuCtxPushCurrent = _bound(library, 'cuCtxPushCurrent_v2', c_void_p)
        self.cuCtxPopCurrent = _bound(library, 'cuCtxPopCurrent_v2', pointer)
        self.cuMemcpy2D = _bound(library, 'cuMemcpy2D_v2', POINTER(CUDA_MEMCPY2D))
        self.cuGLGetDevices = _bound(
            library, 'cuGLGetDevices_v2', POINTER(c_uint), POINTER(c_int), c_uint, c_int)
        self.cuGraphicsGLRegisterImage = _bound(
            library, 'cuGraphicsGLRegisterImage', pointer, c_uint, c_uint, c_uint)
        self.cuGraphicsUnregisterResource = _bound(
            library, 'cuGraphicsUnregisterResource', c_void_p)
        self.cuGraphicsMapResources = _bound(
            library, 'cuGraphicsMapResources', c_uint, handles, c_void_p)
        self.cuGraphicsUnmapResources = _bound(
            library, 'cuGraphicsUnmapResources', c_uint, handles, c_void_p)
        self.cuGraphicsSubResourceGetMappedArray = _bound(
            library, 'cuGraphicsSubResourceGetMappedArray', pointer, c_void_p, c_uint, c_uint)
        self._contexts: dict[int, c_void_p] = {}
        self.check(self.cuInit(0), 'cuInit')

    @classmethod
    def load(cls) -> Cuda:
        """The process-wide driver interface, initialised on first use.

        Raises OSError where the library is absent and :class:`CUDAError`
        where the driver will not initialise.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def error_name(self, status: int) -> str:
        """The driver's name for a ``CUresult``, such as ``CUDA_ERROR_INVALID_VALUE``."""
        name = c_char_p()
        if self.cuGetErrorName(status, ctypes.byref(name)) != CUDA_SUCCESS or not name.value:
            return ''
        return name.value.decode('ascii', 'replace')

    def error_text(self, status: int) -> str:
        """The driver's description of a ``CUresult``, or '' where it has none."""
        text = c_char_p()
        if self.cuGetErrorString(status, ctypes.byref(text)) != CUDA_SUCCESS or not text.value:
            return ''
        return text.value.decode('ascii', 'replace')

    def check(self, status: int, call: str, detail: str = '') -> None:
        """Raise :class:`CUDAError` unless `status` is success."""
        if status != CUDA_SUCCESS:
            described = ': '.join(part for part in (self.error_text(status), detail) if part)
            raise CUDAError(status, call, self.error_name(status), described)

    def devices(self) -> list[int]:
        """Every CUDA device in the machine, as device handles."""
        count = c_int(0)
        self.check(self.cuDeviceGetCount(ctypes.byref(count)), 'cuDeviceGetCount')
        found = []
        for ordinal in range(count.value):
            device = c_int(0)
            self.check(self.cuDeviceGet(ctypes.byref(device), ordinal), 'cuDeviceGet')
            found.append(device.value)
        return found

    def gl_devices(self) -> list[int]:
        """The CUDA devices the current OpenGL context renders on.

        Empty where the context belongs to another vendor's GPU or to a
        software rasteriser, which the driver reports as an error or as no
        devices depending on the platform.
        """
        devices = (c_int * 8)()
        count = c_uint(0)
        status = self.cuGLGetDevices(ctypes.byref(count), devices, len(devices),
                                     CU_GL_DEVICE_LIST_ALL)
        if status != CUDA_SUCCESS:
            return []
        return [int(device) for device in devices[:count.value]]

    def device_name(self, device: int) -> str:
        """The marketing name of a device, such as ``NVIDIA GeForce RTX 3060 Ti``."""
        name = ctypes.create_string_buffer(128)
        self.check(self.cuDeviceGetName(name, len(name), device), 'cuDeviceGetName')
        return name.value.decode('utf-8', 'replace')

    def context(self, device: int) -> c_void_p:
        """The device's primary context, retained for the life of the process.

        The primary context is the one every CUDA library in a process shares
        (PyTorch and CuPy use it too), and the driver counts its users. This
        binding stays one of them once a decoder has been opened: making the
        context takes about 100 ms and destroying it 300 ms, measured on an
        RTX 3060 Ti, and a player opened and closed around each movie would
        otherwise pay both every time, on the thread that draws.
        """
        found = self._contexts.get(device)
        if found is None:
            found = c_void_p()
            self.check(self.cuDevicePrimaryCtxRetain(ctypes.byref(found), device),
                       'cuDevicePrimaryCtxRetain')
            self._contexts[device] = found
        return found

    @contextmanager
    def current(self, context: c_void_p) -> Iterator[None]:
        """Make `context` current on this thread for the scope, then put back the one before."""
        self.check(self.cuCtxPushCurrent(context), 'cuCtxPushCurrent')
        try:
            yield
        finally:
            self.cuCtxPopCurrent(ctypes.byref(c_void_p()))


class Cuvid:
    """NVDEC: the bitstream parser and the hardware decoder. One per process."""

    _instance: Cuvid | None = None

    def __init__(self, library_name: str = CUVID_LIBRARY_NAME) -> None:
        library = self.library = _LOADER(library_name)
        pointer = POINTER(c_void_p)
        self.cuvidCreateVideoParser = _bound(
            library, 'cuvidCreateVideoParser', pointer, POINTER(CUVIDPARSERPARAMS))
        self.cuvidParseVideoData = _bound(
            library, 'cuvidParseVideoData', c_void_p, POINTER(CUVIDSOURCEDATAPACKET))
        self.cuvidDestroyVideoParser = _bound(library, 'cuvidDestroyVideoParser', c_void_p)
        self.cuvidGetDecoderCaps = _bound(
            library, 'cuvidGetDecoderCaps', POINTER(CUVIDDECODECAPS))
        self.cuvidCreateDecoder = _bound(
            library, 'cuvidCreateDecoder', pointer, POINTER(CUVIDDECODECREATEINFO))
        self.cuvidDestroyDecoder = _bound(library, 'cuvidDestroyDecoder', c_void_p)
        self.cuvidDecodePicture = _bound(library, 'cuvidDecodePicture', c_void_p, c_void_p)
        self.cuvidMapVideoFrame = _bound(
            library, 'cuvidMapVideoFrame64', c_void_p, c_int, POINTER(c_ulonglong),
            POINTER(c_uint), POINTER(CUVIDPROCPARAMS))
        self.cuvidUnmapVideoFrame = _bound(
            library, 'cuvidUnmapVideoFrame64', c_void_p, c_ulonglong)

    @classmethod
    def load(cls) -> Cuvid:
        """The process-wide decoder interface, loading the library on first use."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
