"""ctypes declarations for NvEncodeAPI, NVIDIA's video encoder interface.

``libnvidia-encode.so.1`` exports two entry points. ``NvEncodeAPIGetMaxSupportedVersion``
says how new an interface the installed driver understands, and
``NvEncodeAPICreateInstance`` fills in a table of function pointers; every other
call goes through that table. :class:`NvEncodeAPI` wraps both, and binds the
handful of calls this package makes.

Every structure carries a ``version`` field encoding both the interface version
and a per-structure revision, and the driver rejects a structure whose version
it does not recognise. :meth:`NvEncodeAPI.struct_version` builds those, using the
lower of the version described here and the version the driver reports, so a
client built against a newer header still talks to an older driver.

The layouts, enumerations and GUIDs describe the ABI published in
``nvEncodeAPI.h`` version 13.1, as distributed in nv-codec-headers under the MIT
licence -- see NOTICES.md. ``tests/test_nvenc_abi.py`` checks the declarations
here against sizes and offsets recorded from that header.
"""
from __future__ import annotations

import ctypes
import logging
import sys
from collections.abc import Callable
from ctypes import (
    POINTER,
    Structure,
    Union,
    c_char_p,
    c_int,
    c_int8,
    c_int32,
    c_uint8,
    c_uint16,
    c_uint32,
    c_uint64,
    c_void_p,
)
from typing import TYPE_CHECKING

log = logging.getLogger(__name__)

def library_name() -> str:
    """The driver's encoder library on this platform."""
    if sys.platform == 'win32':
        return 'nvEncodeAPI64.dll' if ctypes.sizeof(c_void_p) == 8 else 'nvEncodeAPI.dll'
    return 'libnvidia-encode.so.1'


LIBRARY_NAME = library_name()

# The interface is __stdcall on Windows and cdecl everywhere else. The two are
# the same ABI on 64-bit Windows, so this only bites a 32-bit build -- which is
# reason enough to get it right rather than to explain it later.
_LOADER = ctypes.WinDLL if sys.platform == 'win32' else ctypes.CDLL  # type: ignore[attr-defined]
FUNCTYPE = ctypes.WINFUNCTYPE if sys.platform == 'win32' else ctypes.CFUNCTYPE  # type: ignore[attr-defined]

#: Interface version these declarations describe.
HEADER_VERSION = (13, 1)

# --------------------------------------------------------------------------
# enumerations, as plain constants: they cross the ABI as uint32
# --------------------------------------------------------------------------

NV_ENC_DEVICE_TYPE_DIRECTX = 0x0
NV_ENC_DEVICE_TYPE_CUDA = 0x1
NV_ENC_DEVICE_TYPE_OPENGL = 0x2

NV_ENC_INPUT_RESOURCE_TYPE_OPENGL_TEX = 0x3

NV_ENC_BUFFER_FORMAT_UNDEFINED = 0x00000000
NV_ENC_BUFFER_FORMAT_NV12 = 0x00000001
NV_ENC_BUFFER_FORMAT_ARGB = 0x01000000
NV_ENC_BUFFER_FORMAT_ABGR = 0x10000000

NV_ENC_BUFFER_USAGE_INPUT_IMAGE = 0x0

NV_ENC_PARAMS_RC_CONSTQP = 0x0
NV_ENC_PARAMS_RC_VBR = 0x1
NV_ENC_PARAMS_RC_CBR = 0x2

NV_ENC_PIC_STRUCT_FRAME = 0x01

NV_ENC_PIC_FLAG_FORCEINTRA = 0x1
NV_ENC_PIC_FLAG_FORCEIDR = 0x2
NV_ENC_PIC_FLAG_OUTPUT_SPSPPS = 0x4
NV_ENC_PIC_FLAG_EOS = 0x8

NV_ENC_TUNING_INFO_UNDEFINED = 0
NV_ENC_TUNING_INFO_HIGH_QUALITY = 1
NV_ENC_TUNING_INFO_LOW_LATENCY = 2
NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY = 3
NV_ENC_TUNING_INFO_LOSSLESS = 4
NV_ENC_TUNING_INFO_ULTRA_HIGH_QUALITY = 5

#: A group of pictures that never ends: only the first frame is an IDR.
NVENC_INFINITE_GOPLENGTH = 0xffffffff

#: Picture types, as reported on a locked bitstream.
NV_ENC_PIC_TYPE_P = 0x0
NV_ENC_PIC_TYPE_B = 0x01
NV_ENC_PIC_TYPE_I = 0x02
NV_ENC_PIC_TYPE_IDR = 0x03

#: Status codes, in declaration order, so index == value.
STATUS_NAMES = (
    'SUCCESS', 'ERR_NO_ENCODE_DEVICE', 'ERR_UNSUPPORTED_DEVICE',
    'ERR_INVALID_ENCODERDEVICE', 'ERR_INVALID_DEVICE', 'ERR_DEVICE_NOT_EXIST',
    'ERR_INVALID_PTR', 'ERR_INVALID_EVENT', 'ERR_INVALID_PARAM',
    'ERR_INVALID_CALL', 'ERR_OUT_OF_MEMORY', 'ERR_ENCODER_NOT_INITIALIZED',
    'ERR_UNSUPPORTED_PARAM', 'ERR_LOCK_BUSY', 'ERR_NOT_ENOUGH_BUFFER',
    'ERR_INVALID_VERSION', 'ERR_MAP_FAILED', 'ERR_NEED_MORE_INPUT',
    'ERR_ENCODER_BUSY', 'ERR_EVENT_NOT_REGISTERD', 'ERR_GENERIC',
    'ERR_INCOMPATIBLE_CLIENT_KEY', 'ERR_UNIMPLEMENTED',
    'ERR_RESOURCE_REGISTER_FAILED', 'ERR_RESOURCE_NOT_REGISTERED',
    'ERR_RESOURCE_NOT_MAPPED', 'ERR_NEED_MORE_OUTPUT',
)
NV_ENC_SUCCESS = 0
NV_ENC_ERR_NEED_MORE_INPUT = STATUS_NAMES.index('ERR_NEED_MORE_INPUT')


class NVENCError(RuntimeError):
    """An NvEncodeAPI call failed.

    Carries the numeric status, its name, and whatever the driver put in its
    last-error string for the session.
    """

    def __init__(self, status: int, call: str, detail: str = ''):
        self.status = status
        self.call = call
        name = STATUS_NAMES[status] if 0 <= status < len(STATUS_NAMES) else 'UNKNOWN'
        self.name = name
        message = f'{call} failed: NV_ENC_{name} ({status})'
        if detail:
            message = f'{message}: {detail}'
        super().__init__(message)


# --------------------------------------------------------------------------
# structures
# --------------------------------------------------------------------------

class GUID(Structure):
    """The interface's 16-byte identifier, for codecs, profiles and presets."""

    _fields_ = [
        ('Data1', c_uint32), ('Data2', c_uint16), ('Data3', c_uint16),
        ('Data4', c_uint8 * 8),
    ]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GUID):
            return NotImplemented
        return bytes(memoryview(self)) == bytes(memoryview(other))

    def __hash__(self) -> int:
        return hash(bytes(memoryview(self)))


def _guid(data1: int, data2: int, data3: int, *data4: int) -> GUID:
    return GUID(data1, data2, data3, (c_uint8 * 8)(*data4))


NV_ENC_CODEC_H264_GUID = _guid(
    0x6bc82762, 0x4e63, 0x4ca4, 0xaa, 0x85, 0x1e, 0x50, 0xf3, 0x21, 0xf6, 0xbf)
NV_ENC_CODEC_HEVC_GUID = _guid(
    0x790cdc88, 0x4522, 0x4d7b, 0x94, 0x25, 0xbd, 0xa9, 0x97, 0x5f, 0x76, 0x03)
NV_ENC_H264_PROFILE_HIGH_GUID = _guid(
    0xe7cbc309, 0x4f7a, 0x4b89, 0xaf, 0x2a, 0xd5, 0x37, 0xc9, 0x2b, 0xe3, 0x10)

#: Presets P1 (fastest) to P7 (best quality); the tuning chosen alongside them
#: decides what "quality" is being traded against.
PRESET_GUIDS = {
    'p1': _guid(0xfc0a8d3e, 0x45f8, 0x4cf8, 0x80, 0xc7, 0x29, 0x88, 0x71, 0x59, 0x0e, 0xbf),
    'p2': _guid(0xf581cfb8, 0x88d6, 0x4381, 0x93, 0xf0, 0xdf, 0x13, 0xf9, 0xc2, 0x7d, 0xab),
    'p3': _guid(0x36850110, 0x3a07, 0x441f, 0x94, 0xd5, 0x36, 0x70, 0x63, 0x1f, 0x91, 0xf6),
    'p4': _guid(0x90a7b826, 0xdf06, 0x4862, 0xb9, 0xd2, 0xcd, 0x6d, 0x73, 0xa0, 0x86, 0x81),
    'p5': _guid(0x21c6e6b4, 0x297a, 0x4cba, 0x99, 0x8f, 0xb6, 0xcb, 0xde, 0x72, 0xad, 0xe3),
    'p6': _guid(0x8e75c279, 0x6299, 0x4ab6, 0x83, 0x02, 0x0b, 0x21, 0x5a, 0x33, 0x5c, 0xf5),
    'p7': _guid(0x84848c12, 0x6f71, 0x4c13, 0x93, 0x1b, 0x53, 0xe2, 0x83, 0xf5, 0x79, 0x74),
}

TUNING_INFO = {
    'high_quality': NV_ENC_TUNING_INFO_HIGH_QUALITY,
    'low_latency': NV_ENC_TUNING_INFO_LOW_LATENCY,
    'ultra_low_latency': NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY,
    'lossless': NV_ENC_TUNING_INFO_LOSSLESS,
    'ultra_high_quality': NV_ENC_TUNING_INFO_ULTRA_HIGH_QUALITY,
}

CODEC_GUIDS = {'h264': NV_ENC_CODEC_H264_GUID, 'hevc': NV_ENC_CODEC_HEVC_GUID}


class NV_ENC_QP(Structure):
    _fields_ = [('qpInterP', c_uint32), ('qpInterB', c_uint32), ('qpIntra', c_uint32)]


class NV_ENC_CAPS_PARAM(Structure):
    _fields_ = [('version', c_uint32), ('capsToQuery', c_uint32),
                ('reserved', c_uint32 * 62)]


class NV_ENC_RC_PARAMS(Structure):
    _fields_ = [
        ('version', c_uint32),
        ('rateControlMode', c_uint32),
        ('constQP', NV_ENC_QP),
        ('averageBitRate', c_uint32),
        ('maxBitRate', c_uint32),
        ('vbvBufferSize', c_uint32),
        ('vbvInitialDelay', c_uint32),
        ('enableMinQP', c_uint32, 1),
        ('enableMaxQP', c_uint32, 1),
        ('enableInitialRCQP', c_uint32, 1),
        ('enableAQ', c_uint32, 1),
        ('reservedBitField1', c_uint32, 1),
        ('enableLookahead', c_uint32, 1),
        ('disableIadapt', c_uint32, 1),
        ('disableBadapt', c_uint32, 1),
        ('enableTemporalAQ', c_uint32, 1),
        ('zeroReorderDelay', c_uint32, 1),
        ('enableNonRefP', c_uint32, 1),
        ('strictGOPTarget', c_uint32, 1),
        ('aqStrength', c_uint32, 4),
        ('enableExtLookahead', c_uint32, 1),
        ('reservedBitFields', c_uint32, 15),
        ('minQP', NV_ENC_QP),
        ('maxQP', NV_ENC_QP),
        ('initialRCQP', NV_ENC_QP),
        ('temporallayerIdxMask', c_uint32),
        ('temporalLayerQP', c_uint8 * 8),
        ('targetQuality', c_uint8),
        ('targetQualityLSB', c_uint8),
        ('lookaheadDepth', c_uint16),
        ('lowDelayKeyFrameScale', c_uint8),
        ('yDcQPIndexOffset', c_int8),
        ('uDcQPIndexOffset', c_int8),
        ('vDcQPIndexOffset', c_int8),
        ('qpMapMode', c_uint32),
        ('multiPass', c_uint32),
        ('alphaLayerBitrateRatio', c_uint32),
        ('cbQPIndexOffset', c_int8),
        ('crQPIndexOffset', c_int8),
        ('reserved2', c_uint16),
        ('lookaheadLevel', c_uint32),
        ('viewBitrateRatios', c_uint8 * 7),
        ('reserved3', c_uint8),
        ('reserved1', c_uint32),
    ]


class NV_ENC_CONFIG_H264_VUI_PARAMETERS(Structure):
    _fields_ = [
        ('overscanInfoPresentFlag', c_uint32),
        ('overscanInfo', c_uint32),
        ('videoSignalTypePresentFlag', c_uint32),
        ('videoFormat', c_uint32),
        ('videoFullRangeFlag', c_uint32),
        ('colourDescriptionPresentFlag', c_uint32),
        ('colourPrimaries', c_uint32),
        ('transferCharacteristics', c_uint32),
        ('colourMatrix', c_uint32),
        ('chromaSampleLocationFlag', c_uint32),
        ('chromaSampleLocationTop', c_uint32),
        ('chromaSampleLocationBot', c_uint32),
        ('bitstreamRestrictionFlag', c_uint32),
        ('timingInfoPresentFlag', c_uint32),
        ('numUnitInTicks', c_uint32),
        ('timeScale', c_uint32),
        ('reserved', c_uint32 * 12),
    ]


class NV_ENC_CONFIG_H264(Structure):
    _fields_ = [
        ('enableTemporalSVC', c_uint32, 1),
        ('enableStereoMVC', c_uint32, 1),
        ('hierarchicalPFrames', c_uint32, 1),
        ('hierarchicalBFrames', c_uint32, 1),
        ('outputBufferingPeriodSEI', c_uint32, 1),
        ('outputPictureTimingSEI', c_uint32, 1),
        ('outputAUD', c_uint32, 1),
        ('disableSPSPPS', c_uint32, 1),
        ('outputFramePackingSEI', c_uint32, 1),
        ('outputRecoveryPointSEI', c_uint32, 1),
        ('enableIntraRefresh', c_uint32, 1),
        ('enableConstrainedEncoding', c_uint32, 1),
        ('repeatSPSPPS', c_uint32, 1),
        ('enableVFR', c_uint32, 1),
        ('enableLTR', c_uint32, 1),
        ('qpPrimeYZeroTransformBypassFlag', c_uint32, 1),
        ('useConstrainedIntraPred', c_uint32, 1),
        ('enableFillerDataInsertion', c_uint32, 1),
        ('disableSVCPrefixNalu', c_uint32, 1),
        ('enableScalabilityInfoSEI', c_uint32, 1),
        ('singleSliceIntraRefresh', c_uint32, 1),
        ('enableTimeCode', c_uint32, 1),
        ('reservedBitFields', c_uint32, 10),
        ('level', c_uint32),
        ('idrPeriod', c_uint32),
        ('separateColourPlaneFlag', c_uint32),
        ('disableDeblockingFilterIDC', c_uint32),
        ('numTemporalLayers', c_uint32),
        ('spsId', c_uint32),
        ('ppsId', c_uint32),
        ('adaptiveTransformMode', c_uint32),
        ('fmoMode', c_uint32),
        ('bdirectMode', c_uint32),
        ('entropyCodingMode', c_uint32),
        ('stereoMode', c_uint32),
        ('intraRefreshPeriod', c_uint32),
        ('intraRefreshCnt', c_uint32),
        ('maxNumRefFrames', c_uint32),
        ('sliceMode', c_uint32),
        ('sliceModeData', c_uint32),
        ('h264VUIParameters', NV_ENC_CONFIG_H264_VUI_PARAMETERS),
        ('ltrNumFrames', c_uint32),
        ('ltrTrustMode', c_uint32),
        ('chromaFormatIDC', c_uint32),
        ('maxTemporalLayers', c_uint32),
        ('useBFramesAsRef', c_uint32),
        ('numRefL0', c_uint32),
        ('numRefL1', c_uint32),
        ('outputBitDepth', c_uint32),
        ('inputBitDepth', c_uint32),
        ('tfLevel', c_uint32),
        ('reserved1', c_uint32 * 264),
        ('reserved2', c_void_p * 64),
    ]


class NV_ENC_CODEC_CONFIG(Union):
    """Only the H.264 arm is declared; the reserved array fixes the size.

    A union is as large as its largest member, and the header gives that member
    as a 320-word reserved array, so a codec whose configuration this package
    does not yet write still occupies the right space.
    """

    _fields_ = [('h264Config', NV_ENC_CONFIG_H264), ('reserved', c_uint32 * 320)]


class NV_ENC_CONFIG(Structure):
    _fields_ = [
        ('version', c_uint32),
        ('profileGUID', GUID),
        ('gopLength', c_uint32),
        ('frameIntervalP', c_int32),
        ('monoChromeEncoding', c_uint32),
        ('frameFieldMode', c_uint32),
        ('mvPrecision', c_uint32),
        ('rcParams', NV_ENC_RC_PARAMS),
        ('encodeCodecConfig', NV_ENC_CODEC_CONFIG),
        ('reserved', c_uint32 * 278),
        ('reserved2', c_void_p * 64),
    ]


class NV_ENC_PRESET_CONFIG(Structure):
    _fields_ = [
        ('version', c_uint32), ('reserved', c_uint32),
        ('presetCfg', NV_ENC_CONFIG),
        ('reserved1', c_uint32 * 256), ('reserved2', c_void_p * 64),
    ]


class NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE(Structure):
    _fields_ = [
        ('numCandsPerBlk16x16', c_uint32, 4),
        ('numCandsPerBlk16x8', c_uint32, 4),
        ('numCandsPerBlk8x16', c_uint32, 4),
        ('numCandsPerBlk8x8', c_uint32, 4),
        ('numCandsPerSb', c_uint32, 8),
        ('reserved', c_uint32, 8),
        ('reserved1', c_uint32 * 3),
    ]


class NV_ENC_INITIALIZE_PARAMS(Structure):
    _fields_ = [
        ('version', c_uint32),
        ('encodeGUID', GUID),
        ('presetGUID', GUID),
        ('encodeWidth', c_uint32),
        ('encodeHeight', c_uint32),
        ('darWidth', c_uint32),
        ('darHeight', c_uint32),
        ('frameRateNum', c_uint32),
        ('frameRateDen', c_uint32),
        ('enableEncodeAsync', c_uint32),
        ('enablePTD', c_uint32),
        ('reportSliceOffsets', c_uint32, 1),
        ('enableSubFrameWrite', c_uint32, 1),
        ('enableExternalMEHints', c_uint32, 1),
        ('enableMEOnlyMode', c_uint32, 1),
        ('enableWeightedPrediction', c_uint32, 1),
        ('splitEncodeMode', c_uint32, 4),
        ('enableOutputInVidmem', c_uint32, 1),
        ('enableReconFrameOutput', c_uint32, 1),
        ('enableOutputStats', c_uint32, 1),
        ('enableUniDirectionalB', c_uint32, 1),
        ('reservedBitFields', c_uint32, 19),
        ('privDataSize', c_uint32),
        ('reserved', c_uint32),
        ('privData', c_void_p),
        ('encodeConfig', POINTER(NV_ENC_CONFIG)),
        ('maxEncodeWidth', c_uint32),
        ('maxEncodeHeight', c_uint32),
        ('maxMEHintCountsPerBlock', NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE * 2),
        ('tuningInfo', c_uint32),
        ('bufferFormat', c_uint32),
        ('numStateBuffers', c_uint32),
        ('outputStatsLevel', c_uint32),
        ('reserved1', c_uint32 * 284),
        ('reserved2', c_void_p * 64),
    ]


class NV_ENC_CREATE_BITSTREAM_BUFFER(Structure):
    _fields_ = [
        ('version', c_uint32), ('size', c_uint32), ('memoryHeap', c_uint32),
        ('reserved', c_uint32), ('bitstreamBuffer', c_void_p),
        ('bitstreamBufferPtr', c_void_p),
        ('reserved1', c_uint32 * 58), ('reserved2', c_void_p * 64),
    ]


class NV_ENC_INPUT_RESOURCE_OPENGL_TEX(Structure):
    """The name and target of the texture being handed to the encoder."""

    _fields_ = [('texture', c_uint32), ('target', c_uint32)]


class NV_ENC_REGISTER_RESOURCE(Structure):
    _fields_ = [
        ('version', c_uint32),
        ('resourceType', c_uint32),
        ('width', c_uint32),
        ('height', c_uint32),
        ('pitch', c_uint32),
        ('subResourceIndex', c_uint32),
        ('resourceToRegister', c_void_p),
        ('registeredResource', c_void_p),
        ('bufferFormat', c_uint32),
        ('bufferUsage', c_uint32),
        ('pInputFencePoint', c_void_p),
        ('chromaOffset', c_uint32 * 2),
        ('chromaOffsetIn', c_uint32 * 2),
        ('reserved1', c_uint32 * 244),
        ('reserved2', c_void_p * 61),
    ]


class NV_ENC_MAP_INPUT_RESOURCE(Structure):
    _fields_ = [
        ('version', c_uint32), ('subResourceIndex', c_uint32),
        ('inputResource', c_void_p), ('registeredResource', c_void_p),
        ('mappedResource', c_void_p), ('mappedBufferFmt', c_uint32),
        ('reserved1', c_uint32 * 251), ('reserved2', c_void_p * 63),
    ]


class NV_ENC_CODEC_PIC_PARAMS(Union):
    """Per-picture codec parameters, for a codec whose per-picture controls this
    package does not use.

    Declared as a reservation rather than as its arms, because what the
    surrounding structure needs from it is its size and its alignment: the H.264
    arm holds pointers, so the union aligns to 8 and is larger than a count of
    its reserved words suggests. The recorded ABI in ``tests/nvenc_abi.json``
    is what holds both to the header.
    """

    _fields_ = [('reserved', c_uint64 * 193)]


class NV_ENC_PIC_PARAMS(Structure):
    _fields_ = [
        ('version', c_uint32),
        ('inputWidth', c_uint32),
        ('inputHeight', c_uint32),
        ('inputPitch', c_uint32),
        ('encodePicFlags', c_uint32),
        ('frameIdx', c_uint32),
        ('inputTimeStamp', c_uint64),
        ('inputDuration', c_uint64),
        ('inputBuffer', c_void_p),
        ('outputBitstream', c_void_p),
        ('completionEvent', c_void_p),
        ('bufferFmt', c_uint32),
        ('pictureStruct', c_uint32),
        ('pictureType', c_uint32),
        ('codecPicParams', NV_ENC_CODEC_PIC_PARAMS),
        ('meHintCountsPerBlock', NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE * 2),
        ('meExternalHints', c_void_p),
        ('reserved2', c_uint32 * 7),
        ('reserved5', c_void_p * 2),
        ('qpDeltaMap', POINTER(c_int8)),
        ('qpDeltaMapSize', c_uint32),
        ('reservedBitFields', c_uint32),
        ('meHintRefPicDist', c_uint16 * 2),
        ('diffPicNumHint', c_int32),
        ('alphaBuffer', c_void_p),
        ('meExternalSbHints', c_void_p),
        ('meSbHintsCount', c_uint32),
        ('stateBufferIdx', c_uint32),
        ('outputReconBuffer', c_void_p),
        ('reserved3', c_uint32 * 284),
        ('reserved6', c_void_p * 57),
    ]


class NV_ENC_LOCK_BITSTREAM(Structure):
    _fields_ = [
        ('version', c_uint32),
        ('doNotWait', c_uint32, 1),
        ('ltrFrame', c_uint32, 1),
        ('getRCStats', c_uint32, 1),
        ('reservedBitFields', c_uint32, 29),
        ('outputBitstream', c_void_p),
        ('sliceOffsets', POINTER(c_uint32)),
        ('frameIdx', c_uint32),
        ('hwEncodeStatus', c_uint32),
        ('numSlices', c_uint32),
        ('bitstreamSizeInBytes', c_uint32),
        ('outputTimeStamp', c_uint64),
        ('outputDuration', c_uint64),
        ('bitstreamBufferPtr', c_void_p),
        ('pictureType', c_uint32),
        ('pictureStruct', c_uint32),
        ('frameAvgQP', c_uint32),
        ('frameSatd', c_uint32),
        ('ltrFrameIdx', c_uint32),
        ('ltrFrameBitmap', c_uint32),
        ('temporalId', c_uint32),
        ('intraMBCount', c_uint32),
        ('interMBCount', c_uint32),
        ('averageMVX', c_int32),
        ('averageMVY', c_int32),
        ('alphaLayerSizeInBytes', c_uint32),
        ('outputStatsPtrSize', c_uint32),
        ('reserved', c_uint32),
        ('outputStatsPtr', c_void_p),
        ('frameIdxDisplay', c_uint32),
        ('reserved1', c_uint32 * 219),
        ('reserved2', c_void_p * 63),
        ('reservedInternal', c_uint32 * 8),
    ]


class NV_ENC_SEQUENCE_PARAM_PAYLOAD(Structure):
    _fields_ = [
        ('version', c_uint32), ('inBufferSize', c_uint32),
        ('spsId', c_uint32), ('ppsId', c_uint32),
        ('spsppsBuffer', c_void_p), ('outSPSPPSPayloadSize', POINTER(c_uint32)),
        ('reserved', c_uint32 * 250), ('reserved2', c_void_p * 64),
    ]


class NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS(Structure):
    _fields_ = [
        ('version', c_uint32), ('deviceType', c_uint32),
        ('device', c_void_p), ('reserved', c_void_p),
        ('apiVersion', c_uint32), ('reserved1', c_uint32 * 253),
        ('reserved2', c_void_p * 64),
    ]


#: The function table, in declaration order. Only the entries this package
#: calls are given prototypes; the rest hold the place so the offsets are right.
FUNCTION_NAMES = (
    'nvEncOpenEncodeSession', 'nvEncGetEncodeGUIDCount',
    'nvEncGetEncodeProfileGUIDCount', 'nvEncGetEncodeProfileGUIDs',
    'nvEncGetEncodeGUIDs', 'nvEncGetInputFormatCount', 'nvEncGetInputFormats',
    'nvEncGetEncodeCaps', 'nvEncGetEncodePresetCount', 'nvEncGetEncodePresetGUIDs',
    'nvEncGetEncodePresetConfig', 'nvEncInitializeEncoder', 'nvEncCreateInputBuffer',
    'nvEncDestroyInputBuffer', 'nvEncCreateBitstreamBuffer', 'nvEncDestroyBitstreamBuffer',
    'nvEncEncodePicture', 'nvEncLockBitstream', 'nvEncUnlockBitstream',
    'nvEncLockInputBuffer', 'nvEncUnlockInputBuffer', 'nvEncGetEncodeStats',
    'nvEncGetSequenceParams', 'nvEncRegisterAsyncEvent', 'nvEncUnregisterAsyncEvent',
    'nvEncMapInputResource', 'nvEncUnmapInputResource', 'nvEncDestroyEncoder',
    'nvEncInvalidateRefFrames', 'nvEncOpenEncodeSessionEx', 'nvEncRegisterResource',
    'nvEncUnregisterResource', 'nvEncReconfigureEncoder', 'reserved1',
    'nvEncCreateMVBuffer', 'nvEncDestroyMVBuffer', 'nvEncRunMotionEstimationOnly',
    'nvEncGetLastErrorString', 'nvEncSetIOCudaStreams', 'nvEncGetEncodePresetConfigEx',
    'nvEncGetSequenceParamEx', 'nvEncRestoreEncoderState', 'nvEncLookaheadPicture',
)


class NV_ENCODE_API_FUNCTION_LIST(Structure):
    _fields_ = (
        [('version', c_uint32), ('reserved', c_uint32)]
        + [(name, c_void_p) for name in FUNCTION_NAMES]
        + [('reserved2', c_void_p * 275)]
    )


#: Prototypes for the calls this package makes, by function-table entry name.
PROTOTYPES = {
    'nvEncOpenEncodeSessionEx': FUNCTYPE(
        c_int, POINTER(NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS), POINTER(c_void_p)),
    'nvEncGetEncodeGUIDCount': FUNCTYPE(c_int, c_void_p, POINTER(c_uint32)),
    'nvEncGetEncodeGUIDs': FUNCTYPE(
        c_int, c_void_p, POINTER(GUID), c_uint32, POINTER(c_uint32)),
    'nvEncGetEncodeCaps': FUNCTYPE(
        c_int, c_void_p, GUID, POINTER(NV_ENC_CAPS_PARAM), POINTER(c_int)),
    'nvEncGetEncodePresetConfigEx': FUNCTYPE(
        c_int, c_void_p, GUID, GUID, c_uint32, POINTER(NV_ENC_PRESET_CONFIG)),
    'nvEncInitializeEncoder': FUNCTYPE(
        c_int, c_void_p, POINTER(NV_ENC_INITIALIZE_PARAMS)),
    'nvEncCreateBitstreamBuffer': FUNCTYPE(
        c_int, c_void_p, POINTER(NV_ENC_CREATE_BITSTREAM_BUFFER)),
    'nvEncDestroyBitstreamBuffer': FUNCTYPE(c_int, c_void_p, c_void_p),
    'nvEncRegisterResource': FUNCTYPE(
        c_int, c_void_p, POINTER(NV_ENC_REGISTER_RESOURCE)),
    'nvEncUnregisterResource': FUNCTYPE(c_int, c_void_p, c_void_p),
    'nvEncMapInputResource': FUNCTYPE(
        c_int, c_void_p, POINTER(NV_ENC_MAP_INPUT_RESOURCE)),
    'nvEncUnmapInputResource': FUNCTYPE(c_int, c_void_p, c_void_p),
    'nvEncEncodePicture': FUNCTYPE(c_int, c_void_p, POINTER(NV_ENC_PIC_PARAMS)),
    'nvEncLockBitstream': FUNCTYPE(
        c_int, c_void_p, POINTER(NV_ENC_LOCK_BITSTREAM)),
    'nvEncUnlockBitstream': FUNCTYPE(c_int, c_void_p, c_void_p),
    'nvEncGetSequenceParams': FUNCTYPE(
        c_int, c_void_p, POINTER(NV_ENC_SEQUENCE_PARAM_PAYLOAD)),
    'nvEncDestroyEncoder': FUNCTYPE(c_int, c_void_p),
    'nvEncGetLastErrorString': FUNCTYPE(c_char_p, c_void_p),
}


class NvEncodeAPI:
    """The driver's encoder interface: one instance per process is enough.

    Attribute access gives the bound function-table entries, so a caller writes
    ``api.nvEncEncodePicture(session, params)``. :meth:`check` turns a status
    code into an :class:`NVENCError` carrying the driver's own explanation.
    """

    _instance: NvEncodeAPI | None = None

    if TYPE_CHECKING:
        # Bound in __init__ from PROTOTYPES. Naming them here is what lets a
        # reader, and a type checker, see the interface this class offers.
        nvEncOpenEncodeSessionEx: Callable[..., int]
        nvEncGetEncodeGUIDCount: Callable[..., int]
        nvEncGetEncodeGUIDs: Callable[..., int]
        nvEncGetEncodeCaps: Callable[..., int]
        nvEncGetEncodePresetConfigEx: Callable[..., int]
        nvEncInitializeEncoder: Callable[..., int]
        nvEncCreateBitstreamBuffer: Callable[..., int]
        nvEncDestroyBitstreamBuffer: Callable[..., int]
        nvEncRegisterResource: Callable[..., int]
        nvEncUnregisterResource: Callable[..., int]
        nvEncMapInputResource: Callable[..., int]
        nvEncUnmapInputResource: Callable[..., int]
        nvEncEncodePicture: Callable[..., int]
        nvEncLockBitstream: Callable[..., int]
        nvEncUnlockBitstream: Callable[..., int]
        nvEncGetSequenceParams: Callable[..., int]
        nvEncDestroyEncoder: Callable[..., int]
        nvEncGetLastErrorString: Callable[..., bytes]

    def __init__(self, library_name: str = LIBRARY_NAME):
        self.library = _LOADER(library_name)
        self.driver_version = self._max_supported_version()
        self.version = min(HEADER_VERSION, self.driver_version)
        self.functions = self._create_instance()
        for name, prototype in PROTOTYPES.items():
            setattr(self, name, prototype(getattr(self.functions, name)))

    @classmethod
    def load(cls) -> NvEncodeAPI:
        """The process-wide interface, loading the driver library on first use."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _max_supported_version(self) -> tuple[int, int]:
        """The newest interface version the installed driver understands."""
        self.library.NvEncodeAPIGetMaxSupportedVersion.argtypes = [POINTER(c_uint32)]
        packed = c_uint32()
        status = self.library.NvEncodeAPIGetMaxSupportedVersion(ctypes.byref(packed))
        if status != NV_ENC_SUCCESS:
            raise NVENCError(status, 'NvEncodeAPIGetMaxSupportedVersion')
        return (packed.value >> 4, packed.value & 0xf)

    @property
    def api_version(self) -> int:
        """The interface version, packed as the ``apiVersion`` fields want it."""
        major, minor = self.version
        return major | (minor << 24)

    def struct_version(self, revision: int, big: bool = False) -> int:
        """Build the ``version`` field of a structure at `revision`.

        big -- set for the structures the header marks with the high bit, which
            is how the driver tells the newer, larger layouts apart
        """
        value = self.api_version | (revision << 16) | (0x7 << 28)
        return value | (1 << 31) if big else value

    def _create_instance(self) -> NV_ENCODE_API_FUNCTION_LIST:
        functions = NV_ENCODE_API_FUNCTION_LIST()
        functions.version = self.struct_version(2)
        self.library.NvEncodeAPICreateInstance.argtypes = [
            POINTER(NV_ENCODE_API_FUNCTION_LIST)]
        status = self.library.NvEncodeAPICreateInstance(ctypes.byref(functions))
        if status != NV_ENC_SUCCESS:
            raise NVENCError(status, 'NvEncodeAPICreateInstance')
        return functions

    def check(self, status: int, call: str, session: c_void_p | None = None) -> int:
        """Raise :class:`NVENCError` unless `status` is success.

        Returns the status, so a caller that treats some non-success codes as
        ordinary -- ``NEED_MORE_INPUT``, say -- can check for them first.
        """
        if status == NV_ENC_SUCCESS:
            return status
        detail = ''
        if session is not None:
            message = self.nvEncGetLastErrorString(session)
            if message:
                detail = message.decode('utf-8', 'replace')
        raise NVENCError(status, call, detail)
