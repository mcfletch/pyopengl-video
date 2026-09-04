"""Record the C ABI of the libva structures, for the binding to be checked against.

The declarations in ``pyopengl_video.vaapi.api`` are hand-written, and libva
hands nearly every one of them straight to a driver that reads the fields by
offset. A field declared one word out is not reported: the call succeeds, the
driver reads a frame rate out of a bitrate, and what comes out is a well-formed
stream of the wrong thing. This tool asks a C compiler what the headers really
say -- ``sizeof`` for each structure, ``offsetof`` for each field, and the value
of every constant the binding names -- and writes the answers to
``tests/va_abi.json``, which ``tests/test_va_abi.py`` checks the binding against.

Run it where the libva headers are installed::

    python tools/record_va_abi.py                    # /usr/include
    python tools/record_va_abi.py path/to/include

It needs a C compiler and the headers, neither of which the tests need: the
recording is what travels, so the layout is verified on machines that have
no driver, no GPU and no toolchain.

The headers are Intel's, distributed under the MIT licence, and are not vendored
here -- only the facts they state, which is what ``NOTICES.md`` records.
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from pyopengl_video.vaapi import api

#: Structures the binding declares that have a C counterpart to measure. The
#: private pieces a union or an array is made of are measured through the
#: structure that contains them.
STRUCTURES = [
    api.VAConfigAttrib,
    api.VAGenericValue,
    api.VASurfaceAttrib,
    api.VARectangle,
    api.VACodedBufferSegment,
    api.VAImageFormat,
    api.VAImage,
    api.VAPictureH264,
    api.VAEncSequenceParameterBufferH264,
    api.VAEncPictureParameterBufferH264,
    api.VAEncSliceParameterBufferH264,
    api.VAEncPackedHeaderParameterBuffer,
    api.VAEncMiscParameterRateControl,
    api.VAEncMiscParameterFrameRate,
    api.VAEncMiscParameterHRD,
    api.VAProcColorProperties,
    api.VAProcPipelineParameterBuffer,
    api.VADRMPRIMESurfaceDescriptor,
]

#: Constants the binding names and the headers define. A value that has drifted
#: is as silent a failure as a field at the wrong offset.
CONSTANTS = [
    'VA_STATUS_SUCCESS',
    'VAProfileNone', 'VAProfileH264Main', 'VAProfileH264High',
    'VAProfileH264ConstrainedBaseline',
    'VAEntrypointVLD', 'VAEntrypointEncSlice', 'VAEntrypointEncSliceLP',
    'VAEntrypointVideoProc',
    'VAConfigAttribRTFormat', 'VAConfigAttribRateControl',
    'VAConfigAttribEncPackedHeaders', 'VAConfigAttribEncMaxRefFrames',
    'VA_RT_FORMAT_YUV420', 'VA_RT_FORMAT_RGB32',
    'VA_RC_CBR', 'VA_RC_VBR', 'VA_RC_CQP',
    'VA_ENC_PACKED_HEADER_SEQUENCE', 'VA_ENC_PACKED_HEADER_PICTURE',
    'VA_ENC_PACKED_HEADER_SLICE', 'VA_ENC_PACKED_HEADER_MISC',
    'VA_ENC_PACKED_HEADER_RAW_DATA',
    'VAEncPackedHeaderSequence', 'VAEncPackedHeaderPicture',
    'VAEncPackedHeaderSlice', 'VAEncPackedHeaderRawData',
    'VAEncCodedBufferType', 'VAEncSequenceParameterBufferType',
    'VAEncPictureParameterBufferType', 'VAEncSliceParameterBufferType',
    'VAEncPackedHeaderParameterBufferType', 'VAEncPackedHeaderDataBufferType',
    'VAEncMiscParameterBufferType', 'VAProcPipelineParameterBufferType',
    'VAEncMiscParameterTypeFrameRate', 'VAEncMiscParameterTypeRateControl',
    'VAEncMiscParameterTypeHRD',
    'VASurfaceAttribPixelFormat', 'VASurfaceAttribMemoryType',
    'VASurfaceAttribExternalBufferDescriptor', 'VASurfaceAttribUsageHint',
    'VAGenericValueTypeInteger', 'VAGenericValueTypePointer',
    'VA_SURFACE_ATTRIB_SETTABLE',
    'VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME',
    'VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2',
    'VA_SURFACE_ATTRIB_USAGE_HINT_ENCODER',
    'VA_SURFACE_ATTRIB_USAGE_HINT_VPP_READ',
    'VA_SURFACE_ATTRIB_USAGE_HINT_VPP_WRITE',
    'VA_EXPORT_SURFACE_READ_ONLY', 'VA_EXPORT_SURFACE_WRITE_ONLY',
    'VA_EXPORT_SURFACE_SEPARATE_LAYERS', 'VA_EXPORT_SURFACE_COMPOSED_LAYERS',
    'VA_FOURCC_NV12', 'VA_FOURCC_RGBA', 'VA_FOURCC_BGRA', 'VA_FOURCC_ABGR',
    'VA_FOURCC_ARGB',
    'VA_PICTURE_H264_INVALID', 'VA_PICTURE_H264_SHORT_TERM_REFERENCE',
    'VA_PROGRESSIVE', 'VA_FRAME_PICTURE',
    'VA_SOURCE_RANGE_UNKNOWN', 'VA_SOURCE_RANGE_REDUCED', 'VA_SOURCE_RANGE_FULL',
    'VAProcColorStandardNone', 'VAProcColorStandardBT709',
    'VA_CODED_BUF_STATUS_PICTURE_AVE_QP_MASK',
]

#: Fields the binding spells for its own convenience, or that C reaches through
#: a nested structure the binding flattens.
RENAMED: dict[tuple[str, str], str] = {
    ('VADRMPRIMESurfaceDescriptor', 'objects'): 'objects[0]',
    ('VADRMPRIMESurfaceDescriptor', 'layers'): 'layers[0]',
}

#: Fields with no C counterpart of that name to take an offset of.
SKIPPED: set[tuple[str, str]] = set()

HEADERS = ('va/va.h', 'va/va_vpp.h', 'va/va_enc_h264.h', 'va/va_drmcommon.h',
           'va/va_drm.h')


def fields_of(structure: type) -> list[tuple[str, str]]:
    """Each field the binding declares, with the expression C reaches it by."""
    found = []
    for entry in structure._fields_:
        name = entry[0]
        key = (structure.__name__, name)
        if key in SKIPPED:
            continue
        found.append((name, RENAMED.get(key, name)))
    return found


def c_program() -> str:
    """A C program printing every size, offset and constant value wanted."""
    lines = ['#include <stdio.h>', '#include <stddef.h>']
    lines += [f'#include <{header}>' for header in HEADERS]
    lines.append('int main(void) {')
    for structure in STRUCTURES:
        name = structure.__name__
        lines.append(f'    printf("size {name} %zu\\n", sizeof({name}));')
        for field, expression in fields_of(structure):
            lines.append(
                f'    printf("offset {name}.{field} %zu\\n", '
                f'offsetof({name}, {expression}));')
    for constant in CONSTANTS:
        lines.append(
            f'    printf("constant {constant} %lld\\n", (long long)({constant}));')
    lines += ['    return 0;', '}']
    return '\n'.join(lines) + '\n'


def record(include: Path | None = None) -> dict:
    """Compile and run the probe, against the headers in `include` if given."""
    with tempfile.TemporaryDirectory() as workdir:
        source = Path(workdir) / 'abi.c'
        source.write_text(c_program())
        binary = Path(workdir) / 'abi'
        command = ['gcc', '-std=c11']
        if include is not None:
            command += ['-I', str(include)]
        command += [str(source), '-o', str(binary)]
        subprocess.run(command, check=True)
        output = subprocess.run([str(binary)], check=True, capture_output=True,
                                text=True)

    sizes: dict[str, int] = {}
    offsets: dict[str, dict[str, int]] = {}
    constants: dict[str, int] = {}
    for line in output.stdout.splitlines():
        kind, name, value = line.split(' ')
        if kind == 'size':
            sizes[name] = int(value)
        elif kind == 'offset':
            structure, _, field = name.partition('.')
            offsets.setdefault(structure, {})[field] = int(value)
        else:
            constants[name] = int(value)
    return {
        'source': 'the libva headers from intel/libva (MIT)',
        'sizes': sizes,
        'offsets': offsets,
        'constants': constants,
    }


def disagreements(abi: dict) -> list[str]:
    """What the binding says that the headers do not."""
    wrong = [name for name, size in abi['sizes'].items()
             if ctypes.sizeof(getattr(api, name)) != size]
    wrong += [f'{name} = {getattr(api, name)}, headers say {value}'
              for name, value in abi['constants'].items()
              if getattr(api, name) != value]
    return wrong


def main(argv: list[str]) -> int:
    if len(argv) > 2:
        print(__doc__)
        return 2
    abi = record(Path(argv[1]).resolve() if len(argv) == 2 else None)
    destination = Path(__file__).parent.parent / 'tests' / 'va_abi.json'
    destination.write_text(json.dumps(abi, indent=2, sort_keys=True) + '\n')
    wrong = disagreements(abi)
    print(f'recorded {len(abi["sizes"])} structures and '
          f'{len(abi["constants"])} constants to {destination}')
    for entry in wrong:
        print(f'  binding disagrees: {entry}')
    if not wrong:
        print('binding agrees with the headers')
    return 1 if wrong else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
