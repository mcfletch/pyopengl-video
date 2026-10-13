"""Record the C ABI of the NVDEC and CUDA structures, for the binding to be checked against.

The declarations in ``pyopengl_video.nvdec.api`` are hand-written, and a wrong
field order produces a structure the driver reads past the end of, with no
error to see. This tool asks a C compiler what the headers state --
``sizeof`` for each structure, ``offsetof`` for each field the binding
declares, and the value of each constant -- and writes the answers to
``tests/nvdec_abi.json``, which ``tests/test_nvdec_abi.py`` checks the binding
against on machines with no compiler and no NVIDIA driver.

Run it when the binding grows a structure, or when moving to newer headers,
naming the directory that holds them::

    python tools/record_nvdec_abi.py path/to/nv-codec-headers/include/ffnvcodec

The headers are NVIDIA's ``dynlink_cuda.h``, ``dynlink_cuviddec.h`` and
``dynlink_nvcuvid.h``, distributed under the MIT licence in nv-codec-headers,
and are not vendored here: only the facts they state about layout are.
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from pyopengl_video.nvdec import api

#: Structures the binding declares that the headers name. The private ones a
#: structure is built from are anonymous in C, and are measured through the
#: field that holds them.
STRUCTURES = [
    api.CUDA_MEMCPY2D,
    api.CUVIDEOFORMAT,
    api.CUVIDSOURCEDATAPACKET,
    api.CUVIDPARSERDISPINFO,
    api.CUVIDPARSERPARAMS,
    api.CUVIDPICPARAMS,
    api.CUVIDDECODECAPS,
    api.CUVIDDECODECREATEINFO,
    api.CUVIDPROCPARAMS,
]

#: Constants the binding names and the headers define.
CONSTANTS = [
    'CUDA_SUCCESS',
    'CU_MEMORYTYPE_HOST', 'CU_MEMORYTYPE_DEVICE', 'CU_MEMORYTYPE_ARRAY',
    'CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD', 'CU_GL_DEVICE_LIST_ALL',
    'cudaVideoCodec_H264', 'cudaVideoCodec_HEVC', 'cudaVideoCodec_VP9',
    'cudaVideoCodec_AV1',
    'cudaVideoSurfaceFormat_NV12', 'cudaVideoSurfaceFormat_P016',
    'cudaVideoChromaFormat_Monochrome', 'cudaVideoChromaFormat_420',
    'cudaVideoChromaFormat_422', 'cudaVideoChromaFormat_444',
    'cudaVideoDeinterlaceMode_Weave', 'cudaVideoDeinterlaceMode_Bob',
    'cudaVideoDeinterlaceMode_Adaptive',
    'cudaVideoCreate_PreferCUVID',
    'CUVID_PKT_ENDOFSTREAM', 'CUVID_PKT_TIMESTAMP', 'CUVID_PKT_DISCONTINUITY',
    'CUVID_PKT_ENDOFPICTURE',
]

HEADERS = ('dynlink_cuda.h', 'dynlink_cuviddec.h', 'dynlink_nvcuvid.h')


def addressable_fields(structure: type, prefix: str = '') -> list[str]:
    """Every field of `structure` that C can take the offset of, nested ones dotted.

    A bit-field has no address; the binding's ``_fields_`` entries carry a
    third element for those, which is how they are told apart here.
    """
    found = []
    for entry in structure._fields_:  # type: ignore[attr-defined]  # every ctypes Structure has them
        if len(entry) != 2:
            continue
        name, kind = entry
        found.append(prefix + name)
        if isinstance(kind, type) and issubclass(kind, ctypes.Structure):
            found += addressable_fields(kind, f'{prefix}{name}.')
    return found


def offset_of(structure: type, path: str) -> int:
    """Where the binding puts the field at dotted `path` within `structure`."""
    offset = 0
    for name in path.split('.'):
        offset += getattr(structure, name).offset
        structure = dict((entry[0], entry[1]) for entry in structure._fields_)[name]  # type: ignore[attr-defined]  # every ctypes Structure has them
    return offset


def c_program() -> str:
    """A C program printing every size, offset and constant value wanted."""
    lines = ['#include <stdio.h>', '#include <stddef.h>']
    lines += [f'#include "{header}"' for header in HEADERS]
    lines.append('int main(void) {')
    for structure in STRUCTURES:
        name = structure.__name__
        lines.append(f'    printf("size {name} %zu\\n", sizeof({name}));')
        for field in addressable_fields(structure):
            lines.append(
                f'    printf("offset {name} {field} %zu\\n", offsetof({name}, {field}));')
    for constant in CONSTANTS:
        lines.append(
            f'    printf("constant {constant} %lld\\n", (long long)({constant}));')
    lines += ['    return 0;', '}']
    return '\n'.join(lines) + '\n'


#: The two data models a 64-bit driver is built for, and the compiler flags
#: that give each one's layout from this machine's compiler. ``tcu_ulong`` is
#: the headers' ``unsigned long``: eight bytes on Linux and four on Windows.
#: The headers already narrow it to ``unsigned int`` for Cygwin, which is the
#: Windows layout, and ``__stdcall`` is not a word this compiler knows.
MODELS = {
    'lp64': [],
    'llp64': ['-D__CYGWIN__', '-D__stdcall='],
}


def model() -> str:
    """The data model of the running interpreter, as a key of :data:`MODELS`."""
    return 'lp64' if ctypes.sizeof(ctypes.c_long) == 8 else 'llp64'


def measure(include: Path, flags: list[str]) -> tuple[dict, dict, dict]:
    """Compile and run the probe; the sizes, offsets and constants it prints."""
    with tempfile.TemporaryDirectory() as workdir:
        source = Path(workdir) / 'abi.c'
        source.write_text(c_program())
        binary = Path(workdir) / 'abi'
        subprocess.run(
            ['gcc', '-std=c11', *flags, '-I', str(include), str(source), '-o', str(binary)],
            check=True)
        output = subprocess.run([str(binary)], check=True, capture_output=True, text=True)

    sizes: dict[str, int] = {}
    offsets: dict[str, dict[str, int]] = {}
    constants: dict[str, int] = {}
    for line in output.stdout.splitlines():
        kind, *words = line.split(' ')
        if kind == 'size':
            sizes[words[0]] = int(words[1])
        elif kind == 'offset':
            offsets.setdefault(words[0], {})[words[1]] = int(words[2])
        else:
            constants[words[0]] = int(words[1])
    return sizes, offsets, constants


def record(include: Path) -> dict:
    """The ABI the headers in `include` state, for each data model."""
    models = {}
    constants: dict[str, int] = {}
    for name, flags in MODELS.items():
        sizes, offsets, constants = measure(include, flags)
        models[name] = {'sizes': sizes, 'offsets': offsets}
    return {
        'source': ('dynlink_cuda.h, dynlink_cuviddec.h and dynlink_nvcuvid.h '
                   'from nv-codec-headers (MIT)'),
        'header_version': '{}.{}'.format(*api.HEADER_VERSION),
        'models': models,
        'constants': constants,
    }


def disagreements(abi: dict) -> list[str]:
    """What the binding says, on this machine, that the headers do not."""
    layout = abi['models'][model()]
    wrong = [name for name, size in layout['sizes'].items()
             if ctypes.sizeof(getattr(api, name)) != size]
    wrong += [f'{name}.{field}' for name, fields in layout['offsets'].items()
              for field, offset in fields.items()
              if offset_of(getattr(api, name), field) != offset]
    wrong += [f'{name} = {getattr(api, name)}, headers say {value}'
              for name, value in abi['constants'].items()
              if getattr(api, name) != value]
    return wrong


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    abi = record(Path(argv[1]).resolve())
    destination = Path(__file__).parent.parent / 'tests' / 'nvdec_abi.json'
    # Renamed over the recorded table once written, so the suite never reads half of one.
    partial = destination.with_name(destination.name + '.partial')
    partial.write_text(json.dumps(abi, indent=2, sort_keys=True) + '\n')
    partial.replace(destination)
    wrong = disagreements(abi)
    print(f'recorded {len(STRUCTURES)} structures in {len(abi["models"])} data models '
          f'and {len(abi["constants"])} constants to {destination}')
    print(f'binding disagrees on: {wrong}' if wrong else 'binding agrees with the headers')
    return 1 if wrong else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
