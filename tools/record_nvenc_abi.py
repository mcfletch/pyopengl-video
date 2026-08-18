"""Record the C ABI of the NvEncodeAPI structures, for the binding to be checked against.

The declarations in ``pyopengl_video.nvenc.api`` are hand-written, and a wrong
field order or a missed reserved array produces a structure the driver reads
past the end of rather than an error anyone can see. This tool asks a C compiler
what the header really says -- ``sizeof`` for each structure and ``offsetof`` for
each field the binding declares -- and writes the answers to
``tests/nvenc_abi.json``, which ``tests/test_nvenc_abi.py`` checks the binding
against on machines with no compiler and no NVIDIA driver.

Run it when the binding grows a structure, or when moving to a newer header::

    python tools/record_nvenc_abi.py path/to/nvEncodeAPI.h

The header is NVIDIA's, distributed under the MIT licence in nv-codec-headers,
and is not vendored here: only the facts it states about layout are.
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from pyopengl_video.nvenc import api

#: Structures the binding declares, in the order the report reads best.
STRUCTURES = [
    api.GUID,
    api.NV_ENC_QP,
    api.NV_ENC_CAPS_PARAM,
    api.NV_ENC_RC_PARAMS,
    api.NV_ENC_CONFIG_H264_VUI_PARAMETERS,
    api.NV_ENC_CONFIG_H264,
    api.NV_ENC_CODEC_CONFIG,
    api.NV_ENC_CONFIG,
    api.NV_ENC_PRESET_CONFIG,
    api.NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE,
    api.NV_ENC_INITIALIZE_PARAMS,
    api.NV_ENC_CREATE_BITSTREAM_BUFFER,
    api.NV_ENC_INPUT_RESOURCE_OPENGL_TEX,
    api.NV_ENC_REGISTER_RESOURCE,
    api.NV_ENC_MAP_INPUT_RESOURCE,
    api.NV_ENC_CODEC_PIC_PARAMS,
    api.NV_ENC_PIC_PARAMS,
    api.NV_ENC_LOCK_BITSTREAM,
    api.NV_ENC_SEQUENCE_PARAM_PAYLOAD,
    api.NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS,
    api.NV_ENCODE_API_FUNCTION_LIST,
]


def addressable_fields(structure: type) -> list[str]:
    """The fields of `structure` that C can take the offset of.

    A bit-field has no address, so ``offsetof`` on one does not compile; the
    binding's ``_fields_`` entries carry a third element for those, which is how
    they are told apart here.
    """
    return [entry[0] for entry in structure._fields_ if len(entry) == 2]


def c_program() -> str:
    """A C program printing the size and field offsets of every structure."""
    lines = ['#include <stdio.h>', '#include <stddef.h>', '#include "nvEncodeAPI.h"',
             'int main(void) {']
    for structure in STRUCTURES:
        name = structure.__name__
        lines.append(f'    printf("{name} = %zu\\n", sizeof({name}));')
        for field in addressable_fields(structure):
            lines.append(
                f'    printf("{name}.{field} %zu\\n", offsetof({name}, {field}));')
    lines += ['    return 0;', '}']
    return '\n'.join(lines) + '\n'


def record(header: Path) -> dict:
    """Compile and run the probe against `header`, returning the ABI it reports."""
    with tempfile.TemporaryDirectory() as workdir:
        source = Path(workdir) / 'abi.c'
        source.write_text(c_program())
        binary = Path(workdir) / 'abi'
        subprocess.run(
            ['gcc', '-std=c11', '-I', str(header.parent), str(source), '-o', str(binary)],
            check=True)
        output = subprocess.run([str(binary)], check=True, capture_output=True, text=True)

    sizes: dict[str, int] = {}
    offsets: dict[str, dict[str, int]] = {}
    for line in output.stdout.splitlines():
        name, _, value = line.rpartition(' ')
        if name.endswith(' ='):
            sizes[name[:-2]] = int(value)
        else:
            structure, _, field = name.partition('.')
            offsets.setdefault(structure, {})[field] = int(value)
    return {
        'source': 'nvEncodeAPI.h from nv-codec-headers (MIT)',
        'header_version': '{}.{}'.format(*api.HEADER_VERSION),
        'sizes': sizes,
        'offsets': offsets,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    abi = record(Path(argv[1]).resolve())
    destination = Path(__file__).parent.parent / 'tests' / 'nvenc_abi.json'
    destination.write_text(json.dumps(abi, indent=2, sort_keys=True) + '\n')
    disagreements = [
        name for name, size in abi['sizes'].items()
        if ctypes.sizeof(getattr(api, name)) != size
    ]
    print(f'recorded {len(abi["sizes"])} structures to {destination}')
    print(f'binding disagrees on: {disagreements}' if disagreements
          else 'binding agrees with the header')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
