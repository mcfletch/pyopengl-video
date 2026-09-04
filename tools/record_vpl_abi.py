"""Record the C ABI of the oneVPL structures, for the binding to be checked against.

The declarations in ``pyopengl_video.vpl.api`` are hand-written, and oneVPL packs
each structure to 4 or 8 bytes through its ``MFX_PACK_BEGIN_*`` macros. Get the
packing wrong and the layout shifts silently: the runtime answers
``MFX_ERR_INVALID_VIDEO_PARAM`` to every call taking an ``mfxVideoParam`` and
names nothing at all. This tool asks a C compiler what the headers really say --
``sizeof`` for each structure and ``offsetof`` for each field the binding
declares -- and writes the answers to ``tests/vpl_abi.json``, which
``tests/test_vpl_abi.py`` prefers over its built-in table when it is present.

Run it against a checkout of the oneVPL headers::

    python tools/record_vpl_abi.py path/to/libvpl/api/vpl

It needs a C compiler, which the Windows development machine does not have; run
it on the Linux side, where the same headers describe the same ABI for the parts
this binding uses. The headers are Intel's, distributed under the MIT licence,
and are not vendored here: only the facts they state about layout are.
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from pyopengl_video.vpl import api

#: Structures the binding declares, in the order the report reads best.
STRUCTURES = [
    api.mfxVersion,
    api.mfxFrameInfo,
    api.mfxInfoMFX,
    api.mfxInfoVPP,
    api.mfxVideoParam,
    api.mfxFrameData,
    api.mfxFrameSurface1,
    api.mfxBitstream,
    api.mfxExtBuffer,
    api.mfxExtCodingOptionSPSPPS,
    api.mfxExtVideoSignalInfo,
    api.mfxEncodeCtrl,
    api.mfxFrameAllocRequest,
    api.mfxFrameAllocResponse,
    api.mfxFrameAllocator,
]

#: Fields the binding names for its own convenience, which the C structure
#: reaches through an anonymous union or spells differently. Naming them here
#: keeps the generated program compiling rather than silently skipping them.
RENAMED = {
    ('mfxFrameInfo', 'TemporalId'): 'FrameId.TemporalId',
    ('mfxFrameInfo', 'PriorityId'): 'FrameId.PriorityId',
    ('mfxFrameInfo', 'DependencyId'): 'FrameId.DependencyId',
    ('mfxFrameInfo', 'QualityId'): 'FrameId.QualityId',
    ('mfxVideoParam', 'codec'): 'mfx',
    ('mfxFrameData', 'external'): 'ExtParam',
    ('mfxFrameSurface1', 'interface'): 'FrameInterface',
    ('mfxBitstream', 'header'): 'EncryptedData',
}

#: Fields with no C counterpart to take an offset of at all.
SKIPPED = {('mfxFrameInfo', 'geometry')}


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
    """A C program printing the size and field offsets of every structure."""
    lines = ['#include <stdio.h>', '#include <stddef.h>', '#include "mfxvideo.h"',
             '#include "mfxstructures.h"', 'int main(void) {']
    for structure in STRUCTURES:
        name = structure.__name__
        lines.append(f'    printf("{name} = %zu\\n", sizeof({name}));')
        for field, expression in fields_of(structure):
            lines.append(
                f'    printf("{name}.{field} %zu\\n", '
                f'offsetof({name}, {expression}));')
    lines += ['    return 0;', '}']
    return '\n'.join(lines) + '\n'


def record(include: Path) -> dict:
    """Compile and run the probe against the headers in `include`."""
    with tempfile.TemporaryDirectory() as workdir:
        source = Path(workdir) / 'abi.c'
        source.write_text(c_program())
        binary = Path(workdir) / 'abi'
        subprocess.run(
            ['gcc', '-std=c11', '-I', str(include), str(source), '-o', str(binary)],
            check=True)
        output = subprocess.run([str(binary)], check=True, capture_output=True,
                                text=True)

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
        'source': 'the oneVPL headers from intel/libvpl (MIT)',
        'sizes': sizes,
        'offsets': offsets,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    abi = record(Path(argv[1]).resolve())
    destination = Path(__file__).parent.parent / 'tests' / 'vpl_abi.json'
    destination.write_text(json.dumps(abi, indent=2, sort_keys=True) + '\n')
    disagreements = [
        name for name, size in abi['sizes'].items()
        if ctypes.sizeof(getattr(api, name)) != size
    ]
    print(f'recorded {len(abi["sizes"])} structures to {destination}')
    print(f'binding disagrees on: {disagreements}' if disagreements
          else 'binding agrees with the headers')
    return 1 if disagreements else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
