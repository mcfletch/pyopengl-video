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
from typing import Any

from pyopengl_video.vaapi import api, av1_api, hevc_api, vp9_api

#: The modules that declare libva structures and constants. Every public
#: ``ctypes.Structure`` in one is measured, and so is every integer constant
#: whose name the headers also define.
MODULES = (api, hevc_api, av1_api, vp9_api)

#: Names that begin as a header constant does and are this binding's own.
NOT_IN_THE_HEADERS = frozenset({'VA_ATTRIB_NOT_SUPPORTED'})

#: Fields the binding spells for its own convenience, or that C reaches through
#: a nested structure the binding flattens.
RENAMED: dict[tuple[str, str], str] = {
    ('VADRMPRIMESurfaceDescriptor', 'objects'): 'objects[0]',
    ('VADRMPRIMESurfaceDescriptor', 'layers'): 'layers[0]',
}

#: Fields with no C counterpart of that name to take an offset of.
SKIPPED: set[tuple[str, str]] = set()


def structures() -> list[type]:
    """Every public structure the binding declares, each measured once."""
    found: dict[str, type] = {}
    for module in MODULES:
        for name, value in vars(module).items():
            if (isinstance(value, type) and issubclass(value, ctypes.Structure)
                    and not name.startswith('_') and value.__module__ == module.__name__):
                found[name] = value
    return list(found.values())


def constants() -> list[str]:
    """Every integer the binding names the way the headers name theirs."""
    found: list[str] = []
    for module in MODULES:
        for name, value in vars(module).items():
            if (name.startswith('VA') and type(value) is int
                    and name not in NOT_IN_THE_HEADERS and name not in found):
                found.append(name)
    return found


def declared(name: str) -> Any:
    """The binding's declaration of `name`, from whichever module holds it."""
    for module in MODULES:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f'no libva binding module declares {name}')


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
    for structure in structures():
        name = structure.__name__
        lines.append(f'    printf("size {name} %zu\\n", sizeof({name}));')
        for field, expression in fields_of(structure):
            lines.append(
                f'    printf("offset {name}.{field} %zu\\n", '
                f'offsetof({name}, {expression}));')
    for constant in constants():
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
        # The slice group fields are deprecated in the headers and still in
        # the structure, so they are measured like any other.
        command = ['gcc', '-std=c11', '-Wno-deprecated-declarations']
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
             if ctypes.sizeof(declared(name)) != size]
    wrong += [f'{name} = {declared(name)}, headers say {value}'
              for name, value in abi['constants'].items()
              if declared(name) != value]
    return wrong


def main(argv: list[str]) -> int:
    if len(argv) > 2:
        print(__doc__)
        return 2
    abi = record(Path(argv[1]).resolve() if len(argv) == 2 else None)
    destination = Path(__file__).parent.parent / 'tests' / 'va_abi.json'
    # Renamed over the recorded table once written, so the suite never reads half of one.
    partial = destination.with_name(destination.name + '.partial')
    partial.write_text(json.dumps(abi, indent=2, sort_keys=True) + '\n')
    partial.replace(destination)
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
