"""Record the C ABI of the NvEncodeAPI structures, for the binding to be checked against.

The declarations in ``pyopengl_video.nvenc.api`` are hand-written, and a wrong
field order or a missed reserved array produces a structure the driver reads
past the end of, with no error to see. This tool asks a C compiler what the
header states and writes the answers to ``tests/nvenc_abi.json``, which
``tests/test_nvenc_abi.py`` checks the binding against on machines with no
compiler and no NVIDIA driver:

- ``sizes`` -- ``sizeof`` for each structure and union the header names
- ``offsets`` -- ``offsetof`` for each of its fields
- ``nested`` -- ``offsetof`` for each member of the structures and unions
  inside it
- ``bits`` -- for each bit-field, at any depth, the bytes of the structure
  with that member set to all ones and everything else zero: where the first
  such byte is, and the bytes from there to the last. ``offsetof`` cannot be
  taken of a bit-field, and its position and width are both in the image.

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
from typing import Any

from pyopengl_video.nvenc import api

AGGREGATES = (ctypes.Structure, ctypes.Union)


def structures() -> list[type]:
    """Every structure and union the binding declares under a name the headers give it.

    The ones named with a leading underscore are unnamed in the headers, and
    are measured through each named structure that holds one.
    """
    return [value for name, value in vars(api).items()
            if isinstance(value, type) and issubclass(value, AGGREGATES)
            and value not in AGGREGATES and not name.startswith('_')]


def member_type(kind: Any, name: str) -> Any:
    """The declared type of the member `name` of `kind`, looked for in its unnamed members too."""
    declared = {entry[0]: entry[1] for entry in kind._fields_}
    if name in declared:
        return declared[name]
    for unnamed in getattr(kind, '_anonymous_', ()):
        try:
            return member_type(declared[unnamed], name)
        except KeyError:
            continue
    raise KeyError(f'{kind.__name__} has no member {name}')


def members(kind: Any, path: str = '') -> list[tuple[str, bool]]:
    """Every scalar member reachable inside `kind`, as (C expression, is a bit-field).

    A structure or union member is descended into, and an array of them
    through its first element, so the expression is one C takes an offset of
    or assigns to: ``rcParams.constQP.qpInterP``, ``meHintCountsPerBlock[0].numCandsPerSb``. A member
    the headers leave unnamed adds nothing to the expression.
    """
    found: list[tuple[str, bool]] = []
    unnamed = getattr(kind, '_anonymous_', ())
    for entry in kind._fields_:
        name, field = entry[0], entry[1]
        reached = path if name in unnamed else f'{path}.{name}' if path else name
        if len(entry) == 3:
            found.append((reached, True))
        elif issubclass(field, AGGREGATES):
            found += members(field, reached)
        elif issubclass(field, ctypes.Array) and issubclass(field._type_, AGGREGATES):
            found += members(field._type_, f'{reached}[0]')
        else:
            found.append((reached, False))
    return found


def fields_of(structure: Any) -> list[str]:
    """Each field of `structure` itself that C can take the offset of by its name."""
    unnamed = getattr(structure, '_anonymous_', ())
    return [entry[0] for entry in structure._fields_
            if len(entry) == 2 and entry[0] not in unnamed]


def c_program() -> str:
    """A C program printing every size, offset and bit image wanted."""
    lines = [
        '#include <stdio.h>', '#include <stddef.h>', '#include <string.h>',
        '#include "nvEncodeAPI.h"',
        'static void image(const char *name, const void *bytes, size_t size) {',
        '    printf("bits %s ", name);',
        '    for (size_t index = 0; index < size; index++)',
        '        printf("%02x", ((const unsigned char *)bytes)[index]);',
        '    printf("\\n");',
        '}',
        'int main(void) {',
    ]
    for structure in structures():
        name = structure.__name__
        direct = fields_of(structure)
        lines.append(f'    printf("size {name} %zu\\n", sizeof({name}));')
        for field in direct:
            lines.append(
                f'    printf("offset {name}:{field} %zu\\n", offsetof({name}, {field}));')
        for expression, bit_field in members(structure):
            if bit_field:
                lines.append(
                    f'    {{ static {name} probe; memset(&probe, 0, sizeof probe); '
                    f'probe.{expression} = ~0u; '
                    f'image("{name}:{expression}", &probe, sizeof probe); }}')
            elif expression not in direct:
                lines.append(
                    f'    printf("nested {name}:{expression} %zu\\n", '
                    f'offsetof({name}, {expression}));')
    lines += ['    return 0;', '}']
    return '\n'.join(lines) + '\n'


def trimmed(image: bytes) -> dict[str, Any]:
    """Where the first non-zero byte of `image` is, and the bytes from it to the last.

    A structure with one bit-field set is zero nearly everywhere, and a
    recording of whole structures would be mostly zeroes.
    """
    stripped = image.lstrip(b'\0')
    return {'at': len(image) - len(stripped), 'bytes': stripped.rstrip(b'\0').hex()}


def record(header: Path) -> dict[str, Any]:
    """Compile and run the probe against `header`, returning the ABI it reports."""
    with tempfile.TemporaryDirectory() as workdir:
        source = Path(workdir) / 'abi.c'
        source.write_text(c_program())
        binary = Path(workdir) / 'abi'
        # Assigning all ones to a bit-field narrower than an int is how its
        # image is made.
        subprocess.run(
            ['gcc', '-std=c11', '-Wno-overflow', '-I', str(header.parent), str(source),
             '-o', str(binary)],
            check=True)
        output = subprocess.run([str(binary)], check=True, capture_output=True, text=True)

    abi: dict[str, Any] = {
        'source': 'nvEncodeAPI.h from nv-codec-headers (MIT)',
        'header_version': '{}.{}'.format(*api.HEADER_VERSION),
        'sizes': {}, 'offsets': {}, 'nested': {}, 'bits': {},
    }
    for line in output.stdout.splitlines():
        kind, name, value = line.split(' ')
        structure, _, member = name.partition(':')
        if kind == 'size':
            abi['sizes'][name] = int(value)
        elif kind == 'offset':
            abi['offsets'].setdefault(structure, {})[member] = int(value)
        elif kind == 'nested':
            abi['nested'].setdefault(structure, {})[member] = int(value)
        else:
            abi['bits'].setdefault(structure, {})[member] = trimmed(bytes.fromhex(value))
    return abi


def reach(instance: Any, expression: str) -> tuple[Any, str]:
    """The object holding the member a C `expression` names, and the member's name."""
    *parents, last = expression.replace('[0]', '.[0]').split('.')
    for part in parents:
        instance = instance[0] if part == '[0]' else getattr(instance, part)
    return instance, last


def offset_of(structure: Any, expression: str) -> int:
    """The byte offset the binding gives the member a C `expression` names."""
    total = 0
    kind = structure
    for part in expression.replace('[0]', '.[0]').split('.'):
        if part == '[0]':
            kind = kind._type_
        else:
            total += getattr(kind, part).offset
            kind = member_type(kind, part)
    return total


def bit_image(structure: Any, expression: str) -> dict[str, Any]:
    """The binding's own image of `structure` with one bit-field all ones."""
    instance = structure()
    holder, member = reach(instance, expression)
    setattr(holder, member, 0xFFFFFFFF)
    return trimmed(bytes(instance))


def disagreements(abi: dict[str, Any]) -> list[str]:
    """What the binding says that the header does not."""
    wrong = [name for name, size in abi['sizes'].items()
             if ctypes.sizeof(getattr(api, name)) != size]
    wrong += [f'{name}.{member} at {offset_of(getattr(api, name), member)}, '
              f'header says {offset}'
              for name, recorded in (*abi['offsets'].items(), *abi['nested'].items())
              for member, offset in recorded.items()
              if offset_of(getattr(api, name), member) != offset]
    wrong += [f'{name}.{member} bits {bit_image(getattr(api, name), member)}, '
              f'header says {image}'
              for name, recorded in abi['bits'].items()
              for member, image in recorded.items()
              if bit_image(getattr(api, name), member) != image]
    return wrong


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    abi = record(Path(argv[1]).resolve())
    destination = Path(__file__).parent.parent / 'tests' / 'nvenc_abi.json'
    # Renamed over the recorded table once written, so the suite never reads half of one.
    partial = destination.with_name(destination.name + '.partial')
    partial.write_text(json.dumps(abi, indent=2, sort_keys=True) + '\n')
    partial.replace(destination)
    print(f'recorded {len(abi["sizes"])} structures, '
          f'{sum(map(len, abi["nested"].values()))} nested members and '
          f'{sum(map(len, abi["bits"].values()))} bit-fields to {destination}')
    wrong = disagreements(abi)
    for entry in wrong:
        print(f'  binding disagrees: {entry}')
    if not wrong:
        print('binding agrees with the header')
    return 1 if wrong else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
