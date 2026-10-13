"""Record the C ABI of the libva structures, for the binding to be checked against.

The declarations in ``pyopengl_video.vaapi.api`` are hand-written, and libva
hands nearly every one of them straight to a driver that reads the fields by
offset. A field declared one word out is not reported: the call succeeds, the
driver reads a frame rate out of a bitrate, and what comes out is a well-formed
stream of the wrong thing. This tool asks a C compiler what the headers really
say and writes the answers to ``tests/va_abi.json``, which
``tests/test_va_abi.py`` checks the binding against:

- ``sizes`` -- ``sizeof`` for each structure and union the headers name
- ``offsets`` -- ``offsetof`` for each of its fields
- ``nested`` -- ``offsetof`` for each member of the structures and unions
  inside it, which the headers leave unnamed and the binding declares as
  classes of its own with a leading underscore
- ``bits`` -- for each bit-field, at any depth, the bytes of the structure
  with that member set to all ones and everything else zero: where the first
  such byte is, and the bytes from there to the last. ``offsetof`` cannot be
  taken of a bit-field, and its position and width are both in the image.
- ``constants`` -- the value of every constant the binding names

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
#: ``ctypes.Structure`` and ``ctypes.Union`` in one is measured, with the
#: unnamed ones inside it, and so is every integer constant whose name the
#: headers also define.
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
    """Every structure and union the binding declares under a name the headers give it."""
    found: dict[str, type] = {}
    for module in MODULES:
        for name, value in vars(module).items():
            if (isinstance(value, type)
                    and issubclass(value, (ctypes.Structure, ctypes.Union))
                    and not name.startswith('_') and value.__module__ == module.__name__):
                found[name] = value
    return list(found.values())


def members(kind: type, path: str = '') -> list[tuple[str, bool]]:
    """Every scalar member reachable inside `kind`, as (C expression, is a bit-field).

    A structure or union member is descended into, and an array of them
    through its first element, so the expression is one C takes an offset of
    or assigns to: ``seq_fields.bits.chroma_format_idc``, ``objects[0].fd``.
    """
    found: list[tuple[str, bool]] = []
    for entry in kind._fields_:
        name, field = entry[0], entry[1]
        reached = f'{path}.{name}' if path else name
        if len(entry) == 3:
            found.append((reached, True))
        elif issubclass(field, (ctypes.Structure, ctypes.Union)):
            found += members(field, reached)
        elif (issubclass(field, ctypes.Array)
              and issubclass(field._type_, (ctypes.Structure, ctypes.Union))):
            found += members(field._type_, f'{reached}[0]')
        else:
            found.append((reached, False))
    return found


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


#: ``va.h`` includes each codec's header. ``va_drm.h`` declares one function
#: and nothing measured here, and is in its own package on some systems.
HEADERS = ('va/va.h', 'va/va_vpp.h', 'va/va_enc_h264.h', 'va/va_drmcommon.h')


def fields_of(structure: type) -> list[tuple[str, str]]:
    """Each field the binding declares, with the expression C reaches it by."""
    found = []
    for entry in structure._fields_:
        name = entry[0]
        key = (structure.__name__, name)
        if key in SKIPPED or len(entry) == 3:
            continue
        found.append((name, RENAMED.get(key, name)))
    return found


def c_program() -> str:
    """A C program printing every size, offset, bit image and constant value wanted."""
    lines = ['#include <stdio.h>', '#include <stddef.h>', '#include <string.h>']
    lines += [f'#include <{header}>' for header in HEADERS]
    lines += [
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
        lines.append(f'    printf("size {name} %zu\\n", sizeof({name}));')
        direct = {field for field, _ in fields_of(structure)}
        for field, expression in fields_of(structure):
            lines.append(
                f'    printf("offset {name}.{field} %zu\\n", '
                f'offsetof({name}, {expression}));')
        for expression, bit_field in members(structure):
            if bit_field:
                lines.append(
                    f'    {{ {name} probe; memset(&probe, 0, sizeof probe); '
                    f'probe.{expression} = ~0u; '
                    f'image("{name}:{expression}", &probe, sizeof probe); }}')
            elif expression not in direct:
                lines.append(
                    f'    printf("nested {name}:{expression} %zu\\n", '
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
        # the structure, so they are measured like any other. Assigning all
        # ones to a bit-field narrower than an int is how its image is made.
        command = ['gcc', '-std=c11', '-Wno-deprecated-declarations', '-Wno-overflow']
        if include is not None:
            command += ['-I', str(include)]
        command += [str(source), '-o', str(binary)]
        subprocess.run(command, check=True)
        output = subprocess.run([str(binary)], check=True, capture_output=True,
                                text=True)

    sizes: dict[str, int] = {}
    offsets: dict[str, dict[str, int]] = {}
    nested: dict[str, dict[str, int]] = {}
    bits: dict[str, dict[str, dict[str, Any]]] = {}
    constants: dict[str, int] = {}
    for line in output.stdout.splitlines():
        kind, name, value = line.split(' ')
        if kind == 'size':
            sizes[name] = int(value)
        elif kind == 'offset':
            structure, _, field = name.partition('.')
            offsets.setdefault(structure, {})[field] = int(value)
        elif kind == 'nested':
            structure, _, member = name.partition(':')
            nested.setdefault(structure, {})[member] = int(value)
        elif kind == 'bits':
            structure, _, member = name.partition(':')
            bits.setdefault(structure, {})[member] = trimmed(bytes.fromhex(value))
        else:
            constants[name] = int(value)
    return {
        'source': 'the libva headers from intel/libva (MIT)',
        'sizes': sizes,
        'offsets': offsets,
        'nested': nested,
        'bits': bits,
        'constants': constants,
    }


def trimmed(image: bytes) -> dict[str, Any]:
    """Where the first non-zero byte of `image` is, and the bytes from it to the last.

    A structure with one bit-field set is zero nearly everywhere, and a
    recording of whole structures would be mostly zeroes.
    """
    stripped = image.lstrip(b'\0')
    return {'at': len(image) - len(stripped), 'bytes': stripped.rstrip(b'\0').hex()}


def reach(instance: Any, expression: str) -> tuple[Any, str]:
    """The object holding the member a C `expression` names, and the member's name."""
    *parents, last = expression.replace('[0]', '.[0]').split('.')
    for part in parents:
        instance = instance[0] if part == '[0]' else getattr(instance, part)
    return instance, last


def offset_of(structure: type, expression: str) -> int:
    """The byte offset the binding gives the member a C `expression` names."""
    total = 0
    kind: Any = structure
    for part in expression.replace('[0]', '.[0]').split('.'):
        if part == '[0]':
            kind = kind._type_
        else:
            field = getattr(kind, part)
            total += field.offset
            kind = dict((entry[0], entry[1]) for entry in kind._fields_)[part]
    return total


def bit_image(structure: type, expression: str) -> dict[str, Any]:
    """The binding's own image of `structure` with one bit-field all ones."""
    instance = structure()
    holder, member = reach(instance, expression)
    setattr(holder, member, 0xFFFFFFFF)
    return trimmed(bytes(instance))


def disagreements(abi: dict) -> list[str]:
    """What the binding says that the headers do not."""
    wrong = [name for name, size in abi['sizes'].items()
             if ctypes.sizeof(declared(name)) != size]
    wrong += [f'{name}.{member} at {offset_of(declared(name), member)}, headers say {offset}'
              for name, recorded in (*abi['offsets'].items(), *abi['nested'].items())
              for member, offset in recorded.items()
              if offset_of(declared(name), RENAMED.get((name, member), member)) != offset]
    wrong += [f'{name}.{member} bits {bit_image(declared(name), member)}, headers say {image}'
              for name, recorded in abi['bits'].items()
              for member, image in recorded.items()
              if bit_image(declared(name), member) != image]
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
    print(f'recorded {len(abi["sizes"])} structures, '
          f'{sum(map(len, abi["nested"].values()))} nested members, '
          f'{sum(map(len, abi["bits"].values()))} bit-fields and '
          f'{len(abi["constants"])} constants to {destination}')
    for entry in wrong:
        print(f'  binding disagrees: {entry}')
    if not wrong:
        print('binding agrees with the headers')
    return 1 if wrong else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
