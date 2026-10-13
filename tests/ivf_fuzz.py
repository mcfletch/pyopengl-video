"""Mutation fuzzing shared by the AV1 and VP9 gates.

A case is a fixture's samples with a few bytes changed, most often in its
headers, run through a control layer from parser to the buffers a driver is
handed. :class:`Session` stands in for the decoder's session and gives each
picture to the codec's own checks; :class:`Findings` gathers what they
report, so one run names every value out of its range and not the first
alone. :func:`recorded` wraps the structures a control layer fills, so that
every value it stores is read back.

The seed is fixed, so a failure names a case that runs again.
"""
import ctypes
import random
import signal
import time
import types
from pathlib import Path

from pyopengl_video.decoder import DecoderError, Sample
from pyopengl_video.ivf import IVFReader
from pyopengl_video.vaapi import api
from pyopengl_video.vaapi.h264 import StreamError

DATA = Path(__file__).parent / 'data' / 'decode'

#: The longest one case may take, in seconds.
LIMIT = 1.0

#: How many samples of a fixture a case decodes: its key frame and the frames
#: that predict from it.
SAMPLES = 6


class Findings:
    """What a run found wrong, by name, with the first case that showed each."""

    def __init__(self):
        self.case = None
        self.found = {}
        self.pictures = 0

    def report(self, name, value=None):
        count, first = self.found.get(name, (0, (value, self.case)))
        self.found[name] = (count + 1, first)

    def within(self, name, value, low, high):
        if not low <= value <= high:
            self.report(f'{name} outside {low}..{high}', value)

    def __str__(self):
        return '\n'.join(f'{count:6d}  {name}: first {value!r} in {case}'
                         for name, (count, (value, case)) in sorted(self.found.items()))


# --------------------------------------------------------- stores read back


class Recorded:
    """A ctypes structure, union or array whose every store is read back.

    ctypes keeps the low bits of a value that does not fit a field and raises
    nothing. Each attribute and each item stored through this is read again,
    and a value that reads back as another is reported with the field's name.
    Structures, unions and arrays reached through it are wrapped the same way.
    """

    __slots__ = ('_path', '_report', '_target')

    def __init__(self, target, report, path):
        object.__setattr__(self, '_target', target)
        object.__setattr__(self, '_report', report)
        object.__setattr__(self, '_path', path)

    def _wrapped(self, value, name):
        if isinstance(value, (ctypes.Structure, ctypes.Union, ctypes.Array)):
            return Recorded(value, self._report, f'{self._path}{name}')
        return value

    def _stored(self, name, value, stored):
        if isinstance(stored, ctypes.Array):
            value, stored = list(value), list(stored)
        elif isinstance(stored, list):
            value = list(value)
        if stored != value:
            self._report(f'{self._path}{name} reads back as another value', (value, stored))

    def __getattr__(self, name):
        return self._wrapped(getattr(self._target, name), f'.{name}')

    def __setattr__(self, name, value):
        setattr(self._target, name, value)
        self._stored(f'.{name}', value, getattr(self._target, name))

    def __getitem__(self, index):
        return self._wrapped(self._target[index], f'[{index}]')

    def __setitem__(self, index, value):
        self._target[index] = value
        self._stored(f'[{index}]', value, self._target[index])

    def __len__(self):
        return len(self._target)

    def __iter__(self):
        return (self[index] for index in range(len(self._target)))


def recorded(module, report):
    """`module` with each structure it declares made through :class:`Recorded`.

    module -- a module of ctypes declarations, such as
        :mod:`pyopengl_video.vaapi.av1_api`
    report -- called with a name and the (asked, stored) pair for a store
        that reads back as another value

    A control layer given this in place of the module builds the same
    buffers, with every store checked.
    """
    def made(name, structure):
        def make():
            return Recorded(structure(), report, name)
        return make

    names = {}
    for name, value in vars(module).items():
        if isinstance(value, type) and issubclass(value, (ctypes.Structure, ctypes.Union)):
            names[name] = made(name, value)
        else:
            names[name] = value
    return types.SimpleNamespace(**names)


# ------------------------------------------------------------- the session


class Session:
    """A decode session with no driver, which checks what it is handed.

    check_picture -- called with the findings, the session, the target
        surface and the buffers of each picture submitted
    decoded -- the surfaces of the active format that hold a decoded picture
    """

    def __init__(self, findings, check_picture):
        self.findings = findings
        self.check_picture = check_picture
        self.control = None
        self.format = None
        self.decoded = set()

    def activate(self, format):
        self.format = format
        self.decoded = set()
        self.findings.within('surfaces asked of the session', format.surfaces, 1, 18)
        if min(format.coded_size) < 1 or max(format.coded_size) > 1 << 16:
            self.findings.report('a coded size no surface has', format.coded_size)

    def acquire(self):
        held = self.control.held()
        free = set(range(self.format.surfaces)) - held
        if not free:
            self.findings.report('more surfaces held than the format asked for', len(held))
            raise DecoderError('every surface is in use')
        return min(free)

    def submit(self, surface, batches, what):  # noqa: ARG002 - the session's signature
        self.findings.pictures += 1
        self.check_picture(self.findings, self, surface, batches)
        self.decoded.add(surface)

    def output(self, surface, *, timestamp, duration, visible, colour):  # noqa: ARG002 - the session's signature
        x, y, width, height = visible
        coded_width, coded_height = self.format.coded_size
        if (width < 1 or height < 1 or x < 0 or y < 0
                or x + width > coded_width or y + height > coded_height):
            self.findings.report('a visible rectangle outside the picture', visible)
        if surface not in self.decoded:
            self.findings.report('a surface shown that was never decoded', surface)


def check_references(findings, session, surfaces):
    """Each of `surfaces` an inter picture predicts from holds a decoded picture."""
    for surface in surfaces:
        if surface == api.VA_INVALID_SURFACE:
            findings.report('an inter picture with an invalid reference surface')
        elif surface not in session.decoded:
            findings.report('a reference surface that was never decoded', surface)


# --------------------------------------------------------------- the cases


def streams(pattern):
    """The first samples of each IVF fixture matching `pattern`, by file name."""
    found = {}
    for path in sorted(DATA.glob(pattern)):
        with IVFReader(path) as stream:
            found[path.name] = [stream.sample(index).data
                                for index in range(min(len(stream), SAMPLES))]
    return found


def mutate(chooser, samples, fixtures, regions):
    """`samples` with one to three bytes or runs of bytes changed.

    fixtures -- every stream, for the cases that splice one onto another
    regions -- answers the byte ranges of a sample that hold its headers,
        where most changes are made
    """
    samples = list(samples)
    kind = chooser.random()
    if kind < 0.08:
        other = fixtures[chooser.choice(sorted(fixtures))]
        samples = (samples[:chooser.randrange(1, len(samples) + 1)]
                   + other[chooser.randrange(len(other)):])[:SAMPLES]
    elif kind < 0.12 and len(samples) > 1:
        del samples[chooser.randrange(len(samples))]
    for _ in range(chooser.choice((1, 1, 1, 2, 3))):
        index = chooser.randrange(len(samples))
        data = bytearray(samples[index])
        if not data:
            continue
        if chooser.random() < 0.8:
            low, high = chooser.choice(regions(bytes(data)))
            position = chooser.randrange(low, max(high, low + 1))
        else:
            position = chooser.randrange(len(data))
        position = min(position, len(data) - 1)
        change = chooser.random()
        if change < 0.6:
            data[position] ^= 1 << chooser.randrange(8)
        elif change < 0.9:
            data[position] = chooser.randrange(256)
        else:
            length = chooser.randrange(1, 5)
            data[position:position + length] = bytes(
                chooser.choice((0, 0xFF, chooser.randrange(256))) for _ in range(length))
        samples[index] = bytes(data)
    return samples


def cases(fixtures, count, seed, regions):
    """`count` mutated streams as ``(label, samples)``, the same for the same seed."""
    chooser = random.Random(seed)
    names = sorted(fixtures)
    for case in range(count):
        name = chooser.choice(names)
        yield (f'{name} seed {seed} case {case}',
               mutate(chooser, fixtures[name], fixtures, regions))


class TookTooLong(BaseException):
    """A case ran past :data:`LIMIT`."""


def _interrupt(*_arguments):
    raise TookTooLong


def guarded(findings, what, function, *arguments):
    """Call `function`; report an exception that is not the package's own.

    Answers what the function returned, or None where it raised.
    """
    try:
        return function(*arguments)
    except (StreamError, DecoderError):
        return None
    except Exception as error:  # noqa: BLE001 - any other exception is the finding
        findings.report(f'{type(error).__name__} escaped {what}', str(error)[:120])
        return None


def decode(control_class, samples, findings, check_picture, others=()):
    """One stream through a control layer, sample by sample.

    others -- further readers of the same samples, each called with a sample's
        bytes and held to the same exceptions

    The layer is given every sample whatever the one before raised, as a
    player that carries on past a damaged frame gives them.
    """
    session = Session(findings, check_picture)
    control = control_class(session, 4)
    session.control = control
    for index, data in enumerate(samples):
        guarded(findings, 'the control layer', control.decode,
                Sample(data, index, 1, False, index))
        for what, function in others:
            guarded(findings, what, function, data)


def run(control_class, fixtures, count, seed, regions, check_picture, findings, others=()):
    """Fuzz `control_class` with `count` cases, adding to `findings`.

    Reported: an exception that is not the package's, a case that takes
    longer than :data:`LIMIT`, and whatever `check_picture` and the session
    report of the buffers.
    """
    # A case that does not return is interrupted where the platform can, so
    # that the run reports it.
    interruptible = hasattr(signal, 'setitimer')
    previous = signal.signal(signal.SIGALRM, _interrupt) if interruptible else None
    try:
        for label, samples in cases(fixtures, count, seed, regions):
            findings.case = label
            started = time.perf_counter()
            if interruptible:
                signal.setitimer(signal.ITIMER_REAL, LIMIT)
            try:
                decode(control_class, samples, findings, check_picture, others)
            except TookTooLong:
                pass                            # reported from the time taken
            finally:
                if interruptible:
                    signal.setitimer(signal.ITIMER_REAL, 0)
            elapsed = time.perf_counter() - started
            if elapsed >= LIMIT:
                findings.report(f'a case took {LIMIT:g} s or longer', round(elapsed, 2))
    finally:
        if interruptible:
            signal.signal(signal.SIGALRM, previous)
    return findings
