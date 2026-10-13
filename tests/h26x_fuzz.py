"""Mutation fuzzing shared by the H.264 and HEVC gates.

A case is the first NAL units of a fixture with one to three of them mutated,
run through a control layer from parser to the buffers a driver is handed.
:class:`Session` stands in for the decoder's session and gives each picture
to the codec's own checks; :class:`Findings` gathers what they report, so one
run names every element out of its range and not the first alone.

The seed is fixed, so a failure names a case that runs again.
"""
import random
import signal
import time
from pathlib import Path

from pyopengl_video.backend import VideoError
from pyopengl_video.decoder import DecoderError, Sample
from pyopengl_video.mp4 import split_annexb
from pyopengl_video.vaapi.h264 import StreamError, emulation_prevented, rbsp_of

DATA = Path(__file__).parent / 'data' / 'decode'

#: The longest one case may take, in seconds.
LIMIT = 1.0

#: The most surfaces a decoded picture buffer of 16 and the picture being
#: decoded ask for.
MOST_SURFACES = 17


class Findings:
    """What a run found wrong, by name, with the first case that showed each."""

    def __init__(self):
        self.case = None
        self.found = {}

    def report(self, name, value=None):
        count, first = self.found.get(name, (0, (value, self.case)))
        self.found[name] = (count + 1, first)

    def within(self, name, value, low, high):
        if not low <= value <= high:
            self.report(f'{name} outside {low}..{high}', value)

    def __str__(self):
        return '\n'.join(f'{count:6d}  {name}: first {value!r} in {case}'
                         for name, (count, (value, case)) in sorted(self.found.items()))


class Session:
    """A decode session with no driver, which checks what it is handed.

    check_picture -- called with the findings, the session, the target
        surface and the buffers of each picture submitted
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
        self.findings.within('surfaces asked of the session', format.surfaces,
                             1, MOST_SURFACES)
        if min(format.coded_size) < 1:
            self.findings.report('a coded size with no samples', format.coded_size)

    def acquire(self):
        held = self.control.held()
        if len(held) > self.format.surfaces:
            self.findings.report('more surfaces held than the format asked for', len(held))
            raise DecoderError('every surface is in use')
        return min(set(range(self.format.surfaces + 1)) - held)

    def submit(self, surface, batches, what):  # noqa: ARG002 - the session's signature
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


def mutate(unit, random, header):
    """`unit` with its payload changed, most often in its first bytes.

    header -- how many bytes the unit's NAL header is
    """
    payload = bytearray(rbsp_of(unit[header:]))
    if not payload:
        return unit
    span = min(len(payload), random.choice((4, 8, 16, 48, 200)))
    kind = random.random()
    if kind < 0.45:                             # flipped bits
        for _ in range(random.choice((1, 1, 1, 2, 3, 6))):
            bit = random.randrange(span * 8)
            payload[bit >> 3] ^= 0x80 >> (bit & 7)
    elif kind < 0.6:                            # one byte replaced
        payload[random.randrange(span)] = random.choice(
            (0, 0, 0xFF, 1, 0x80, random.randrange(256)))
    elif kind < 0.75:                           # a run of zeroes, as long codes are
        position = random.randrange(span)
        length = random.choice((1, 2, 3, 4, 5, 6))
        payload[position:position + length] = bytes(length)
    elif kind < 0.85:                           # bytes inserted
        position = random.randrange(span)
        payload[position:position] = bytes(
            random.choice((0, 0, 0xFF, random.randrange(256)))
            for _ in range(random.choice((1, 2, 4, 5))))
    elif kind < 0.93:                           # cut short
        del payload[random.randrange(span):]
    else:                                       # bits removed, shifting the rest
        total = len(payload) * 8
        bit = random.randrange(span * 8)
        drop = random.choice((1, 1, 2, 3))
        value = int.from_bytes(payload, 'big')
        kept = max(total - bit - drop, 0)
        value = ((value >> (total - bit)) << kept | value & ((1 << kept) - 1)) << drop
        payload = bytearray((value & ((1 << total) - 1)).to_bytes(len(payload), 'big'))
    changed = bytes(unit[:header]) + emulation_prevented(bytes(payload))
    if random.random() < 0.05:
        changed = bytes(unit[:header]) + bytes(payload)     # with no escapes
    if random.random() < 0.03:
        head = bytearray(changed[:header])
        head[random.randrange(header)] ^= 1 << random.randrange(8)
        changed = bytes(head) + changed[header:]
    return changed


def streams(pattern):
    """The NAL units of each fixture matching `pattern`, by file name."""
    return {path.name: list(split_annexb(path.read_bytes()))
            for path in sorted(DATA.glob(pattern))}


def cases(fixtures, count, seed, header):
    """`count` mutated streams as ``(label, units)``, the same for the same seed."""
    chooser = random.Random(seed)
    names = sorted(fixtures)
    for case in range(count):
        name = chooser.choice(names)
        units = list(fixtures[name])
        # The first units hold the parameter sets and the first pictures.
        units = units[:chooser.choice((6, 10, 16, 30))]
        for _ in range(chooser.choice((1, 1, 1, 2, 3))):
            index = chooser.randrange(min(len(units), chooser.choice((3, 4, 6, len(units)))))
            units[index] = mutate(units[index], chooser, header)
        if chooser.random() < 0.05:
            units.insert(chooser.randrange(len(units)), chooser.choice(units))
        yield f'{name} seed {seed} case {case}', units


class TookTooLong(BaseException):
    """A case ran past :data:`LIMIT`."""


def _interrupt(*_arguments):
    raise TookTooLong


def decode(control_class, units, findings, check_picture):
    """One stream through a control layer, as one sample and a flush."""
    session = Session(findings, check_picture)
    control = control_class(session, 4)
    session.control = control
    data = b''.join(b'\x00\x00\x00\x01' + unit for unit in units)
    control.decode(Sample(data, 0, 1, True, 0))
    control.flush()


def run(control_class, fixtures, count, seed, header, check_picture, findings=None):
    """Fuzz `control_class` with `count` cases; the :class:`Findings`.

    findings -- the findings to add to, where the caller's own checks share them

    Reported: an exception that is not the package's, a case that takes
    longer than :data:`LIMIT`, and whatever `check_picture` and the session
    report of the buffers.
    """
    findings = findings or Findings()
    # A case that does not return is interrupted where the platform can, so
    # that the run reports it.
    interruptible = hasattr(signal, 'setitimer')
    previous = signal.signal(signal.SIGALRM, _interrupt) if interruptible else None
    try:
        for label, units in cases(fixtures, count, seed, header):
            findings.case = label
            started = time.perf_counter()
            if interruptible:
                signal.setitimer(signal.ITIMER_REAL, LIMIT)
            try:
                decode(control_class, units, findings, check_picture)
            except (StreamError, DecoderError) as error:
                if not isinstance(error, VideoError):
                    findings.report(f'{type(error).__name__} is not a VideoError')
            except TookTooLong:
                pass                            # reported from the time taken
            except Exception as error:  # noqa: BLE001 - any other exception is the finding
                findings.report(f'{type(error).__name__} escaped', str(error)[:120])
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
