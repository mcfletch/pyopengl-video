"""Scenarios on NVIDIA hardware that are run in a process of their own.

A driver that is handed a bad picture does not raise: it ends the process with
a signal, or leaves it waiting at exit for a thread that never finishes. A
test that ran such a scenario itself would take the whole run down with it, so
each one here runs in a child process, and the test asserts on how the child
ended: :func:`run` starts it, and :class:`Outcome` says whether it exited, was
killed or had to be stopped.

A scenario is a function of this module, named on the child's command line
with its arguments as JSON. It returns what the test asserts on, which the
child prints as its last line. A scenario that meets an exception other than
the package's own lets it end the child, so the test sees a non-zero exit.
"""
import dataclasses
import json
import random
import signal
import subprocess
import sys
from pathlib import Path

from pyopengl_video import VideoError, obu
from pyopengl_video.annexb import AnnexBReader
from pyopengl_video.ivf import IVFReader
from pyopengl_video.nvdec.decoder import NVDECDecoder

ROOT = Path(__file__).parent.parent
DATA = Path(__file__).parent / 'data' / 'decode'

#: The inputs the security review saved, each of which ended the process: a
#: fixture, with the samples that were changed, as an index to the
#: ``[offset, byte]`` pairs stored in the sample or the length it is cut to.
SAVED = {
    'a truncated temporal unit states a picture longer than its bytes (N2)': (
        'av1_large_grain10.ivf',
        {2: 3, 7: 233, 8: 337, 11: 1267, 15: 355, 16: 3, 18: 236}),
    'flipped header bits give a picture of length 0xFFFFFFF4 (N2)': (
        'av1_large_altref.ivf',
        {0: [[50, 0xf8], [93, 0xc3]], 1: [[2, 0x36], [13, 0x74]],
         2: [[25, 0x2e], [47, 0xb8], [50, 0x76]],
         6: [[12, 0x47], [33, 0xc4], [68, 0x0f], [72, 0x16]],
         17: [[22, 0xdd], [28, 0x68], [70, 0xc5], [80, 0x55]],
         18: [[4, 0x00], [10, 0x01], [27, 0xce], [59, 0x7e]],
         20: [[15, 0x22], [60, 0xa4]]}),
    'one bit of the sequence header, order_hint_bits 7 to 5 (N3)': (
        'av1_large_grain.ivf', {0: [[12, 0xe2]]}),
    'header bytes set in a stream with super-resolution (N3)': (
        'av1_large_superres.ivf',
        {0: [[15, 0x80], [35, 0x80]], 1: [[47, 0x80]],
         10: [[10, 0x80], [29, 0xff], [45, 0x00]], 16: [[0, 0x80], [3, 0xff]],
         17: [[61, 0x80]]}),
    'header bytes set in a stream with hidden frames (N3)': (
        'av1_large_altref.ivf',
        {0: [[45, 0xff], [49, 0x00]], 5: [[54, 0x00]], 9: [[58, 0x00]],
         14: [[18, 0xa9], [43, 0x00], [58, 0x6a]], 17: [[34, 0xff]], 18: [[4, 0x80]],
         20: [[5, 0x06], [33, 0x7f]]}),
    'flipped header bits in a stream with film grain (N3)': (
        'av1_large_grain.ivf',
        {0: [[12, 0xe2], [30, 0x92], [85, 0xac]], 3: [[76, 0xfe]],
         5: [[28, 0x3f], [55, 0xb9], [83, 0x8e], [86, 0xf3]],
         6: [[0, 0x1a], [2, 0x12], [4, 0x68]], 9: [[32, 0x01], [53, 0xed]],
         13: [[31, 0x7e], [47, 0x86], [63, 0xc6], [67, 0xd6], [68, 0x14]],
         17: [[4, 0x01], [5, 0x12], [93, 0x35]]}),
}

#: What opens the line a child's report is printed on.
REPORT = 'REPORT '

SCENARIOS = {}

#: What a scenario leaves alive until the child's interpreter shuts down.
KEPT = []


def scenario(function):
    """Make `function` a scenario a child can be asked for by name."""
    SCENARIOS[function.__name__] = function
    return function


# ------------------------------------------------------------ the parent's side


@dataclasses.dataclass
class Outcome:
    """How a child ended.

    returncode -- its exit status, negative for the signal that killed it, or
        None where it had to be stopped at the time limit
    report -- what its scenario returned, or None where it did not finish
    output -- everything it wrote, for a failure's message
    """

    returncode: int | None
    report: dict | None
    output: str

    @property
    def ended(self):
        """How the child ended, in words."""
        if self.returncode is None:
            return 'did not exit within its time limit'
        if self.returncode < 0:
            return f'was killed by {signal.Signals(-self.returncode).name}'
        return f'exited with status {self.returncode}'

    def clean(self):
        """The scenario's report; fails the test unless the child finished and exited 0."""
        assert self.returncode == 0 and self.report is not None, (
            f'the child process {self.ended}:\n{self.output[-3000:]}')
        return self.report


def run(name, *arguments, timeout=60.0):
    """Run scenario `name` in a child process and say how it ended."""
    command = [sys.executable, '-X', 'faulthandler', '-m', 'tests.nvidia_child',
               name, json.dumps(arguments)]
    try:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                   timeout=timeout, check=False)
    except subprocess.TimeoutExpired as stopped:
        output = ''.join(part.decode('utf-8', 'replace') if isinstance(part, bytes) else part
                         for part in (stopped.stdout or '', stopped.stderr or ''))
        return Outcome(None, None, output)
    report = None
    for line in completed.stdout.splitlines():
        if line.startswith(REPORT):
            report = json.loads(line[len(REPORT):])
    return Outcome(completed.returncode, report, completed.stdout + completed.stderr)


# ------------------------------------------------------------- the child's side


def context():
    """A current OpenGL context on a window that is never shown."""
    import glfw  # noqa: PLC0415 - the child alone opens a window

    assert glfw.init()
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    if sys.platform.startswith('linux'):
        glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
    window = glfw.create_window(64, 64, 'pyopengl-video child', None, None)
    assert window
    glfw.make_context_current(window)
    return window


def codec_of(fixture):
    """The codec a fixture in ``tests/data/decode`` is coded in."""
    stem, _, extension = fixture.rpartition('.')
    return {'h264': 'h264', 'h265': 'hevc'}.get(extension) or stem.split('_')[0]


def samples_of(fixture):
    """Every sample of a fixture, the parameter sets of an Annex-B one on its first."""
    path = DATA / fixture
    if fixture.endswith(('.h264', '.h265')):
        with AnnexBReader(path) as stream:
            return [stream.sample(index, parameter_sets=index == 0)
                    for index in range(len(stream))]
    with IVFReader(path) as stream:
        return [stream.sample(index) for index in range(len(stream))]


def changed(samples, changes):
    """`samples` with `changes` made: a sample's index to what becomes of its bytes.

    A change is a list of ``[offset, byte]`` pairs to store, or a length to
    cut the sample to.
    """
    found = list(samples)
    for index, change in changes.items():
        data = bytearray(found[int(index)].data)
        if isinstance(change, int):
            del data[change:]
        else:
            for offset, value in change:
                data[offset] = value
        found[int(index)] = dataclasses.replace(found[int(index)], data=bytes(data))
    return found


class Tally:
    """What a run of decoding produced: frames, and the package's errors by kind."""

    def __init__(self):
        self.frames = 0
        self.errors = {}

    def attempt(self, call):
        """Run `call`, releasing the frames it returns; count a package error it raises."""
        try:
            frames = call() or ()
        except VideoError as error:
            self.count(error)
            return
        for frame in frames:
            self.frames += 1
            frame.release()

    def count(self, error):
        """Note one of the package's errors, by its type and how its message opens."""
        kind = f'{type(error).__name__}: {str(error)[:60]}'
        self.errors[kind] = self.errors.get(kind, 0) + 1

    def report(self):
        return {'frames': self.frames, 'errors': self.errors}


@scenario
def decode(fixture, start=0, changes=None, headers=False, interop=True, gate=True):
    """Decode a fixture from sample `start`, with `changes` made to its samples.

    headers -- give the decoder the first sample's sequence header up front,
        as a container that keeps it apart from the samples does
    gate -- False to hand every AV1 sample to NVIDIA's parser unread, so that
        what its picture is held to in the decode callback is all there is
    """
    context()
    original = samples_of(fixture)
    samples = changed(original, changes or {})
    configuration = obu.sequence_header(original[0].data) if headers else b''
    decoder = NVDECDecoder(codec_of(fixture), headers=configuration, interop=interop)
    if not gate:
        decoder._gate = None  # noqa: SLF001 - the one way to reach the callback with what the gate stops
    tally = Tally()
    for sample in samples[start:]:
        tally.attempt(lambda sample=sample: decoder.decode(sample))
    tally.attempt(decoder.flush)
    decoder.close()
    return tally.report()


#: The ways a fuzz case differs from its fixture, the header mutations listed
#: twice so that they are half of all cases.
KINDS = ('flip-headers', 'flip', 'truncate', 'noise-first', 'shuffle', 'splice',
         'set-headers', 'empty', 'repeat', 'flip-headers', 'set-headers')


def mutated(rng, samples, others):
    """One fuzz case: `samples` changed in one of :data:`KINDS`."""
    kind = rng.choice(KINDS)
    if kind == 'splice' and others:
        at = rng.randrange(1, len(samples))
        return samples[:at] + rng.choice(others) + samples[at:]
    if kind == 'shuffle':
        at = rng.randrange(len(samples))
        return samples[:at] + rng.sample(samples[at:], len(samples) - at)
    if kind == 'repeat':
        at = rng.randrange(len(samples))
        return samples[:at] + [samples[at]] * 5 + samples[at + 3:]
    found = []
    for index, sample in enumerate(samples):
        data = bytearray(sample.data)
        if kind == 'flip-headers' and (index == 0 or rng.random() < .15):
            for _ in range(rng.randrange(1, 6)):
                data[rng.randrange(min(len(data), 96))] ^= 1 << rng.randrange(8)
        elif kind == 'set-headers' and (index == 0 or rng.random() < .15):
            for _ in range(rng.randrange(1, 4)):
                data[rng.randrange(min(len(data), 64))] = rng.choice(
                    [0, 0xff, 0x80, 0x7f, rng.randrange(256)])
        elif kind == 'flip' and rng.random() < .5:
            for _ in range(rng.randrange(1, 10)):
                data[rng.randrange(len(data))] ^= 1 << rng.randrange(8)
        elif kind == 'truncate' and rng.random() < .4:
            del data[rng.randrange(len(data) + 1):]
        elif kind == 'noise-first' and index == 0:
            found.append(dataclasses.replace(
                sample, data=rng.randbytes(rng.choice([1, 7, 100, 5000]))))
        elif kind == 'empty' and rng.random() < .2:
            data = bytearray()
        found.append(dataclasses.replace(sample, data=bytes(data)))
    return found


@scenario
def fuzz(fixture, seed, cases, interop=True, others=()):
    """Decode `cases` mutations of a fixture, each in a decoder of its own.

    others -- fixtures of the same codec whose samples a case may splice in

    Each case's seed is printed before it runs, so the last line a killed
    child wrote names the case that killed it.
    """
    context()
    base = samples_of(fixture)
    spliced = [samples_of(other) for other in others]
    tally = Tally()
    for case in range(seed, seed + cases):
        rng = random.Random(case)
        samples = mutated(rng, base, spliced)
        print('case', case, flush=True)
        decoder = NVDECDecoder(codec_of(fixture), interop=interop)
        held = []
        for index, sample in enumerate(samples):
            sample = dataclasses.replace(sample, timestamp=index * 3000)
            try:
                for frame in decoder.decode(sample):
                    tally.frames += 1
                    if rng.random() < .05:
                        held.append(frame)
                    else:
                        frame.release()
            except VideoError as error:
                tally.count(error)
            if rng.random() < .03:
                tally.attempt(decoder.reset)
        tally.attempt(decoder.flush)
        decoder.close()
    return {'cases': cases, **tally.report()}


def _attempt(call):
    """What `call` raised of the package's errors, as its type's name; '' for nothing."""
    try:
        call()
    except VideoError as error:
        return type(error).__name__
    return ''


@scenario
def encoder_lifetime(steps, bframes=0):
    """An NVENC encoder with inputs, then `steps` in order, then the end of the process.

    The steps: ``encode`` submits a picture from each input, ``drop-context``
    leaves no OpenGL context current, ``restore-context`` makes the window's
    current again, ``terminate`` ends GLFW, ``close`` closes the encoder,
    ``collect`` drops the last reference to it, and ``collect-on-thread``
    drops it on a thread that has no context.

    The report says, after each ``close``, what it raised, whether the session
    is still open and how many inputs still hold OpenGL objects.
    """
    import gc  # noqa: PLC0415 - the collecting scenarios alone need it
    import threading  # noqa: PLC0415 - as above

    import glfw  # noqa: PLC0415 - the child alone opens a window

    from pyopengl_video.nvenc.encoder import NVENCEncoder  # noqa: PLC0415 - loads the driver

    window = context()
    held = {'encoder': NVENCEncoder(320, 240, bframes=bframes)}
    # Kept past the scenario, so that an encoder nothing closed or collected
    # is still open when the interpreter shuts down.
    KEPT.append(held)
    encoder = held['encoder']
    held['inputs'] = [encoder.new_input() for _ in range(encoder.input_slots)]
    closes = []
    for step in steps:
        if step == 'encode':
            for index, handle in enumerate(held['inputs']):
                encoder.encode(handle, index * 1500)
        elif step == 'drop-context':
            glfw.make_context_current(None)
        elif step == 'restore-context':
            glfw.make_context_current(window)
        elif step == 'terminate':
            glfw.terminate()
        elif step == 'close':
            raised = _attempt(encoder.close)
            closes.append({'raised': raised, 'session': bool(encoder.session),
                           'inputs': sum(1 for handle in held['inputs']
                                         if not handle.released)})
        elif step == 'collect':
            del encoder
            held.clear()
            gc.collect()
        elif step == 'collect-on-thread':
            del encoder

            def drop():
                held.clear()
                gc.collect()

            thread = threading.Thread(target=drop)
            thread.start()
            thread.join()
    return {'closes': closes}


@scenario
def close_holding_pictures(rounds):
    """Open an encoder, leave it holding pictures back for reordering, and close it; `rounds` times."""
    from pyopengl_video.nvenc.encoder import NVENCEncoder  # noqa: PLC0415 - loads the driver

    context()
    held = 0
    for _ in range(rounds):
        encoder = NVENCEncoder(640, 480, fps=30, bframes=2)
        handles = [encoder.new_input() for _ in range(encoder.input_slots)]
        packets = []
        # A key frame and two B-pictures: the reference that lets them out is not sent.
        for index, handle in enumerate(handles[:3]):
            packets += encoder.encode(handle, index * 3000)
        held += 3 - len(packets)
        encoder.close()
    return {'held': held}


@scenario
def foreign_handles():
    """Encode with an input another encoder registered, open and then closed."""
    from pyopengl_video.nvenc.encoder import NVENCEncoder  # noqa: PLC0415 - loads the driver

    context()
    first, second = NVENCEncoder(320, 240), NVENCEncoder(320, 240)
    own, foreign = first.new_input(), second.new_input()
    report = {'another encoder\'s': _attempt(lambda: first.encode(foreign, 0))}
    second.close()
    report['a closed encoder\'s'] = _attempt(lambda: first.encode(foreign, 3000))
    report['its own'] = _attempt(lambda: first.encode(own, 6000))
    first.close()
    report['its own, closed'] = _attempt(lambda: first.encode(own, 9000))
    return report


def main(arguments):
    name, encoded = arguments
    report = SCENARIOS[name](*json.loads(encoded))
    print(REPORT + json.dumps(report), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
