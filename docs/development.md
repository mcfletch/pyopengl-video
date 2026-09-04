# Working on pyopengl-video

## Getting set up

```bash
pip install -e '.[dev]'
pytest
```

Tests that need an OpenGL context or a real encoder skip themselves when there
is neither, so the suite runs on a laptop with no GPU and says so rather than
failing. On a machine with an NVIDIA card the whole suite runs, hardware
included, in about ten seconds.

```bash
pytest                        # everything this machine can run
pytest tests/test_mp4.py      # muxer only: pure Python, no hardware
ruff check . && mypy src/pyopengl_video
```

The hardware tests open a hidden GLFW window. Nothing is presented — encoding
reads a texture and never swaps a buffer — so a test run does not flash over
whatever else is on screen.

## What is where

```
src/pyopengl_video/
    encoder.py      the Encoder interface, Packet, and the backend registry
    inputs.py       InputHandle: the texture an encoder reads, and its framebuffer
    mp4.py          Annex-B into an MP4 file
    linux/dmabuf.py exporting a texture as a DMA-BUF, and importing one back
    windows/        the Direct3D interop shim every Windows backend uses
    nvenc/api.py    ctypes over NvEncodeAPI: structures, GUIDs, function table
    nvenc/encoder.py    the encoder: register, map, encode, lock, unmap
    vpl/            Intel's oneVPL, over the Windows interop
    vaapi/api.py    ctypes over libva: display, config, context, surfaces, buffers
    vaapi/h264.py   the control layer libva leaves to the caller
    vaapi/encoder.py    the encoder: export, import, convert, code
tests/
    nvenc_abi.json  sizes and offsets recorded from NVIDIA's header
    vpl_abi.json    the same, from the oneVPL headers
    va_abi.json     the same, from the libva headers, plus their constant values
tools/
    record_nvenc_abi.py    regenerates those files from real headers
    record_vpl_abi.py
    record_va_abi.py
```

## How the NVENC binding works

`libnvidia-encode.so.1` exports two symbols. `NvEncodeAPIGetMaxSupportedVersion`
reports how new an interface the driver understands;
`NvEncodeAPICreateInstance` fills in a table of function pointers, and every
other call goes through the table. `NvEncodeAPI` in `api.py` does both once per
process and binds the entries this package uses.

Two things about the interface shape the code.

**Structures carry versions.** Each has a `version` field packing the interface
version with a per-structure revision, and the driver rejects one it does not
recognise. `NvEncodeAPI.struct_version` builds them from the *lower* of the
version the binding describes and the version the driver reports, so a binding
written against a newer header still talks to an older driver. Fields the older
driver does not know sit in what it considers reserved space, and they are zero.

**Reserved arrays are load-bearing.** Nearly every structure ends in reserved
words and reserved pointers, sized so the structure keeps its size as the
interface grows. A union is sized by its largest arm, which may be an arm this
binding does not declare — `NV_ENC_CODEC_PIC_PARAMS` is one, and it aligns to
eight because the arm it does not declare holds pointers.

## The ABI harness

A hand-written binding to a large C ABI fails silently: a field at the wrong
offset reads a bitrate out of a frame rate, and nothing reports an error. So the
layout is checked by machine.

`tools/record_nvenc_abi.py` writes a small C program that prints `sizeof` for
every structure the binding declares and `offsetof` for every field, compiles it
against a real `nvEncodeAPI.h`, and records the answers in
`tests/nvenc_abi.json`. `tests/test_nvenc_abi.py` then checks the binding
against that file — with no compiler, no driver and no GPU, so the layout is
verified everywhere the tests run.

Re-record after adding or changing a structure, or when moving to a newer
header:

```bash
python tools/record_nvenc_abi.py path/to/nvEncodeAPI.h
```

It prints which structures the binding disagrees with, so a mismatch is visible
before the tests run. `test_every_declared_structure_is_recorded` fails if a
structure is added to the binding and the recording is not refreshed, so an
unchecked structure cannot slip in.

The header is NVIDIA's, distributed under the MIT licence in
[nv-codec-headers](https://github.com/FFmpeg/nv-codec-headers). It is not
vendored here: only the facts it states about layout are, which is what
`NOTICES.md` records.

The libva and oneVPL headers are treated the same way, and are MIT too.
`tools/record_va_abi.py` does the same for libva and **also records the value of
every constant the binding names**. An enumerator that moved is as quiet a
failure as a field at the wrong offset, and libva has two spellings of "slice"
whose values differ -- the config-attribute bit is 4 and the packed-header type
is 3 -- which is exactly the mistake a recorded table catches and a careful
reading does not:

```bash
python tools/record_va_abi.py            # /usr/include, where libva-dev put them
```

## Adding a backend

A backend is a module exposing a `Backend` record. `pyopengl_video/__init__.py`
registers the ones that ship; the registry is a plain list, so an out-of-tree
backend can append to it.

```python
BACKEND = Backend(
    name='vaapi', vendor='Intel/AMD', codecs=frozenset({'h264'}),
    max_size=(4096, 4096), zero_copy=True,
    probe=probe, factory=_build, explain=explain,
)
```

`probe()` must be **cheap and quiet**: discovery calls it, so it loads a library
and checks the platform, and it returns False rather than raising when the
answer is no. Importing the backend module must not require its hardware — the
encoder module is imported inside the factory, so discovery costs one `dlopen`.

**Say why, when the answer is no.** A backend may carry an `explain` callable
beside its `probe`, returning a sentence naming what is missing when that is
something the caller could act on. `open_encoder` puts those in the error it
raises, because "none available" sends someone to look at their hardware when
the answer was a window hint. Keep it to what is fixable: a backend that is
simply on the wrong platform has nothing useful to say and returns `''`.

Where the library being present does not answer the question, a probe may ask
the device and **keep the answer**: `libva` is installed on plenty of machines
whose GPU has no encoder, so `pyopengl_video.vaapi.probe` opens each render node
once and remembers. It also refuses a context it could not record from — the
`vaapi` backend needs an EGL context to export a texture from, and a backend
that cannot use the current context is unavailable rather than broken. Silence
matters as much as speed: a driver that logs to stderr on open must be given a
callback that swallows it, since a library must not print during discovery.

The encoder implements five methods, and the contract around them is what makes
backends interchangeable:

- **`register(texture, target)`** returns a handle. Registration is the
  expensive half, and handles are reused for the whole recording.
- **`encode(handle, timestamp, duration, force_idr)`** returns a *list* of
  packets, possibly empty. Backends that hold frames back set
  `reorders_frames` and return packets in decode order carrying display
  timestamps.
- **`flush()`** returns everything still held.
- **`headers()`** returns the parameter sets as an Annex-B fragment.
- **`close()`** releases everything and is safe to call twice.

Set `size`, `timescale`, `zero_copy`, `reorders_frames` and `input_slots` before
returning from `__init__`. `input_slots` is what a caller allocates textures
from, so derive it from what the driver actually settled on rather than from
what was requested — a preset can turn lookahead on by itself, and the caller
then needs more textures than the arguments suggest.

**Find out how much of the stream the driver writes.** It is not a given. NVENC
and oneVPL hand back a complete elementary stream; a libva driver may write only
the coded macroblocks, leaving the NAL header, the slice header and the
parameter sets to the caller — Mesa's does. `VAConfigAttribEncPackedHeaders`
says which of those a driver will accept, and **asking for one is a promise to
supply it**: request the full mask it reports and a driver that would have
written its own stops, and every picture reaches the stream with no header on
it. Nothing fails; the bytes are simply not a video. Ask for exactly what the
backend writes.

The same applies in reverse to what comes back out. A driver may amend the
parameter sets it was handed — Mesa clears `transform_8x8_mode_flag`, because
its encoder writes no per-macroblock transform flag and a stream claiming
otherwise is one a decoder mis-parses — so `headers()` must report the sets the
*stream* carries once there are any, not the ones the backend built.

Two conventions the tests rely on:

- **Timestamps come back as they went in.** The caller decides what a frame time
  means.
- **The picture is not flipped or converted on the way through.** The first row
  of the texture is the top of the picture, and colour is limited-range BT.709
  declared in the stream. A backend whose hardware differs converts, rather than
  making every caller ask which backend it has.

New backends need entries in `docs/usage.md` and in the README's support table,
and the plan in `plans/` updated to say what landed.

## Testing a backend

`tests/test_nvenc_encode.py` is the pattern. Beyond the obvious — packet counts,
key frame positions, timestamps — two kinds of test are worth copying:

**Content sensitivity.** A registration that quietly went nowhere still produces
a well-formed stream, of something else. Encoding noise must cost far more than
encoding a flat colour, and a still sequence far less than a moving one. Measure
the second at a constant quantiser: under a bitrate target the encoder spends
its budget either way, which measures the rate control rather than the content.

**Reordering.** Run with `bframes` above zero and check that some calls return
nothing, that some return several, that every timestamp comes back exactly once,
and that the order they arrive in is not the order they went in.

**Orientation and colour, without a decoder.** These cannot be read out of a
bitstream, and a stream of the wrong picture is as well formed as a stream of
the right one. Where the driver can address its own surfaces -- libva's
`vaDeriveImage` does -- the surface the encoder is *about to code* can be read
back and checked directly, which is what `tests/test_vaapi_encode.py` does for
limited-range BT.709 luma and for which way up the picture is. It is a stronger
check than decoding the output, because it isolates the path from the texture to
the encoder from anything the encoder then does.

**Conformance needs something from outside.** A stream can be well formed,
correctly timed and still describe itself in a way that sets a decoder up
wrongly, and nothing inside this package can notice. So where `ffmpeg` happens
to be on the path, `test_a_muxed_recording_decodes_without_complaint` decodes a
recording and requires it to be silent; where it is not, the test skips. It is
not a dependency and the suite does not need it. It earned its keep on the first
run, on a container whose sample description did not match its samples.

## Style and licensing

The house rules are the workspace's: deliberate edits, no corner-cutting,
documentation updated in the same change as the code it describes.

**Nothing copyleft comes in.** Not GPL, LGPL, AGPL or CC-BY-SA, in whole or in
part or translated. The vendor interfaces this package needs are all permissive —
NVIDIA's header, `libva`, AMF and oneVPL are MIT — so the question does not
normally arise. If some behaviour is only documented by a copyleft
implementation, the workspace's `CLEAN-ROOM.md` procedure applies: a separate
reader writes a specification of the facts, that specification goes in `specs/`,
and the implementer works from it.

Record where facts come from in `NOTICES.md`, and cite the record rather than
the original in the code.
