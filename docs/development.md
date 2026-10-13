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
tox                           # the suite on every supported Python, plus the gates
tox -e lint,typecheck         # what CI holds a merge to
```

`tox.ini` is what CI runs, so the gates are the same either way. `typecheck` is
two mypy runs, not one: half this package is Windows-only and half is
Linux-only, and each is invisible to the other's platform, so the Windows shim
and the oneVPL backend are checked with `--platform win32` and everything else
as the host sees it.

The hardware tests open a hidden GLFW window. Nothing is presented — encoding
reads a texture and never swaps a buffer — so a test run does not flash over
whatever else is on screen.

## Releasing

The version in `src/pyopengl_video/__init__.py` is the release, and pushing it
to `main` is what publishes it. `.github/workflows/release.yml` runs the whole
test matrix, asks PyPI whether that version is already there, and uploads the
sdist and wheel by trusted publishing only if the answer is no and everything is
green. So a push to `main` that does not bump the version is an ordinary CI run,
and bumping the version is the release.

Nothing is published from a red build. PyPI versions are immutable, and a
release number burned on a broken build cannot be taken back — which is also why
the version check treats anything but HTTP 200 or 404 from PyPI as a failure
rather than guessing.

Every other branch is built by `test.yml` on push, so work in progress is
checked as it goes.

## What is where

```
src/pyopengl_video/
    backend.py      the Backend record, discovery and VideoError, for both directions
    encoder.py      the Encoder interface, Packet, and the encoder registry
    decoder.py      the Decoder interface, Sample, DecodedFrame, the decoder registry
    inputs.py       InputHandle: the texture an encoder reads, and its framebuffer
    codecs.py       the codec names, and the spellings that reach them
    mp4.py          packets into an MP4 file, and MP4Reader back out of one
    annexb.py       AnnexBReader: access units of a raw .h264 or .h265 stream
    ivf.py          IVFReader and IVFWriter, for AV1 and VP9
    obu.py          splitting an AV1 temporal unit into its OBUs
    colour.py       Colour: H.273 code points and the YUV to RGB transform
    convert.py      NV12Converter: two planes into an RGBA texture
    glstate.py      preserved_state(), which every pass draws inside
    playback.py     Movie, Player and Timeline
    linux/dmabuf.py exporting a texture as a DMA-BUF, and importing one back
    windows/        the Direct3D interop shim every Windows backend uses
    nvenc/api.py    ctypes over NvEncodeAPI: structures, GUIDs, function table
    nvenc/encoder.py    the encoder: register, map, encode, lock, unmap
    nvdec/api.py    ctypes over NVDEC and the CUDA driver API it decodes into
    nvdec/decoder.py    the decoder: parser callbacks, the copy into textures
    nvdec/description.py    the picture size and colour read here, for each codec
    nvdec/vp9.py    the VP9 reference slots, for a frame shown again
    vpl/api.py      ctypes over oneVPL: packed structures, status names, entry points
    vpl/allocator.py    the Direct3D 11 surfaces the runtime allocates through us
    vpl/encoder.py  the encoder: shared textures in, a stream out
    vpl/decoder.py  the decoder: Direct3D surfaces, the video processor, shared textures out
    presentation.py durations and the pool of output textures that copying decoders share
    vaapi/api.py    ctypes over libva: display, config, context, surfaces, buffers
    vaapi/device.py a libva display on a render node, shared by both directions
    vaapi/hevc_api.py, av1_api.py, vp9_api.py   each codec's libva structures
    vaapi/codecs.py     the codecs with a control layer, and which a device offers
    vaapi/control.py    what a control layer is: DecodeControl, EncodeControl
    vaapi/encoder.py    the encoder shell: export, import, convert, submit
    vaapi/decoder.py    the decoder shell: surfaces imported as textures, lending
    vaapi/bits.py       MSB-first bit reader and writer, for AV1 and VP9
    vaapi/h264.py       H.264 header writer, and the RBSP bit reader and writer
    vaapi/h264_parse.py, h264_dpb.py    H.264 syntax and picture management
    vaapi/h264_decode.py, h264_encode.py    H.264's two control layers
    vaapi/hevc.py, hevc_parse.py, hevc_dpb.py, hevc_decode.py, hevc_encode.py
    vaapi/av1.py, av1_parse.py, av1_decode.py, av1_encode.py
    vaapi/vp9_parse.py, vp9_tables.py, vp9_decode.py
tests/
    nvenc_abi.json  sizes and offsets recorded from NVIDIA's header
    nvdec_abi.json  the same for NVDEC and CUDA, in the Linux and Windows layouts
    vpl_abi.json    the same, from the oneVPL headers
    va_abi.json     the same, from the libva headers, plus their constant values
    data/decode/    streams in each codec and the checksums of their decoded pictures
tools/
    record_nvenc_abi.py    regenerates those files from real headers
    record_nvdec_abi.py
    record_vpl_abi.py
    record_va_abi.py
    make_decode_fixtures.py  rewrites tests/data/decode/ with ffmpeg, one codec or all
```

## How the NVENC binding works

`libnvidia-encode.so.1` exports two symbols. `NvEncodeAPIGetMaxSupportedVersion`
reports the newest interface version the driver supports.
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

The NVDEC and CUDA headers come from the same repository, and
`tools/record_nvdec_abi.py` takes the directory holding them:

```bash
python tools/record_nvdec_abi.py path/to/nv-codec-headers/include/ffnvcodec
```

Two of those structures are made of `unsigned long`, which is eight bytes on
Linux and four on Windows. The tool records both layouts from one compiler,
and `tests/test_nvdec_abi.py` checks the one the running interpreter uses, so
the Windows layout is held on Windows without a compiler there.

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
    name='vaapi', vendor='Intel/AMD', codecs=frozenset(ENCODERS),
    max_size=(8192, 8192), sizes={'h264': (4096, 4096)}, zero_copy=True,
    probe=probe, factory=_build, explain=explain, capabilities=encoder_codecs,
)
```

`codecs` is everything the backend drives. **`capabilities` says which of them
this machine's hardware has**: it returns each codec with the largest frame the
hardware takes, and discovery narrows the record to that, so `encoders()` and
`decoders()` report the machine and not the software. It is called after
`probe` has passed and must not raise. A backend that cannot ask without
building an encoder leaves it out, and refuses the codec by name when it is
built.

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
that cannot use the current context is unavailable rather than broken. Ask
through `pyopengl_video.linux.dmabuf`, which answers for a machine that has no
EGL library at all as readily as for one whose context is the wrong kind: the
EGL bindings do not import where there is nothing to bind to, so a probe that
reaches for them itself raises on exactly the machines it is there to
serve. Silence
matters as much as speed: a driver that logs to stderr on open must be given a
callback that swallows it, since a library must not print during discovery.

The encoder implements six methods, and the contract around them is what makes
backends interchangeable:

- **`new_input()`** returns a handle: a texture to draw into, a framebuffer
  filling it, and a `for_drawing()` scope for backends that share the texture
  with another API. It is how a caller gets an input on every backend. The
  base class makes a plain texture and calls `register()`; a backend whose
  inputs must be allocated by the driver overrides it and sets
  `allocates_inputs`. A handle unregisters itself as it closes, and `encode()`
  refuses one that is closed, from another encoder, held for drawing, or still
  being encoded.
- **`register(texture, target)`** returns a handle for a texture the caller
  made, on backends that accept one. Registration is the expensive half, and
  handles are reused for the whole recording.
- **`encode(handle, timestamp, duration, force_idr)`** returns a *list* of
  packets, possibly empty. Backends that hold frames back set
  `reorders_frames` and return packets in decode order carrying display
  timestamps.
- **`flush()`** returns everything still held.
- **`headers()`** returns the parameter sets as an Annex-B fragment.
- **`close()`** releases everything and is safe to call twice.

**Everything a backend raises at a caller is an `EncoderError`.** One `except`
clause has to be enough to know whether recording failed, so a driver error
class either subclasses it -- which is what `NVENCError` and `VPLError` do --
or is translated where it crosses into the encoder's own methods, which is what
`VAAPIEncoder` does for libva's error and the DMA-BUF shim's. Translate rather
than subclass when the error comes from something shared: `linux/dmabuf.py`
serves every Linux backend and libva decodes as well as encodes, so neither
error is an *encoder's* until an encoder raised it. Keep the original on
`__cause__` either way. `tests/test_errors.py` holds every backend to this, and
needs no hardware to do it.

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

## Choosing the GPU on a hybrid-graphics Windows machine

The Windows backends read a Direct3D texture that the OpenGL context draws into,
so each runs only when the context is on its vendor's GPU: `nvenc` and
`nvdec` on NVIDIA, `vpl` on Intel. On a laptop with both, Windows picks a GPU
for each executable, and the suite's contexts get the integrated one unless that
is changed. `tools/gpu_preference.py` in the workspace root sets it for the
virtualenv's interpreters:

```
python tools/gpu_preference.py set nvidia    # nvenc, nvdec and their round trips run
python tools/gpu_preference.py set intel     # vpl runs
python tools/gpu_preference.py show --probe  # what is set, and what a context lands on
python tools/gpu_preference.py reset         # back to Windows' choice
```

The tests for the other vendor's backend skip with the GPU they found. Run the
suite once for each setting to cover every backend a machine has.

## Adding a codec to the VA-API backend

libva is a slice-level interface. The driver reconstructs or codes blocks, and
the caller parses or writes the headers, orders the pictures and keeps the
references. That part is the codec's *control layer*, and
`vaapi/control.py` states what one is:

- `DecodeControl` parses each sample, tells its `DecodeSession` the `Format`
  its sequence needs (profile, coded size, surfaces held, bit depth), acquires
  a surface for each picture, submits the picture's parameter buffers, and
  gives the pictures to `output()` in display order. `held()` names the
  surfaces still referenced or waiting to be shown.
- `EncodeControl` decides what each picture is, returns the parameter buffers
  that code it, and turns the bytes the driver wrote into a packet.

The shells, `VAAPIDecoder` and `VAAPIEncoder`, hold the device, the surfaces,
the OpenGL textures and the calls, and are the same for every codec. A layer
holds no driver, so its rules are tested with a stand-in session
(`tests/test_vaapi_decode_session.py`, and each codec's own no-driver tests).

A codec is switched on by one `Support` entry in `DECODERS` or `ENCODERS` in
`vaapi/codecs.py`, naming the libva profiles that carry it and the layer's
class. Discovery, `open_decoder`, `open_encoder` and the players follow from
that entry. The codec's libva structures go in `vaapi/<codec>_api.py`, where
`tools/record_va_abi.py` finds them: it measures every public structure and
every `VA*` constant in the binding modules.

## Testing a decoder

Decoding these codecs is exact, so every correct decoder produces the same
bytes. The hardware tests (`tests/test_vaapi_decode.py`,
`test_nvdec_decode.py`, `test_nvdec_codecs.py`, `test_hevc_decode.py`,
`test_av1_decode.py`, `test_vp9_decode.py`) decode each
stream under `tests/data/decode/` and read the planes back from the textures:
NV12 from an 8-bit stream, P010 from a 10-bit one. Each picture's MD5 must
equal the one `ffmpeg`'s software decoder recorded. The streams come from
x264, x265, libaom, SVT-AV1 and libvpx, and use what our encoders do not
write: B-frame pyramids, weighted prediction, several slices or tiles a
picture, scaling matrices, hidden frames, film grain and open groups of
pictures. One interlaced H.264 stream is recorded apart from those, under
`interlaced` in its table, for the NVDEC backends. So are the HEVC, VP9 and
AV1 streams under `large`: each is at least 144 samples each way, the
smallest picture an NVIDIA decoder takes in HEVC (128 in VP9 and AV1), where
most of the others are smaller. `test_nvdec_codecs.py` decodes those, and
checks that a smaller one is refused by its size.
`python tools/make_decode_fixtures.py` writes the streams listed in it that
are not yet under `tests/data/decode/`, with their checksums, and
`python tools/make_decode_fixtures.py hevc av1` those codecs alone. A stream
already written with the settings listed is kept, since an encoder run twice
does not always write the same bytes; `--rewrite` writes every stream again.
Each codec has its own table, `tests/data/decode/<codec>.json`. The suite
needs neither `ffmpeg` nor the tool.

A parser is checked against `ffmpeg -bsf:v trace_headers`, which prints every
header field of a stream as a reference parser reads it. The values are
copied into the parser's test as constants.

These tests need no driver:

- `tests/test_h264_parse.py` writes syntax with `BitWriter` in the order of
  the specification's tables, then reads it back field by field. The writers
  are in `tests/h264_syntax.py`. What our encoders write is parsed too.
- `tests/test_h264_dpb.py` drives the decoded picture buffer with slice
  headers. It checks picture order counts, marking, reference lists and output
  order against clause 8.2.
- `tests/test_annexb_reader.py` runs the buffer over the x264 B-pyramid stream.
  The display order it derives must equal the composition times `ffmpeg`
  wrote into the MP4 of the same stream.
- `tests/test_vaapi_decode_buffers.py` builds libva's picture, matrix and
  slice buffers from known headers.
- `tests/test_playback.py` drives the player with a fake decoder, converter
  and texture factory, so seeking and decoding ahead run without GL.

A new decoder backend uses the parser only if its driver wants parsed syntax.
NVDEC and oneVPL parse the stream themselves. Every backend shares the
`Decoder` interface, the `DecodedFrame` lending rule and the conversion pass.

The oneVPL decoder has no bit-exact reference to compare with, because the
video processor converts to RGB on the GPU. `tests/test_vpl_decode.py` holds
the frame count, display order, timestamps and size of every fixture, and
decodes a stream the Intel encoder made from known colours to check the colour
and the orientation. oneVPL needs the surfaces the decoder asked for at
initialisation to be the ones the application decodes into, so
`SurfaceAllocator.latest` names the most recent allocation and the decoder
takes its work surfaces from that.
Its hardware test is the same checksum comparison.

## How the NVDEC backend works

NVDEC has no OpenGL device type, so it is not the mirror of NVENC. A decoded
picture is in CUDA device memory, and the CUDA driver API (`libcuda.so.1`,
which every NVIDIA driver installs) moves it into textures.

`cuvidCreateVideoParser` is given three callbacks, and `cuvidParseVideoData`
calls them before it returns:

- the sequence callback, at the first picture of a sequence. The decoder asks
  `cuvidGetDecoderCaps` whether the hardware takes the stream, creates the
  hardware decoder at the coded size, and answers with the number of decode
  surfaces the parser may cycle through.
- the decode callback, for each picture in decoding order. Its parameters go
  to `cuvidDecodePicture` as the parser built them.
- the display callback, for each picture in display order, with the timestamp
  of the sample it came from.

The parser reuses a picture as soon as the display callback returns, so the
copy out happens inside it: `cuvidMapVideoFrame64` gives the address and pitch
of the picture, and each plane is copied by `cuMemcpy2D` into a texture.
Chroma starts one picture height of rows below luma. An 8-bit stream is
decoded to NV12 and a 10-bit one to P016, two bytes a sample with the sample
in the high bits, which is P010's layout; `SURFACES` in `nvdec/decoder.py`
holds the two. The picture is mapped at its coded size made even each way:
VP9 and AV1 code pictures of odd sizes, and the decoder scales a picture
asked for at one. The `nvdec` backend's
textures are registered with `cuGraphicsGLRegisterImage` and mapped around the
copy, which stays on the GPU; `nvdec-copy` copies to host memory and uploads
with `glTexSubImage2D`.

An exception cannot cross the C frames between a callback and the caller of
`cuvidParseVideoData`. Each callback keeps the exception and returns 0, and
the decoder raises it once the parser has returned.

A seek destroys the parser and makes a new one, which holds no references.
The new parser reports the sequence again, and a sequence equal to the one
in use keeps its hardware decoder and its textures.

The parser reads H.264, HEVC, VP9 and AV1, and one `NVDECDecoder` serves all
four. Two things come from this package's own parsers instead
(`nvdec/description.py`): the picture size before the first sample, and the
colour description. The parser never reports where chroma samples sit, and
for VP9 it reports no colour at all.

VP9 has two more differences, both in `nvdec/vp9.py` and the decoder's
`_decode_vp9`:

- The parser shows no picture for a frame that is `show_existing_frame`. The
  decoder reads each frame's header for the reference slots it is stored in,
  takes the picture it was decoded into from the decode callback
  (`CUVIDPICPARAMS.CurrPicIdx`), and maps the slot's picture itself when a
  frame says to show it again. Each frame of a superframe is handed to the
  parser by itself, so the next decode callback is that frame's.
- Once a sample has held more than one frame, the parser times each picture
  by another sample than its own. VP9 has no reordering, so the parser is
  given no display delay and a picture shown takes the timestamp of the
  sample being parsed.

What the decoder decides without a driver is in plain objects, tested in
`tests/test_nvdec.py` on any machine: `Sequence` (what a stream needs and what
refuses it), `Presentation` (durations, and the leading pictures dropped after
a seek), `Outputs` (the texture pairs lent and returned), `SurfaceFormat` and
`plane_copies`. `tests/test_nvdec_description.py` and `tests/test_nvdec_vp9.py`
cover the two modules above the same way. `tests/test_nvdec_decode.py` (H.264)
and `tests/test_nvdec_codecs.py` (HEVC, VP9 and AV1) are the checksum
comparison on the hardware, for both ways of copying.

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
