# pyopengl-video

Hardware video encoding of OpenGL colour buffers, and hardware decoding into
OpenGL textures, with the frame never leaving the GPU.

[![A recording made by this encoder](https://img.youtube.com/vi/hprqotb3TXg/maxresdefault.jpg)](https://youtu.be/hprqotb3TXg)

The renderer has already put the frame in GPU memory, and the GPU has a video
encoder on the same die. `pyopengl-video` hands one to the other: a texture goes
in, a compressed stream comes out, and nothing crosses the bus but the
compressed result. The codec is any the GPU's encoder has: H.264, HEVC, AV1 or
VP9.

```python
from pyopengl_video import open_encoder
from pyopengl_video.mp4 import MP4Writer

with open_encoder(1920, 1080, fps=60, bitrate=12_000_000) as encoder:
    handle = encoder.new_input()
    with MP4Writer('out.mp4', encoder) as movie:
        for index in range(600):
            render()
            with handle.for_drawing():
                copy_the_frame_into(handle.framebuffer)
            movie.write(encoder.encode(handle, timestamp=index * 1500))
        movie.write(encoder.flush())
```

`examples/record_triangle.py` is that loop around a real renderer, start to
finish.

Playback runs the other way. The GPU's video decoder writes each picture,
OpenGL reads it as textures, and a conversion pass makes an RGBA texture.

```python
from pyopengl_video.playback import Movie

movie = Movie.open('intro.mp4')          # any thread: reads the index
player = movie.player()                  # GL thread
texture = player.frame_at(seconds)       # right way up, sRGB
```

`examples/play_movie.py` plays a file in a window.
[docs/playback.md](docs/playback.md) covers the player, the decoders and what
they decode. H.264, HEVC, AV1 and VP9 decode at 8 and 10 bits on AMD under
Linux through VA-API and on NVIDIA under Linux through NVDEC. NVDEC decodes
H.264 on Windows too, and runs the same code for the other codecs there,
untested. H.264, HEVC and VP9 decode on Intel under Windows through oneVPL.
Intel on Linux runs the VA-API code, untested. AMD on Windows is
[planned](plans/GPU-VIDEO-DECODE.md).

## What it needs

Python 3.10 or newer, PyOpenGL, and a GPU whose encoder it knows about. Nothing
else: the encoder is the one in the graphics driver, loaded by name, and the MP4
muxer is part of this package.

| Backend | Hardware | Platform | Codecs | Frame handed over as | State |
| --- | --- | --- | --- | --- | --- |
| `nvenc` | NVIDIA, Kepler and later | Linux | H.264, HEVC, AV1 | an OpenGL texture, in place | H.264 and HEVC working; AV1 written from the header and not yet run |
| `nvenc` | NVIDIA, Kepler and later | Windows | H.264, HEVC, AV1 | a Direct3D 11 surface OpenGL draws into | H.264 working, decoded back through NVDEC; HEVC runs and has not been decoded; AV1 needs an RTX 40 series part |
| `vaapi` | AMD, VCN | Linux | H.264, HEVC, AV1 | a DMA-BUF exported from a texture | working |
| `vpl` | Intel, Gen9 and later | Windows | H.264, HEVC, AV1, VP9 | a Direct3D 11 surface OpenGL draws into | H.264 working; the others written from the headers and not yet run |
| `vaapi` | Intel, Gen9 and later | Linux | H.264, HEVC, AV1 | the same DMA-BUF | the same code, untested |
| `amf` | AMD | Windows | | the same Direct3D 11 surface | [planned](plans/WINDOWS-SUPPORT.md) |

The codecs column is what each backend drives. Which of them a machine gets
depends on its GPU: `encoders()` and `decoders()` ask the hardware and list
only the codecs it has, and `open_encoder(width, height, codec='av1')` takes
the first backend whose hardware encodes it.

The frame reaches the encoder by whatever handle the platform has for one.
NVIDIA takes an OpenGL texture by name on Linux; Intel and AMD on Linux take a
DMA-BUF exported from one; on Windows every vendor's encoder takes a Direct3D 11
texture, and `WGL_NV_DX_interop2` makes one allocation that is a
Direct3D texture and an OpenGL texture at the same time. The muxer, the
interface and the recorder are the same either way.

**The `vaapi` backend needs an EGL context.** Exporting a texture as a DMA-BUF
is an EGL extension, and GLFW makes a GLX context by default on X11, so ask for
one before the window is created:

```python
glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
```

A context that cannot export is one this backend cannot record from, so it
reports itself unavailable rather than failing later; `encoders()` returning
nothing on a machine that should have an encoder is the first thing this
explains. It also needs a VA-API driver for the GPU installed beside `libva`
itself — `mesa-va-drivers` for AMD, `intel-media-va-driver` for Intel.

**Zero-copy needs the encoder on the same GPU as the renderer.** On a machine
with more than one, the backends match the OpenGL context's adapter and offer
themselves only there — see [the plan](plans/WINDOWS-SUPPORT.md).

Ask what a machine can do:

```python
from pyopengl_video import encoders

for backend in encoders():
    print(backend.name, backend.vendor,
          {codec: backend.largest(codec) for codec in sorted(backend.codecs)})
```

On an AMD Radeon 8060S that prints
`vaapi Intel/AMD {'av1': (8192, 4352), 'h264': (4096, 4096), 'hevc': (8192, 4352)}`.

## Two things to know before recording your own renderer

**The picture comes out upside down unless the copy turns it over.** OpenGL's
framebuffer starts at the bottom left; the encoder reads a texture from its
first row and calls that the top of the picture. A `glBlitFramebuffer` with its
destination Y coordinates reversed flips the frame as it copies, at no cost —
`capture()` in the example does exactly that.

**One texture is not enough.** An encoder that reorders frames is still reading
a texture after `encode()` has returned. Register `encoder.input_slots` textures
and cycle through them; handing back one the encoder still holds raises an error
that says so.

## Timing, colour and reordering

Timestamps and durations are in the encoder's `timescale`, 90 kHz by default,
and the encoder echoes back what it is given — the caller decides what a frame
time means, and a recording of fixed-step frames stays smooth however long each
frame took to render.

The hardware converts RGB to YUV, and the stream says which way: limited-range
BT.709 primaries, transfer and matrix, written into the stream's own colour
description (the video usability information of H.264 and HEVC, the sequence
header of AV1) along with the frame rate.

With `bframes` above zero the encoder holds pictures back and `encode()` returns
an empty list until it lets several go at once. Packets then arrive in decode
order carrying display timestamps, and `MP4Writer` records the difference as
composition offsets. An empty list is an ordinary answer at any setting: never
assume one frame in means one packet out.

## Documentation

- [docs/usage.md](docs/usage.md) — the recording loop, every encoder setting,
  timing, colour, reordering, muxing, errors and limits.
- [docs/playback.md](docs/playback.md): playing a movie into a texture, the
  decoders, seeking, colour and what is decoded.
- [docs/development.md](docs/development.md) — how the binding works, the ABI
  harness, and what a new backend has to implement.
- [plans/](plans/) — the design and its open questions:
  [the whole picture](plans/GPU-VIDEO-ENCODE.md),
  [Intel and AMD](plans/VAAPI-BACKEND.md),
  [Windows](plans/WINDOWS-SUPPORT.md),
  [decoding into a texture](plans/GPU-VIDEO-DECODE.md).

## Development

```bash
pip install -e '.[dev]'
pytest
```

Tests that need an encoder, a decoder or a GL context skip themselves without
one, so the suite runs anywhere. The hardware tests use a hidden GLFW window and
synthetic frames. The decoders' tests also decode the streams under
`tests/data/decode/`, written by x264, x265, libaom, SVT-AV1 and libvpx. Each
picture must match the checksum `ffmpeg`'s software decoder recorded for it.
`tools/make_decode_fixtures.py` writes the streams.

The vendor bindings are hand-written ctypes over large C ABIs, and a layout
mistake there corrupts a structure rather than raising anything. So each is
checked against the sizes and offsets its header states —
`tests/test_nvenc_abi.py`, `tests/test_nvdec_abi.py`, `tests/test_va_abi.py`
and `tests/test_vpl_abi.py` — and all of them run with no compiler, no driver
and no GPU. oneVPL packs each structure to 4 or 8 bytes,
which changes the layout and which a runtime reports only as an invalid
parameter, so those checks earn their keep.

Re-record after changing a structure, or when moving to a newer header:

```bash
python tools/record_nvenc_abi.py path/to/nvEncodeAPI.h
python tools/record_vpl_abi.py path/to/libvpl/api/vpl
python tools/record_va_abi.py                    # /usr/include, from libva-dev
```

The libva recording carries the value of every constant the binding names as
well as the layouts, because an enumerator that moved is as quiet a failure as a
field at the wrong offset.

## Licence

BSD-3-Clause; see `license.txt`. It contains no third-party code; see
`NOTICES.md` for where the driver ABI facts and the specifications come from,
and under what terms.
