# pyopengl-video

Hardware video encoding of OpenGL colour buffers, with the frame never leaving
the GPU.

The renderer has already put the frame in GPU memory, and the GPU has a video
encoder on the same die. `pyopengl-video` hands one to the other: a texture goes
in by name, an H.264 stream comes out, and nothing crosses the bus but the
compressed result.

```python
from pyopengl_video import open_encoder
from pyopengl_video.mp4 import MP4Writer

with open_encoder(1920, 1080, fps=60, bitrate=12_000_000) as encoder:
    handle = encoder.register(texture)
    with MP4Writer('out.mp4', encoder) as movie:
        for index in range(600):
            render()
            copy_the_frame_into(texture)
            movie.write(encoder.encode(handle, timestamp=index * 1500))
        movie.write(encoder.flush())
```

`examples/record_triangle.py` is that loop around a real renderer, start to
finish.

## What it needs

Python 3.10 or newer, PyOpenGL, and a GPU whose encoder it knows about. Nothing
else: the encoder is the one in the graphics driver, loaded by name, and the MP4
muxer is part of this package.

| Backend | Hardware | Platform | Codec | Frame handed over as | State |
| --- | --- | --- | --- | --- | --- |
| `nvenc` | NVIDIA, Kepler and later | Linux | H.264 | an OpenGL texture, in place | working |
| `vaapi` | Intel and AMD | Linux | H.264 | a DMA-BUF exported from a texture | [planned](plans/VAAPI-BACKEND.md) |

**Linux only, for now.** NVIDIA supports the encoder's OpenGL device type on
Linux alone, so a texture cannot be handed over by name on Windows however good
the card is. The muxer and the interface are platform-independent, and
[what Windows would take](plans/WINDOWS-SUPPORT.md) is written down.

Ask what a machine can do:

```python
from pyopengl_video import encoders

for backend in encoders():
    print(backend.name, backend.vendor, sorted(backend.codecs), backend.max_size)
```

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
BT.709 primaries, transfer and matrix, written into the H.264 video usability
information along with the frame rate.

With `bframes` above zero the encoder holds pictures back and `encode()` returns
an empty list until it lets several go at once. Packets then arrive in decode
order carrying display timestamps, and `MP4Writer` records the difference as
composition offsets. An empty list is an ordinary answer at any setting: never
assume one frame in means one packet out.

## Documentation

- [docs/usage.md](docs/usage.md) — the recording loop, every encoder setting,
  timing, colour, reordering, muxing, errors and limits.
- [docs/development.md](docs/development.md) — how the binding works, the ABI
  harness, and what a new backend has to implement.
- [plans/](plans/) — the design and its open questions:
  [the whole picture](plans/GPU-VIDEO-ENCODE.md),
  [Intel and AMD](plans/VAAPI-BACKEND.md),
  [Windows](plans/WINDOWS-SUPPORT.md).

## Development

```bash
pip install -e '.[dev]'
pytest
```

Tests that need an encoder or a GL context skip themselves without one, so the
suite runs anywhere. The hardware tests use a hidden GLFW window and synthetic
frames.

The NVENC binding is hand-written ctypes over a large C ABI, so
`tests/test_nvenc_abi.py` checks every structure against sizes and offsets
recorded from NVIDIA's header — it runs with no compiler, no driver and no GPU.
Re-record after changing a structure, or when moving to a newer header:

```bash
python tools/record_nvenc_abi.py path/to/nvEncodeAPI.h
```

## Licence

BSD-3-Clause; see `license.txt`. It contains no third-party code — see
`NOTICES.md` for where the NVENC ABI facts come from and under what terms.
