# Using pyopengl-video

Recording what a renderer draws, without the frame leaving the GPU.

- [Installing](#installing)
- [The recording loop](#the-recording-loop)
- [Choosing an encoder](#choosing-an-encoder)
- [Encoder settings](#encoder-settings)
- [Getting the frame to the encoder](#getting-the-frame-to-the-encoder)
- [Timing](#timing)
- [Colour](#colour)
- [Reordering, and why `encode` returns a list](#reordering-and-why-encode-returns-a-list)
- [Writing a file](#writing-a-file)
- [Errors](#errors)
- [Performance](#performance)
- [Limits](#limits)

## Installing

```bash
pip install pyopengl-video
```

The encoder itself comes with the graphics driver and is loaded by name, so
there is nothing else to install and no build step. PyOpenGL is the only
dependency.

## The recording loop

```python
from pyopengl_video import open_encoder
from pyopengl_video.mp4 import MP4Writer

with open_encoder(1920, 1080, fps=60, bitrate=12_000_000) as encoder:
    ring = [encoder.new_input() for _ in range(encoder.input_slots)]
    step = encoder.timescale // 60

    with MP4Writer('out.mp4', encoder) as movie:
        for index in range(600):
            render_one_frame()
            handle = ring[index % len(ring)]
            with handle.for_drawing():
                copy_the_frame_into(handle.framebuffer)  # flipping as it copies
            movie.write(encoder.encode(handle, timestamp=index * step))
        movie.write(encoder.flush())
```

Four things are happening, and each has a section below: the encoder is asked
for the textures it will read; the finished frame is copied into one; the frame
is stamped with a time; and whatever the encoder produced goes to the muxer.
`examples/record_triangle.py` is this loop around a working renderer.

## Choosing an encoder

`open_encoder` takes the first backend that can do what was asked. To see what a
machine offers, or to choose deliberately:

```python
from pyopengl_video import encoders, open_encoder

for backend in encoders():
    print(backend.name, backend.vendor, sorted(backend.codecs),
          backend.max_size, 'zero-copy' if backend.zero_copy else 'copies')

encoder = open_encoder(1280, 720, backend='nvenc')
```

`encoders()` lists only what is usable here: it loads each driver library and
checks the platform. A machine with no encoder returns an empty list, and
`open_encoder` raises `EncoderUnavailable` naming what was asked for and what
was available.

## Encoder settings

All are keyword arguments to `open_encoder`. These are taken by every backend:

| Setting | Default | What it does |
| --- | --- | --- |
| `codec` | `'h264'` | The codec to produce. |
| `fps` | `60` | Declared frame rate, a number or `(numerator, denominator)`. A float becomes the broadcast ratio for it, so `29.97` records as `30000/1001`. It sets what the stream says and what rate control believes about time; it paces nothing. |
| `bitrate` | derived | Target bits per second. The default is the frame size times the rate times `DEFAULT_BITS_PER_PIXEL`, which puts 1080p60 near 8.7 Mbit/s. |
| `gop` | two seconds | Frames between key frames. Every group opens with an IDR, which is where a player can seek to. `gop=1` makes every frame a key frame. |
| `bframes` | `0` | B-pictures between reference pictures. Better compression, at the cost of reordering — see below. |
| `rate_control` | per backend | How the bits are spent; the names differ, below. |

`nvenc` also takes:

| Setting | Default | What it does |
| --- | --- | --- |
| `preset` | `'p4'` | `'p1'` (fastest) to `'p7'` (best quality). |
| `tuning` | `'high_quality'` | What the preset is trading against: `'high_quality'`, `'low_latency'`, `'ultra_low_latency'`, `'lossless'`, `'ultra_high_quality'`. |
| `rate_control` | `'vbr'` | `'vbr'`, `'cbr'`, or `'constqp'`. |

`vaapi` also takes:

| Setting | Default | What it does |
| --- | --- | --- |
| `rate_control` | `'cbr'` | `'cbr'`, `'vbr'`, or `'cqp'` to code at a fixed quantiser. A mode the driver does not offer is refused, naming the ones it does. |
| `qp` | `26` | The quantiser, 0 to 51. What `'cqp'` codes at, and what the other modes start from. Lower is better quality and more bits. |
| `device` | first that has an encoder | Which DRM render node to open, such as `'/dev/dri/renderD128'`. |
| `bframes` | `0` | Must be zero: this backend codes pictures in display order. |

Unknown values are refused by name, listing what is known.

## Getting the frame to the encoder

**On Linux, make the context an EGL context.** A Linux encoder other than
NVIDIA's is handed the frame as a DMA-BUF exported from the texture, which is
an EGL extension; GLFW makes a GLX context by default on X11, and a GLX context
cannot export one:

```python
glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
window = glfw.create_window(width, height, 'recording', None, None)
```

The `vaapi` backend reports itself unavailable from a context that cannot
export, rather than failing when a texture is handed over, and `open_encoder`
says so rather than leaving a caller to guess:

```
EncoderUnavailable: no encoder here can produce 1920x1080 h264; available: none
  vaapi: this OpenGL context is not an EGL context, so it cannot export a
  texture as a DMA-BUF; create the window with
  glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
```

To ask directly:

```python
from pyopengl_video.linux import dmabuf

print(dmabuf.unavailable_because() or 'this context can export')
```

**A machine with no EGL at all answers the same way.** EGL ships with the
graphics driver, so a virtual machine, a container built without one or a CI
runner may have no `libEGL` for PyOpenGL to bind to. `unavailable_because()`
says so, `vaapi` reports itself unavailable, and nothing raises: it is one more
reason a context cannot export. Installing a driver — `libegl1` or `libglvnd`
on most distributions — is what changes the answer.

**X11 is not the difficulty; GLX is.** EGL runs on X11 as well as on Wayland,
so an X11 application records through `vaapi` perfectly well — it just has to
ask for an EGL context, which is the one hint above. What cannot export is a
*GLX* context, and there is no GLX counterpart to
`EGL_MESA_image_dma_buf_export` to reach for.

If an application is tied to GLX — an older toolkit that offers nothing else —
then on that context:

- **`nvenc` still records.** It takes the texture by name and does not care how
  the context was made.
- **`vaapi` cannot**, and no arrangement of this library changes that: objects
  are not shared between a GLX context and an EGL one, so the frame would have
  to travel through host memory to reach an EGL context that could export it.

That last route is a real one, and it is the readback tier in
`plans/GPU-VIDEO-ENCODE.md`: a fenced `glReadPixels` into the encoder's
host-memory input, one crossing each way and no pipeline stall. It is designed
and not yet written, and it is what would let any context on any driver record.

**Ask the encoder for its inputs.** `new_input()` returns a handle carrying a
texture the encoder can read and a framebuffer with that texture attached:

```python
handle = encoder.new_input()
handle.texture        # the OpenGL texture name
handle.framebuffer    # a framebuffer with it as the colour attachment
```

The encoder allocates them because on some platforms it is the only side that
can. Intel's encoder on Windows reads a Direct3D 11 surface, which has to exist
as a Direct3D resource before OpenGL can be given a name for it, so the backend
creates the pair and hands both back. `encoder.allocates_inputs` says when
`register()` — for a caller bringing a texture of its own — is not available.

**Draw inside `for_drawing()`.** That scope is where a surface shared with
another graphics API changes hands, and it is what orders the renderer's writes
against the encoder's reads. On a backend sharing nothing it does nothing, so
the same loop runs everywhere:

```python
with handle.for_drawing():
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)       # or whatever was drawn into
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, handle.framebuffer)
    glBlitFramebuffer(0, 0, width, height,
                      0, height, width, 0,          # destination Y reversed: the flip
                      GL_COLOR_BUFFER_BIT, GL_NEAREST)
packets = encoder.encode(handle, timestamp)
```

**The destination Y coordinates run backwards on purpose.** OpenGL's framebuffer
starts at the bottom left, and the encoder reads a texture from its first row and
calls that the top of the picture. Blitting the source's bottom edge to the
destination's top edge turns the frame over as it copies, which costs nothing.
Without it the video comes out upside down.

**Take as many inputs as `encoder.input_slots` says, and cycle through them.**
An encoder that reorders frames is still reading a texture after `encode()` has
returned, and handing it back one it has not finished with raises an error that
says so. Two is the minimum even without reordering, which is what a renderer
wants anyway: it can be drawing the next frame while the encoder reads the last.

A texture handed to `register()` must be RGBA with eight bits a channel —
`GL_RGBA8` — and exactly the encoder's frame size. One from `new_input()` already
is.

## Timing

Timestamps and durations are integers in the encoder's `timescale`, 90 kHz by
default, and the encoder gives back what it is given. A frame every 1/60 s is
`timestamp += encoder.timescale // 60`.

Stamp frames from a **fixed step** rather than from the wall clock: frame *n* is
at *n*/fps, however long it took to render. The recording is then smooth and
reproducible, and a slow frame stretches the recording rather than jerking it.
Wall-clock stamps make a variable-rate file, which the muxer will write happily
and some players will handle poorly.

Pass `duration` explicitly for a variable-rate recording; left at zero it is one
frame at the declared rate.

## Colour

The hardware converts RGB to YUV, and the stream declares what it did: BT.709
primaries, BT.709 transfer, BT.709 matrix, limited range (16-235), with the frame
rate in the timing information. A player that reads the declaration reproduces
the colours it was given.

This is fixed for now. Full-range or a different matrix means converting in a
shader before the encoder sees the frame, and telling the encoder to leave the
result alone.

## Reordering, and why `encode` returns a list

`encode()` returns a **list of packets**, which may be empty:

```python
packets = encoder.encode(handle, timestamp)     # 0, 1, or several
```

With `bframes` above zero, a B-picture is predicted from the frame after it, so
the encoder cannot emit one until that frame arrives. Several calls return
nothing and then one returns a handful. `flush()` at the end returns whatever is
still inside; without it the recording loses its last frames.

Packets then arrive in **decode order**, each carrying its **display**
timestamp, so the two orders differ. `encoder.reorders_frames` says whether that
can happen, and `MP4Writer` records the difference as composition offsets so
players show frames in the right order. Code that writes its own container has
to do the same.

Even with `bframes=0` the rule holds: never assume one frame in means one packet
out.

## Writing a file

`MP4Writer` takes the frame size, timescale and parameter sets from the encoder:

```python
with MP4Writer('out.mp4', encoder) as movie:
    movie.write(packets)          # a packet, or any iterable of them
```

Compressed data goes into the file as it arrives, so a long recording costs no
more memory than a short one; the sample tables are written when the movie
closes. That means the file is not ready to stream over a network without a
further pass — the trade for not buffering the video.

Packets can also be written straight to a `.h264` file: they are an Annex-B
elementary stream, which players and `ffmpeg` read directly.

```python
with open('out.h264', 'wb') as raw:
    for packet in packets:
        raw.write(packet.data)
```

To mux with something else, `Packet` carries everything a container needs:
`data`, `timestamp`, `duration`, `keyframe`.

## Errors

| Raised | When |
| --- | --- |
| `EncoderUnavailable` | Nothing on this machine can encode what was asked for. The message lists what was available, and why any backend that declined for a fixable reason did so. |
| `EncoderError` | A texture still in flight was handed back; an unknown preset, tuning or rate control; a size or setting the driver refuses; encoding after `close()`. |
| `NVENCError` | NVIDIA's driver refused a call. Carries the status code, its name, and the driver's own explanation. |
| `VPLError` | Intel's runtime refused a call, with the same. |

**One `except EncoderError` is enough** to know whether recording failed, on
every backend. `NVENCError` and `VPLError` are `EncoderError`s. The VA-API
backend reaches the same place from the other direction: libva's own `VAError`
and the DMA-BUF shim's `DMABufError` are deliberately *not* encoder errors —
that shim is shared by every Linux backend and libva decodes as well as
encodes, so neither is an encoder's failure until an encoder is what raised it —
and `VAAPIEncoder` translates them at its own boundary, leaving the original on
`__cause__` where a traceback shows it.

So a caller catches one thing, and whoever reads the traceback still sees which
driver said no and what it said.

## Performance

Measured on an RTX 3060 Ti: encoding 720p costs well under a millisecond a
frame, and the hardware advertises about 983,000 macroblocks a second for
H.264 — roughly 1080p at 120 fps. The encode is not what limits a recording.

What does cost something is the copy into the capture texture, which is a
GPU-side blit of one full frame, and the renderer's own work. A recording of a
game is the game plus a blit.

## Limits

- **H.264 only**, up to 4096x4096. HEVC and AV1 are hardware the parts have and
  the library does not use yet.
- **NVIDIA and AMD on Linux, Intel on Windows.** Intel on Linux is the `vaapi`
  backend against a different driver and has not been run; NVIDIA on Windows
  goes through the same Direct3D interop the Intel backend uses, which is
  written and not yet wired to that encoder. `plans/GPU-VIDEO-ENCODE.md` holds
  the whole matrix.
- **`vaapi` needs an EGL context**, because exporting a texture as a DMA-BUF is
  an EGL extension — see *Getting the frame to the encoder*.
- **`vaapi` codes in display order.** `bframes` above zero is refused there.
- **One context.** A session belongs to the OpenGL context that was current when
  it was opened, and every call must come from that context's thread.
- **No audio.** The muxer writes a video track.

## Which GPU the encoder is on

On a machine with more than one GPU, an encoder can only read the renderer's
frames without a copy if it is on the *same* GPU. A laptop with switchable
graphics often runs OpenGL on the integrated part while a discrete one sits idle,
so the Windows backends ask the current context which adapter it is on and offer
themselves only there:

```python
from pyopengl_video.windows import interop

adapter = interop.adapter_for_context()
print(adapter)        # Intel(R) UHD Graphics 630 (Intel, LUID 6334010000000000)
```

`open_encoder` then picks the backend whose vendor owns that adapter, which is
why an Intel-rendered frame records through Intel's encoder even on a machine
that also has an NVIDIA part. Which GPU a context lands on is a per-application
driver setting — the Windows graphics preference, or the vendor's control panel —
and not something this library changes.
