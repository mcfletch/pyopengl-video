# Using pyopengl-video

Recording what a renderer draws, without the frame leaving the GPU. Playing a
movie back into a texture is [docs/playback.md](playback.md).

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

`encoders()` lists only what is usable here: it loads each driver library,
checks the platform, and asks the hardware which codecs it encodes. A machine
with no encoder returns an empty list, and `open_encoder` raises
`EncoderUnavailable` naming what was asked for and what was available.

### Codecs

`codec` is `'h264'`, `'hevc'`, `'av1'` or `'vp9'`. `'h265'` and `'avc'` are
accepted too. A backend's record lists the codecs this machine's hardware
encodes through it, each with the largest frame it takes:

```python
for backend in encoders():
    for codec in sorted(backend.codecs):
        print(backend.name, codec, backend.largest(codec))

encoder = open_encoder(3840, 2160, codec='hevc', fps=60)
```

| Backend | Codecs it drives | How the hardware is asked |
| --- | --- | --- |
| `vaapi` | H.264, HEVC, AV1 | the device's profile list, when discovery runs |
| `nvenc` | H.264, HEVC, AV1 | the encoder session, which needs a current OpenGL context on the NVIDIA GPU. Without one the record lists all three, and building the encoder refuses a codec the GPU lacks, naming those it has. |
| `vpl` | H.264, HEVC, AV1, VP9 | not asked at discovery. The runtime refuses a codec the part lacks when the encoder is built. |

A codec nothing here encodes raises `EncoderUnavailable`, listing each
backend's codecs. `encoder.codec` is the name of the codec being produced.

## Encoder settings

All are keyword arguments to `open_encoder`. These are taken by every backend:

| Setting | Default | What it does |
| --- | --- | --- |
| `codec` | `'h264'` | The codec to produce: `'h264'`, `'hevc'`, `'av1'` or `'vp9'`, where the hardware encodes it. |
| `fps` | `60` | Declared frame rate, a number or `(numerator, denominator)`. A float becomes the broadcast ratio for it, so `29.97` records as `30000/1001`. It sets what the stream says and what rate control believes about time; it paces nothing. |
| `bitrate` | derived | Target bits per second. The default is the frame size times the rate times `DEFAULT_BITS_PER_PIXEL`, which puts 1080p60 near 8.7 Mbit/s. |
| `gop` | two seconds | Frames between key frames. Every group opens with an IDR picture or an AV1 or VP9 key frame, which is where a player can seek to. `gop=1` makes every frame a key frame. |
| `bframes` | `0` | B-pictures between reference pictures. Better compression, at the cost of reordering — see below. |
| `rate_control` | per backend | How the bits are spent; the names differ, below. |

`nvenc` also takes:

| Setting | Default | What it does |
| --- | --- | --- |
| `preset` | `'p4'` | `'p1'` (fastest) to `'p7'` (best quality). |
| `tuning` | `'high_quality'` | What the preset is trading against: `'high_quality'`, `'low_latency'`, `'ultra_low_latency'`, `'lossless'`, `'ultra_high_quality'`. |
| `rate_control` | `'vbr'` | `'vbr'`, `'cbr'`, or `'constqp'`. |

`vpl` also takes:

| Setting | Default | What it does |
| --- | --- | --- |
| `preset` | `'p4'` | `'p1'` (fastest) to `'p7'` (best quality). |
| `rate_control` | `'vbr'` | `'vbr'`, `'cbr'`, or `'constqp'`. |
| `qp` | `26` | The quantiser `'constqp'` codes every picture at, 0 to 51. The other modes do not read it. |
| `adapter` | the OpenGL context's | Which GPU to encode on. One that is not Intel's is refused. |

On `vpl`, `gop` is 1 to 65535, `bframes` 0 to 65534, and a width or height is
at most 65520; a value outside its range is refused before anything is opened.
Each input is submitted in a surface of its own, so an input the encoder has
not finished reading is refused: cycle through `input_slots` of them. Close
the encoder with its OpenGL context current. One that is collected unclosed
gives back its session, warns with a `ResourceWarning`, and leaves its
textures and framebuffers to the context.

`vaapi` also takes:

| Setting | Default | What it does |
| --- | --- | --- |
| `rate_control` | `'cbr'` | `'cbr'`, `'vbr'`, or `'cqp'` to code at a fixed quantiser. A mode the driver does not offer is refused, naming the ones it does. |
| `qp` | `26` | The quantiser, 0 to 51. What `'cqp'` codes at, and what the other modes start from. Lower is better quality and more bits. The scale is H.264's and HEVC's; for AV1 it is multiplied by five onto AV1's own scale of 1 to 255. |
| `device` | first that has an encoder | Which DRM render node to open, such as `'/dev/dri/renderD128'`. |
| `bframes` | `0` | Must be zero: this backend codes pictures in display order. |

On `vaapi`, a width or height is even and at most 65472, `bitrate` is 1 to
2147483647, `gop` 1 to 4294967295 and `qp` 0 to 51. libva carries the frame
rate as two 16-bit halves, so the ratio in lowest terms has to fit them:
`(30000, 1001)` does, `(90000, 3003)` is carried as that same ratio, and
`(100003, 1001)` is refused. A value outside its range is refused before a
device is opened. A texture handed to `register()` has to be `GL_RGBA8` of the
encoder's size; OpenGL is asked, and another size or format is refused.

An `encode()` that fails once the picture has gone to the driver stops a
`vaapi` encoder, since the pictures after it would predict from one the stream
does not hold. Later calls to `encode()` are refused, `flush()` still returns
the packets of the pictures coded before the failure, and `close()` releases
everything. A coded picture the driver marks as overflowing its buffer, or as
a bad bitstream, is such a failure and is not returned as a packet.

Close a `vaapi` encoder, and each input from `new_input()`, with the OpenGL
context they were made in current. The libva objects, the display and the
render node are released whichever context is current. The framebuffers,
textures and EGL images are objects of that one context: with another
current, or none, they are kept, `EncoderError` says so, and closing again
with the right context current gives them back.

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
frame at the declared rate, rounded to the timescale. `MP4Writer` takes each
sample's duration from the gap to the next timestamp in display order, and
uses a packet's own `duration` only for the last sample, so a rate that is not
a whole number of ticks (24000/1001 fps is 3753.75 at 90 kHz) does not drift
from the timestamps. Timestamps that repeat leave it using the packets'
durations. A frame rate of no frames per second raises `EncoderError`.

**Count timestamps from the start of the recording.** An MP4 track's decode
times start at zero, and the file states each display time as a signed 32-bit
offset from a decode time and each duration in 32 bits. `MP4Writer.write`
raises `MP4Error` for a packet those fields cannot state, and writes nothing
of it, so the packets before it still close into a whole movie:

| Refused | The limit, in the timescale's units |
| --- | --- |
| a first timestamp too far from zero | under 2^30 (3.3 hours at 90 kHz), and none below -2^30 |
| a timestamp far behind the latest | at most 2^30 behind |
| a gap to the next timestamp | at most 2^32 - 1 (13.2 hours at 90 kHz) |
| a duration | 0 to 2^32 - 1 |
| a timestamp or a duration that is not a whole number | |

A recording may be any length. One over 2^32 - 1 units long is written with
64-bit `mdhd`, `tkhd` and `mvhd` boxes wherever a duration needs them. With a
clock that does not start at zero, such as a monotonic one, subtract its
first reading from each timestamp.

## Colour

The hardware converts RGB to YUV, and the stream declares what it did: BT.709
primaries, BT.709 transfer, BT.709 matrix, limited range (16-235), with the frame
rate in the timing information. H.264 and HEVC carry the declaration in their
video usability information and AV1 in its sequence header. A player that
reads the declaration reproduces the colours it was given. Every codec is
coded at 8 bits, 4:2:0.

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

`MP4Writer` takes the codec, frame size, timescale and configuration from the
encoder, and writes an `avc1`, `hvc1`, `av01` or `vp09` track:

```python
with MP4Writer('out.mp4', encoder) as movie:
    movie.write(packets)          # a packet, or any iterable of them
```

Compressed data goes into the file as it arrives, so a long recording costs no
more memory than a short one; the sample tables are written when the movie
closes. That means the file is not ready to stream over a network without a
further pass — the trade for not buffering the video.

The width and the height are each from 1 to 65535 and the timescale from 1 to
2^32 - 1; `MP4Writer` and `IVFWriter` raise `MP4Error` and `IVFError` for
anything else before they create the file. A configuration the sample entry
cannot be made from, such as a parameter set cut short, is refused by
`write()` with the packet that carried it.

When the `with` block ends in an exception, the writer finishes the file with
the packets it has and that exception goes on; a failure to finish is logged
and not raised. `close()` on a movie with no samples removes the file and
raises `MP4Error`. A movie whose samples came with no parameter sets or
sequence header cannot be indexed: `close()` raises `MP4Error` and leaves the
file as an `mdat` box with no `moov`, which no player opens.

What a packet holds depends on the codec:

| Codec | `packet.data` | `encoder.headers()` | Stored in MP4 as |
| --- | --- | --- | --- |
| H.264 | an Annex-B access unit | SPS and PPS, Annex-B | `avc1`, the sets in `avcC` |
| HEVC | an Annex-B access unit | VPS, SPS and PPS, Annex-B | `hvc1`, the sets in `hvcC` |
| AV1 | a temporal unit: OBUs with size fields | the sequence header OBU | `av01`, the header in `av1C`, temporal delimiters dropped |
| VP9 | one frame | nothing | `vp09`, with a `vpcC` describing profile 0 |

H.264 and HEVC packets can be written straight to a `.h264` or `.h265` file:
they are an Annex-B elementary stream, which players and `ffmpeg` read
directly.

```python
with open('out.h265', 'wb') as raw:
    for packet in packets:
        raw.write(packet.data)
```

AV1 and VP9 have no such raw form, so `IVFWriter` writes theirs: a 32-byte
header and each packet behind its size and timestamp, which `ffmpeg`, the
reference decoders and `Movie.open` read. An IVF timestamp is a whole number
from 0 to 2^64 - 1, and `write()` raises `IVFError` for any other.

```python
from pyopengl_video.ivf import IVFWriter

with open_encoder(1280, 720, codec='av1') as encoder, \
        IVFWriter('out.ivf', encoder) as stream:
    ...
    stream.write(encoder.encode(handle, timestamp))
```

To mux with something else, `Packet` carries everything a container needs:
`data`, `timestamp`, `duration`, `keyframe`.

## Errors

| Raised | When |
| --- | --- |
| `EncoderUnavailable` | Nothing on this machine can encode what was asked for. The message lists what was available, and why any backend that declined for a fixable reason did so. |
| `EncoderError` | A texture still in flight was handed back; an unknown preset, tuning or rate control; a size or setting outside its range, or one the driver refuses; encoding after `close()`, or after an `encode()` that failed; closing with another OpenGL context current than the one the inputs were made in. |
| `NVENCError` | NVIDIA's driver refused a call. Carries the status code, its name, and the driver's own explanation. |
| `VPLError` | Intel's runtime refused a call, with the same. |
| `MP4Error`, `IVFError` | A writer was given a size, a timescale, a timestamp or a configuration its file cannot state. Both are `ContainerError`s, which are `VideoError`s and `ValueError`s. |

**One `except EncoderError` is enough** to know whether recording failed, on
every backend. `NVENCError` and `VPLError` are `EncoderError`s. The VA-API
backend reaches the same place from the other direction: libva's own `VAError`
and the DMA-BUF shim's `DMABufError` are deliberately *not* encoder errors —
that shim is shared by every Linux backend and libva decodes as well as
encodes, so neither is an encoder's failure until an encoder is what raised it —
and `VAAPIEncoder` translates them at its own boundary, leaving the original on
`__cause__` where a traceback shows it. It translates every other exception a
setting or a driver brings about the same way, so `IndexError` or
`MemoryError` does not reach a caller from it either; `KeyboardInterrupt` and
`SystemExit` pass through.

So a caller catches one thing, and whoever reads the traceback still sees which
driver said no and what it said. A caller using `pyopengl_video.linux.dmabuf`
directly, without an encoder, catches `DMABufError` there instead. That is
also what `export_texture` raises for a texture that is not `GL_RGBA8` of the
size stated, and what closing an exported or imported image raises with
another OpenGL context current than the one it was made in. It names the
texture or the buffer that was refused and the layout it was given, whichever
entry point declined and whether that driver declines by answering with a null
handle or by raising.

`NV_ENC_ERR_UNSUPPORTED_DEVICE` from `nvEncOpenEncodeSessionEx` has two
causes and the message says which one it met: no OpenGL context current on the
calling thread, or a context that is current but draws with something other
than the NVIDIA GPU holding the encoder. The second is what a software
renderer or the integrated part of a hybrid machine gives, and the message
names the renderer it found.

## Performance

Measured on an RTX 3060 Ti: encoding 720p costs well under a millisecond a
frame, and the hardware advertises about 983,000 macroblocks a second for
H.264 — roughly 1080p at 120 fps. The encode is not what limits a recording.

What does cost something is the copy into the capture texture, which is a
GPU-side blit of one full frame, and the renderer's own work. A recording of a
game is the game plus a blit.

## Limits

- **8-bit 4:2:0 in every codec.** HEVC Main and AV1 Main, from a `GL_RGBA8`
  texture. The largest frame is the hardware's for each codec:
  `backend.largest(codec)`.
- **AV1 on `nvenc`, and HEVC, AV1 and VP9 on `vpl`, have not been run.** They
  are written from the vendors' headers, their structures are checked against
  those headers, and their settings are tested without a driver. H.264 has
  run on both, and HEVC on `nvenc`: on an RTX 3060 Ti, with and without
  B-frames, to 5120x2880, into MP4 files that `ffmpeg` and this package's
  NVDEC decoder both read back. AV1 needs an RTX 40-series GPU or later; the
  hardware cases in `tests/test_nvenc_encode_codecs.py` run for it there.
- **AV1 on AMD pads the frame.** The VCN encoder codes a frame of at least
  128x128 whose size it rounds up to its own blocks, and writes the size asked
  for as the stream's render size. A 1920x1080 recording is a 1920x1082 frame
  with two black rows; a player that follows the render size scales the frame
  to 1920x1080. Sizes the encoder does not round, such as 1280x720 and
  3840x2160, are coded as asked.
- **NVIDIA on Linux and Windows, AMD on Linux, Intel on Windows.** Intel on
  Linux is the `vaapi` backend against a different driver and has not been
  run. `plans/GPU-VIDEO-ENCODE.md` holds the whole matrix.
- **`nvenc` on Windows needs the OpenGL context on the NVIDIA GPU.** The
  encoder reads a Direct3D texture that OpenGL draws into, and both must be on
  one GPU. On a laptop with switchable graphics a program gets the integrated
  GPU unless Windows is told otherwise: Settings, System, Display, Graphics,
  then add the Python executable and choose High performance. Without that
  `encoders()` omits `nvenc` and `open_encoder(..., backend='nvenc')` says why.
  Inputs come from `encoder.new_input()`; `register()` is refused there.
- **`vaapi` needs an EGL context**, because exporting a texture as a DMA-BUF is
  an EGL extension — see *Getting the frame to the encoder*.
- **`vaapi` codes in display order.** `bframes` above zero is refused there,
  in every codec.
- **One context.** A session belongs to the OpenGL context that was current when
  it was opened, and every call must come from that context's thread, `close()`
  included: an OpenGL name is deleted only in the context that made it.
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

On Linux the driver answers the same question when the session is opened:
NVENC takes the current OpenGL context as its device, and refuses a context
that is not on the GPU holding the encoder. A context that has landed on a
software renderer fails there too, which is what an EGL context on a display
server with no hardware EGL driver gives.
