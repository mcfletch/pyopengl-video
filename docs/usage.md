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
    ring = [encoder.register(texture) for texture in make_textures(encoder.input_slots)]
    step = encoder.timescale // 60

    with MP4Writer('out.mp4', encoder) as movie:
        for index in range(600):
            render_one_frame()
            handle = ring[index % len(ring)]
            copy_the_frame_into(handle.texture)          # flipping as it copies
            movie.write(encoder.encode(handle, timestamp=index * step))
        movie.write(encoder.flush())
```

Four things are happening, and each has a section below: a texture is registered
once and encoded from many times; the finished frame is copied into it; the
frame is stamped with a time; and whatever the encoder produced goes to the
muxer. `examples/record_triangle.py` is this loop around a working renderer.

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

All are keyword arguments to `open_encoder`.

| Setting | Default | What it does |
| --- | --- | --- |
| `codec` | `'h264'` | The codec to produce. |
| `fps` | `60` | Declared frame rate, a number or `(numerator, denominator)`. A float becomes the broadcast ratio for it, so `29.97` records as `30000/1001`. It sets what the stream says and what rate control believes about time; it paces nothing. |
| `bitrate` | derived | Target bits per second. The default is the frame size times the rate times `DEFAULT_BITS_PER_PIXEL`, which puts 1080p60 near 8.7 Mbit/s. |
| `preset` | `'p4'` | `'p1'` (fastest) to `'p7'` (best quality). |
| `tuning` | `'high_quality'` | What the preset is trading against: `'high_quality'`, `'low_latency'`, `'ultra_low_latency'`, `'lossless'`, `'ultra_high_quality'`. |
| `rate_control` | `'vbr'` | `'vbr'`, `'cbr'`, or `'constqp'` to spend bits by content rather than to a target. |
| `gop` | two seconds | Frames between key frames. Every group opens with an IDR, which is where a player can seek to. `gop=1` makes every frame a key frame. |
| `bframes` | `0` | B-pictures between reference pictures. Better compression, at the cost of reordering — see below. |

Unknown values are refused by name, listing what is known.

## Getting the frame to the encoder

The encoder reads a texture, so the finished frame has to be in one. A renderer
that draws into a framebuffer copies it across with a blit:

```python
glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)           # or whatever was drawn into
glBindFramebuffer(GL_DRAW_FRAMEBUFFER, capture_framebuffer)
glBlitFramebuffer(0, 0, width, height,
                  0, height, width, 0,              # destination Y reversed: the flip
                  GL_COLOR_BUFFER_BIT, GL_NEAREST)
```

**The destination Y coordinates run backwards on purpose.** OpenGL's framebuffer
starts at the bottom left, and the encoder reads a texture from its first row and
calls that the top of the picture. Blitting the source's bottom edge to the
destination's top edge turns the frame over as it copies, which costs nothing.
Without it the video comes out upside down.

**Register as many textures as `encoder.input_slots` says, and cycle through
them.** An encoder that reorders frames is still reading a texture after
`encode()` has returned, and handing it back one it has not finished with raises
an error that says so. Two is the minimum even without reordering, which is what
a renderer wants anyway: it can be drawing the next frame while the encoder
reads the last.

The texture must be RGBA with eight bits a channel — `GL_RGBA8` — and exactly the
encoder's frame size.

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
| `EncoderUnavailable` | Nothing on this machine can encode what was asked for. The message lists what was available. |
| `EncoderError` | A texture still in flight was handed back; an unknown preset, tuning or rate control; encoding after `close()`. |
| `NVENCError` | The driver refused a call. Carries the status code, its name, and the driver's own explanation. |

`NVENCError` is an `EncoderError`, so a caller that only wants to know whether
recording failed can catch the one.

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
- **NVIDIA only, on Linux**, until the VA-API backend lands for Intel and AMD.
  NVIDIA supports the encoder's OpenGL device type on Linux alone; see
  `plans/WINDOWS-SUPPORT.md` for what Windows would take.
- **One context.** A session belongs to the OpenGL context that was current when
  it was opened, and every call must come from that context's thread.
- **No audio.** The muxer writes a video track.
