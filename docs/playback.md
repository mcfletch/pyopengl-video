# Playing video into OpenGL textures

`pyopengl-video` decodes H.264 on the GPU's video decoder and hands each
picture to OpenGL as textures. The frame never passes through host memory. A
conversion pass turns the decoded YUV into an RGBA texture, the right way up
for OpenGL. Any PyOpenGL program can play a movie; no scenegraph is needed.

```python
from pyopengl_video.playback import Movie

movie = Movie.open('intro.mp4')          # any thread: reads the index, no GL
player = movie.player()                  # GL thread: decoder, converter, textures
texture = player.frame_at(seconds)       # GL_SRGB8_ALPHA8, row 0 at the bottom
...
player.close()
movie.close()
```

`examples/play_movie.py` plays a movie in a GLFW window. Space pauses, the
arrow keys seek, and `--screenshot` saves one frame to a PNG.

## What it needs

Python 3.10 or newer, PyOpenGL, and a GPU with a decoder this package drives.
The decoder library ships with the graphics driver and is loaded by name. The
MP4 and raw H.264 readers are part of this package.

| Backend | Hardware | Platform | Codec | Picture handed over as | State |
| --- | --- | --- | --- | --- | --- |
| `vaapi` | AMD, VCN | Linux | H.264 | two DMA-BUF planes imported as textures | working |
| `vaapi` | Intel, Gen9 and later | Linux | H.264 | the same | the same code, untested |
| `nvdec` | NVIDIA | Linux, Windows | H.264 | CUDA memory copied into textures | [planned](../plans/GPU-VIDEO-DECODE.md) |
| `vpl` | Intel | Windows | H.264 | a Direct3D 11 texture | [planned](../plans/GPU-VIDEO-DECODE.md) |

**The `vaapi` backend needs an EGL context.** Each decode surface reaches
OpenGL through `EGL_EXT_image_dma_buf_import`. Ask GLFW for one before the
window is created:

```python
glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
```

It also needs the VA-API driver for the GPU: `mesa-va-drivers` for AMD,
`intel-media-va-driver` for Intel.

`decoders()` lists what this machine can decode:

```python
from pyopengl_video import decoders

for backend in decoders():
    print(backend.name, backend.vendor, sorted(backend.codecs), backend.max_size)
```

## Movies and players

`Movie.open(path)` reads an MP4 file or a raw H.264 (Annex-B) stream. The first
bytes of the file say which. Opening reads the index and none of the media, so
a two-hour movie costs the size of its sample table. It needs no GL, so a game
can open movies on a loading thread.

| `Movie` | |
| --- | --- |
| `size` | the visible picture size, `(width, height)` |
| `frames` | how many pictures the video track holds |
| `duration` | the length in seconds, to the end of the last frame |
| `headers` | the stream's SPS and PPS as Annex-B |
| `player(backend=None, ahead=2, mipmaps=False, **options)` | a player in the current GL context |

A raw stream has no timestamps. Its pictures are timed in display order, at
the frame rate its sequence parameter set states. A stream that states none
plays at 25 frames a second, and `Movie.open(path, frame_rate=(30, 1))` sets
another rate.

`player.frame_at(seconds)` returns the texture showing the movie at that time.
That is the last frame shown at or before the time. Before the start it is the
first frame, and after the end the last. The texture is valid until the next
call. Times count from the first frame shown, whatever time an MP4's edit list
gives it.

| `Player` | |
| --- | --- |
| `frame_at(seconds)` | the texture for that time |
| `texture` | the texture last returned, or 0 |
| `frame` | the display index of the frame shown, or None |
| `ready_ahead` | converted frames waiting beyond the one shown |
| `close()` | closes the decoder and deletes the textures |

The program's clock decides the time. A paused clock gets the same texture
with no decoding. A fixed-step capture gets every frame, in order.

### Seeking

Playing forward decodes each sample once. The player keeps `ahead` converted
frames ready beyond the one shown. A time behind the frame shown resets the
decoder, and so does a time past the next key frame. Decoding then starts at
the key frame before that time. Playing backward decodes up to one group of
pictures per frame shown.

A key frame is an IDR picture, or a picture coded entirely of I slices. In a
stream with open groups of pictures, some pictures follow a key frame in
decoding order and precede it in display order. They refer to the group before
it, so after a seek the player drops them.

### Colour, orientation and size

The output texture is `GL_SRGB8_ALPHA8`, holding non-linear RGB. A shader
sampling it reads linear colour, as it does from an image texture. To draw it
straight to the screen, enable `GL_FRAMEBUFFER_SRGB` on an sRGB-capable
framebuffer (`glfw.SRGB_CAPABLE`). BT.709's transfer curve is treated as
sRGB's.

The matrix and range come from the stream's video usability information.
BT.601, BT.709 and BT.2020 are converted, at limited or full range. A stream
that states no matrix is BT.601 below 720 rows and BT.709 from 720 rows up.
Chroma sits where the stream's `chroma_sample_loc_type` puts it, left-sited by
default.

The output is the stream's cropped size. Row 0 is the bottom of the picture, so
ordinary texture coordinates draw it the right way up. With `mipmaps=True`,
mipmaps are made after each conversion.

## Decoding without the player

The player's parts are public, for a program that schedules its own decoding
or wants the planes.

```python
from pyopengl_video import open_decoder
from pyopengl_video.convert import NV12Converter, create_output_texture
from pyopengl_video.mp4 import MP4Reader

with MP4Reader('movie.mp4') as movie:
    track = movie.video
    decoder = open_decoder('h264', headers=track.parameter_sets())
    converter = NV12Converter()
    target = create_output_texture(*decoder.size)
    for index in range(len(track)):
        for frame in decoder.decode(track.sample(index)):
            converter.convert(frame, target)
            frame.release()
            draw(target)
    for frame in decoder.flush():
        ...
```

- `MP4Reader` reads an MP4 file's `moov` box and hands out samples by index.
  `track.sample(index, parameter_sets=False)` returns a `Sample` whose `data`
  is Annex-B, with the SPS and PPS in front when asked. Its `timestamp` is the
  presentation time after the edit list, in `track.timescale` units.
  `display_order()`, `sync_samples` and `keyframe_before(index)` plan a seek.
  Fragmented MP4 is not read.
- `AnnexBReader` reads a raw stream the same way. It scans the stream once for
  access units and their display order.
- `decoder.decode(sample)` takes one access unit and returns a list of the
  frames now ready, in display order. The decoder holds pictures back until
  their order is known, so the list may be empty or hold several. `flush()`
  returns the rest at the end of a stream. `reset()` drops what is held,
  before decoding from a key frame.
- A `DecodedFrame` is lent by the decoder. Its `planes` are the decoder's
  surfaces: a `GL_R8` luma texture and a `GL_RG8` chroma texture, Cb in red
  and Cr in green. The picture sits at `visible`, first row at the top, inside
  textures of `texture_size`. Release every frame: a decoder with all its
  surfaces lent raises `DecoderError` naming the frames held. A caller may
  hold `output_slots` frames at once, 4 by default.
- `NV12Converter.convert(frame, target)` draws a frame into an RGBA texture of
  the frame's size. One pass applies colour, crop and the vertical flip. It
  restores every piece of GL state it changes, so it can run inside a
  renderer's frame.

`MP4Reader.audio` reports each audio track's codec, sample rate, channels and
configuration, and reads its samples. No audio is decoded yet; see *Audio* in
the [plan](../plans/GPU-VIDEO-DECODE.md).

## What is decoded

- H.264 Constrained Baseline, Main and High profiles, 8-bit 4:2:0, progressive
  frames. Several slices a picture, CAVLC and CABAC, B-frames and B-frame
  pyramids, weighted prediction, long-term references and scaling matrices are
  all decoded.
- Refused, with an error naming the feature: interlaced coding (field pictures
  and MBAFF), the High 10, 4:2:2 and 4:4:4 profiles, and slice groups.
- Sizes from 64x64 to 4096x4096 on AMD VCN, as the driver reports them. A
  larger picture is refused by its size. A smaller one decodes into a
  minimum-size surface, and only its visible part is converted.

## Errors

A decoder raises `DecoderError`, with libva's, the DMA-BUF shim's or the
parser's error as its cause. `DecoderUnavailable` means no backend here
decodes the stream. Its message includes each unavailable backend's reason.
`EncoderError` and `DecoderError` share the base `VideoError`, so one
`except VideoError` covers recording and playback.

## Performance

A 1920x1080 60 fps H.264 High stream with three B-frames takes 0.75 ms a frame
to decode and convert, or 1300 frames a second. That was measured on a Radeon
8060S with Mesa 25.2.8, frame by frame on the GL thread. Most of the time is
spent waiting for the hardware decode before the conversion reads the surface.
Decoding on a worker thread is phase D7 of the
[plan](../plans/GPU-VIDEO-DECODE.md).

Each decode surface is imported into OpenGL once, when the decoder opens, so
decoding creates no GL objects.

## Threading

`Movie`, `MP4Reader` and `AnnexBReader` need no GL and work from any thread. A
decoder, a converter and a player belong to the thread whose OpenGL context
was current when they were made.
