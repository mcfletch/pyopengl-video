# Playing video into OpenGL textures

`pyopengl-video` decodes H.264 on the GPU's own video decoder and hands each
picture to OpenGL as textures, with the frame never passing through host
memory. A conversion pass turns the decoded YUV into an RGBA texture the right
way up for OpenGL. Nothing here needs a scenegraph: any PyOpenGL program can
play a movie.

```python
from pyopengl_video.playback import Movie

movie = Movie.open('intro.mp4')          # any thread: reads the index, no GL
player = movie.player()                  # GL thread: decoder, converter, textures
texture = player.frame_at(seconds)       # GL_SRGB8_ALPHA8, row 0 at the bottom
...
player.close()
movie.close()
```

`examples/play_movie.py` is a complete player in a GLFW window: a quad, a
clock that pauses and seeks, and a `--screenshot` mode that saves one frame.

## What it needs

Python 3.10 or newer, PyOpenGL, and a GPU whose decoder this package drives.
The decoder is the one in the graphics driver, loaded by name; the MP4 and raw
H.264 readers are part of this package.

| Backend | Hardware | Platform | Codec | Picture handed over as | State |
| --- | --- | --- | --- | --- | --- |
| `vaapi` | AMD, VCN | Linux | H.264 | two DMA-BUF planes imported as textures | working |
| `vaapi` | Intel, Gen9 and later | Linux | H.264 | the same | the same code, untested |
| `nvdec` | NVIDIA | Linux, Windows | H.264 | CUDA memory copied into textures | [planned](../plans/GPU-VIDEO-DECODE.md) |
| `vpl` | Intel | Windows | H.264 | a Direct3D 11 texture | [planned](../plans/GPU-VIDEO-DECODE.md) |

**The `vaapi` backend needs an EGL context**, as the encoder does: each decode
surface reaches OpenGL through `EGL_EXT_image_dma_buf_import`. Ask GLFW for one
before the window is created:

```python
glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
```

It also needs a VA-API driver for the GPU beside `libva`: `mesa-va-drivers` for
AMD, `intel-media-va-driver` for Intel.

Ask what a machine can decode:

```python
from pyopengl_video import decoders

for backend in decoders():
    print(backend.name, backend.vendor, sorted(backend.codecs), backend.max_size)
```

## Movies and players

`Movie.open(path)` reads an MP4 file or a raw H.264 (Annex-B) stream, told
apart by their first bytes. Opening reads the index and none of the media, so
a two-hour movie costs the size of its sample table, and it needs no GL, so a
game can open movies on a loading thread.

| `Movie` | |
| --- | --- |
| `size` | the visible picture size, `(width, height)` |
| `frames` | how many pictures the video track holds |
| `duration` | its length in seconds, to the end of the last frame |
| `headers` | the stream's SPS and PPS as Annex-B |
| `player(backend=None, ahead=2, mipmaps=False, **options)` | a player in the current GL context |

A raw stream has no timestamps. Its pictures are timed by their display order
at the frame rate its sequence parameter set states, or at 25 frames a second
when it states none; `Movie.open(path, frame_rate=(30, 1))` overrides that.

`player.frame_at(seconds)` returns the texture showing the movie at that time:
the last frame shown at or before it, the first frame before the start, and
the last frame after the end. The texture is valid until the next call. Times
count from the first frame shown, so an MP4 whose edit list starts its
presentation a few frames into the media still starts at zero.

| `Player` | |
| --- | --- |
| `frame_at(seconds)` | the texture for that time |
| `texture` | the texture last returned, or 0 |
| `frame` | the display index of the frame shown, or None |
| `ready_ahead` | converted frames waiting beyond the one shown |
| `close()` | closes the decoder and deletes the textures |

The caller's clock decides the time. A paused clock asks for the same time and
gets the same texture with no decoding; a fixed-step capture asks for every
frame time in turn and gets every frame in order.

### Seeking

Playing forward decodes each sample once, and the player keeps `ahead`
converted frames ready beyond the one shown. A time behind the frame shown, or
past the next key frame, resets the decoder and decodes from the key frame
before the time asked for. Playing backward therefore decodes up to one group
of pictures per frame shown.

A key frame is an IDR picture, or, in a stream with open groups of pictures, a
picture coded entirely of I slices. Pictures that follow such a key frame in
decoding order and precede it in display order refer to the group before it,
so after a seek they are dropped rather than shown.

### Colour, orientation and size

The output texture is `GL_SRGB8_ALPHA8` with the decoded picture converted to
non-linear RGB. Sampled in a shader it reads as linear colour, as an image
texture does; drawn straight to the screen, enable `GL_FRAMEBUFFER_SRGB` on a
framebuffer that is sRGB-capable (`glfw.SRGB_CAPABLE`) so it is encoded again.
BT.709's transfer curve is treated as sRGB's.

The matrix and range come from the stream's video usability information:
BT.601, BT.709 and BT.2020 matrices, limited or full range. A stream that
states no matrix is taken as BT.601 below 720 rows and BT.709 from 720 rows up.
Chroma sits where the stream's `chroma_sample_loc_type` puts it, H.264's
left-sited default when it says nothing.

The output is the stream's cropped size, and row 0 is the bottom of the
picture, so ordinary texture coordinates draw it the right way up.

With `mipmaps=True` the output textures carry mipmaps, made after each
conversion.

## Decoding without the player

The pieces the player is made of are public, for a program that schedules its
own decoding or wants the planes.

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

- `MP4Reader` reads the `moov` box of an MP4 file and hands out samples by
  index. `movie.video.sample(index, parameter_sets=False)` is a `Sample` whose
  `data` is Annex-B, with the SPS and PPS in front when asked; `timestamp` is
  its presentation time with the edit list applied, in `track.timescale`
  units. `display_order()`, `sync_samples` and `keyframe_before(index)` are
  what seeking is planned from. Fragmented MP4 is not read.
- `AnnexBReader` reads a raw stream the same way, scanning it once to find
  each access unit and its display order.
- `decoder.decode(sample)` takes one access unit and returns the frames now
  ready, in display order. A list, because the decoder holds pictures back
  until it knows their order: empty is ordinary, and so is more than one.
  `flush()` returns the rest at the end of a stream, and `reset()` drops
  everything held before decoding from a key frame.
- A `DecodedFrame` is lent by the decoder. Its `planes` are the decoder's own
  surfaces as a `GL_R8` luma texture and a `GL_RG8` chroma texture (Cb in red,
  Cr in green), with the picture at `visible`, first row at the top, inside
  textures of `texture_size`. **Release every frame**: a decoder whose
  surfaces are all lent raises `DecoderError` naming the frames still held.
  `output_slots` (4 by default) is how many a caller may hold at once.
- `NV12Converter.convert(frame, target)` draws a frame into an RGBA texture of
  the frame's size, applying colour, crop and the vertical flip in one pass.
  It puts back every piece of GL state it changes, so it can run inside a
  renderer's frame.

Both readers report audio tracks too, `MP4Reader.audio`, with their codec,
sample rate, channels and configuration, and read their samples. Nothing here
decodes audio yet; see *Audio* in the [plan](../plans/GPU-VIDEO-DECODE.md).

## What is decoded

- H.264 Constrained Baseline, Main and High profiles: 8-bit, 4:2:0,
  progressive frames, one or more slices a picture, CAVLC or CABAC, B-frames
  and B-frame pyramids, weighted prediction, long-term references, scaling
  matrices.
- Refused, with an error naming what the stream uses: interlaced coding
  (field pictures and MBAFF), High 10, 4:2:2 and 4:4:4 profiles, and slice
  groups.
- Sizes from 64x64 to 4096x4096 on AMD VCN; the driver states its limits and a
  larger picture is refused by its size. A smaller one decodes into a surface
  of the minimum size, and only its visible part is converted.

## Errors

Every error from a decoder is a `DecoderError`, with libva's, the DMA-BUF
shim's or the parser's own error as its cause. `DecoderUnavailable` means no
backend here decodes what was asked for, and its message carries what each
unavailable backend said about why. `EncoderError` and `DecoderError` share
the base `VideoError`, so one `except VideoError` covers a program that
records and plays.

## Performance

Measured on a Radeon 8060S (Mesa 25.2.8 `radeonsi`) playing a 1920x1080 60 fps
H.264 High stream with three B-frames, frame by frame on the GL thread: about
0.75 ms a frame for decoding and conversion together, which is 1300 frames a
second. Most of that is waiting for the hardware decode to finish before the
conversion reads the surface. Decoding runs on the GL thread for now; moving
it to a worker thread is in the [plan](../plans/GPU-VIDEO-DECODE.md).

Each decoder surface is exported and imported into OpenGL once, when the
decoder opens, so decoding creates no GL objects.

## Threading

`Movie`, `MP4Reader` and `AnnexBReader` need no GL and can be used from any
thread. A decoder, a converter and a player belong to the thread whose OpenGL
context was current when they were made.
