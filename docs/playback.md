# Playing video into OpenGL textures

`pyopengl-video` decodes H.264 on the GPU's video decoder and hands each
picture to OpenGL as textures. Where the GPU that decodes is the one that
draws, the frame never passes through host memory. A conversion pass turns
the decoded YUV into an RGBA texture, the right way up for OpenGL. Any
PyOpenGL program can play a movie; no scenegraph is needed.

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
| `nvdec` | NVIDIA | Linux | H.264 | CUDA memory copied on the GPU into two textures | working |
| `nvdec` | NVIDIA | Windows | H.264 | the same | the same code, untested |
| `vaapi` | AMD, VCN | Linux | H.264 | two DMA-BUF planes imported as textures | working |
| `vaapi` | Intel, Gen9 and later | Linux | H.264 | the same | the same code, untested |
| `nvdec-copy` | NVIDIA | Linux, Windows | H.264 | CUDA memory copied through host memory into two textures | working on Linux |
| `vpl` | Intel | Windows | H.264 | a Direct3D 11 texture | [planned](../plans/GPU-VIDEO-DECODE.md) |

The table is in the order `movie.player()` prefers the backends in.

**The `vaapi` backend needs an EGL context.** Each decode surface reaches
OpenGL through `EGL_EXT_image_dma_buf_import`. Ask GLFW for one before the
window is created:

```python
glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
```

From a GLX or WGL context the backend reports itself unavailable. Where no
other backend serves the context either, `movie.player()` raises
`DecoderUnavailable` saying the context is not an EGL context. It also needs the VA-API driver for the GPU: `mesa-va-drivers` for
AMD, `intel-media-va-driver` for Intel.

**The `nvdec` backend needs the OpenGL context to render on the NVIDIA GPU.**
NVDEC decodes into CUDA memory, and CUDA copies each picture into a pair of
textures it has registered, which it can do only for a context on its own GPU.
An EGL context and a GLX context both serve; a WGL context goes through the
same call and has not been run. It uses two libraries from the
NVIDIA driver, `libcuda.so.1` and `libnvcuvid.so.1` (`nvcuda.dll` and
`nvcuvid.dll` on Windows); the CUDA toolkit is not needed. Debian and Ubuntu
package the second apart from the driver, as `libnvidia-decode-<version>`.

Where another GPU draws the window, as on a laptop whose integrated GPU does,
`nvdec` reports itself unavailable and says so, and `nvdec-copy` is the
backend that works: it decodes on the NVIDIA GPU and copies each picture
through host memory, about 3 MB a frame at 1080p. `decoders()` lists it last,
so a backend that keeps the picture on the GPU is chosen first where there is
one, and `backend.zero_copy` and `decoder.zero_copy` say which was chosen.

`decoders()` lists what this machine can decode:

```python
from pyopengl_video import decoders

for backend in decoders():
    print(backend.name, backend.vendor, sorted(backend.codecs), backend.max_size)
```

## Movies and players

`Movie.open(path)` reads an MP4 file or a raw H.264 (Annex-B) stream. The first
bytes of the file say which. An AVI file, an MPEG program or video stream, a
Matroska or WebM file and an Ogg file are refused with a `ValueError` naming
the container. Opening reads the index and none of the media, so
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

The `options` of `player()` go to the decoder. Every backend takes
`output_slots`, the frames a caller may hold unreleased at once, 4 by default.
The NVDEC backends also take `deinterlace` for an interlaced stream:
`'adaptive'` (the default), `'bob'` or `'weave'`.

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
  textures: a `GL_R8` luma texture and a `GL_RG8` chroma texture, Cb in red
  and Cr in green. The picture sits at `visible`, first row at the top, inside
  textures of `texture_size`. Release every frame: a decoder with all its
  textures lent raises `DecoderError` naming the frames held. A caller may
  hold `output_slots` frames at once, 4 by default. On VA-API the textures
  are the decode surfaces themselves; on NVDEC each frame is a copy of the
  picture in textures of its own.
- `NV12Converter.convert(frame, target)` draws a frame into an RGBA texture of
  the frame's size. One pass applies colour, crop and the vertical flip. It
  restores every piece of GL state it changes, so it can run inside a
  renderer's frame.

`MP4Reader.audio` reports each audio track's codec, sample rate, channels and
configuration, and reads its samples. No audio is decoded yet; see *Audio* in
the [plan](../plans/GPU-VIDEO-DECODE.md).

## What is decoded

- H.264 Constrained Baseline, Main and High profiles, 8-bit 4:2:0. Several
  slices a picture, CAVLC and CABAC, B-frames and B-frame pyramids, weighted
  prediction, long-term references and scaling matrices are all decoded.
- Refused, with an error naming the feature: the High 10, 4:2:2 and 4:4:4
  profiles on every backend.

| | `vaapi` | `nvdec`, `nvdec-copy` |
| --- | --- | --- |
| Interlaced streams (field pictures, MBAFF) | refused by name | decoded, one frame a sample, deinterlaced as `deinterlace` says |
| Slice groups | refused by name | passed to the hardware, not tested |
| Sizes | 64x64 to 4096x4096 on AMD VCN, as the driver reports them | 48x16 to 4096x4096 on an RTX 3060 Ti, as the driver reports them |
| A picture under the smallest size | decodes into a minimum-size surface, and only its visible part is converted | refused by its size |
| A picture over the largest size | refused by its size | refused by its size |
| When a stream is refused | at the first picture that uses the sequence | at the first picture that uses the sequence |

`deinterlace='weave'` gives the two fields interleaved as they were coded,
which is exact and shows combing where the picture moves. `'bob'` scales one
field to the frame's height. `'adaptive'` chooses between the two for each part
of the picture by its motion. A progressive stream is not deinterlaced,
whichever is named.

## Errors

A decoder raises `DecoderError`. On VA-API it carries libva's, the DMA-BUF
shim's or the parser's error as its cause. On NVDEC a failed driver call is a
`CUDAError`, a `DecoderError` with the driver's `status`, its `name` (such as
`CUDA_ERROR_INVALID_VALUE`) and the `call` that returned it.
`DecoderUnavailable` means no backend here
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

The same stream takes 1.4 ms a frame on NVDEC, or 700 frames a second, and
2.8 ms through `nvdec-copy`, measured on an RTX 3060 Ti with driver 580.
Nearly all of the 1.4 ms is the hardware decoding the picture; the copy into
the textures is 0.1 ms of it. An NVDEC decoder makes a pair of output textures
the first time none is free. The stream above played forward makes four
pairs, and decoding creates none after that.

The first NVDEC decoder in a process initialises CUDA, about 170 ms in all:
65 ms the first time `decoders()` is asked and 100 ms when the decoder opens.
The CUDA context then stays for the life of the process, so each later
decoder opens in about 1 ms. A stream's first picture takes about 30 ms more,
which is the driver creating its hardware decoder. Closing a decoder returns
in about 1 ms: the driver's own teardown takes up to 240 ms, and runs on a
thread of its own so the thread that draws does not wait for it.

## Threading

`Movie`, `MP4Reader` and `AnnexBReader` need no GL and work from any thread. A
decoder, a converter and a player belong to the thread whose OpenGL context
was current when they were made.
