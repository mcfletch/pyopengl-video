# Playing video into OpenGL textures

`pyopengl-video` decodes H.264, HEVC, AV1 and VP9 on the GPU's video decoder
and hands each picture to OpenGL as textures. Where the GPU that decodes is
the one that draws, the frame never passes through host memory. A conversion
pass turns the decoded YUV into an RGBA texture, the right way up for OpenGL.
Any PyOpenGL program can play a movie; no scenegraph is needed.

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
MP4, IVF and raw-stream readers are part of this package.

| Backend | Hardware | Platform | Codecs | Picture handed over as | State |
| --- | --- | --- | --- | --- | --- |
| `nvdec` | NVIDIA | Linux | H.264 | CUDA memory copied on the GPU into two textures | working |
| `nvdec` | NVIDIA | Windows | H.264 | the same | the same code, untested |
| `vaapi` | AMD, VCN | Linux | H.264, HEVC, AV1, VP9 | two DMA-BUF planes imported as textures | working |
| `vaapi` | Intel, Gen9 and later | Linux | H.264, HEVC, AV1, VP9 | the same | the same code, untested |
| `nvdec-copy` | NVIDIA | Linux, Windows | H.264 | CUDA memory copied through host memory into two textures | working on Linux |
| `vpl` | Intel, Gen9 and later | Windows | H.264, HEVC, VP9 | a Direct3D 11 texture converted to RGB on the GPU | working |

The codecs are the ones the backend drives. A GPU decodes those its hardware
has, and `decoders()` lists them for this machine.

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

**The `vpl` backend needs the OpenGL context to render on the Intel GPU.** The
decoder writes Direct3D 11 surfaces, and a texture shared with OpenGL through
`WGL_NV_DX_interop2` is what carries the picture across, which works only for a
context on the decoder's own GPU. On a laptop whose NVIDIA GPU draws the
window the backend reports itself unavailable and says so, and `nvdec-copy`
serves. oneVPL's video processor converts each decoded picture to RGB using the
matrix and range the stream states (BT.601 or BT.709; a stream in BT.2020 is
refused by name), so the frame is an `'rgba'` frame and the conversion pass
copies it. Its pixels differ from the shader path's by the video processor's
chroma interpolation. It uses `libvpl.dll` and the Intel graphics driver; a
10-bit stream converts to 8 bits. AV1 is listed for parts whose Direct3D 11
decoder profiles include it and has not been run, since the Gen9 part it was
written on has no AV1 decoder.

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
    print(backend.name, backend.vendor,
          {codec: backend.largest(codec) for codec in sorted(backend.codecs)})
```

On a Radeon 8060S that is H.264 to 4096x4096, and HEVC, AV1 and VP9 to
8192x4352.

## Movies and players

`Movie.open(path)` reads what the first bytes of the file say it is:

| File | Codecs | Reader |
| --- | --- | --- |
| MP4 (`.mp4`, `.mov`, `.m4v`) | H.264 (`avc1`, `avc3`), HEVC (`hvc1`, `hev1`), AV1 (`av01`), VP9 (`vp09`) | `MP4Reader` |
| IVF (`.ivf`) | AV1, VP9 | `IVFReader` |
| A raw Annex-B stream (`.h264`, `.h265`) | H.264, HEVC | `AnnexBReader` |

An AVI file, an MPEG program or video stream, a Matroska or WebM file and an
Ogg file are refused with a `ValueError` naming the container, and an MP4
whose video is in another codec with one naming the codec. Opening an MP4
reads the index and none of the media, so a two-hour movie costs the size of
its sample table. It needs no GL, so a game can open movies on a loading
thread.

| `Movie` | |
| --- | --- |
| `codec` | `'h264'`, `'hevc'`, `'av1'` or `'vp9'` |
| `size` | the visible picture size, `(width, height)` |
| `frames` | how many pictures the video track holds |
| `duration` | the length in seconds, to the end of the last frame |
| `headers` | the stream's configuration as its decoder takes it: parameter sets as Annex-B for H.264 and HEVC, the configuration OBUs for AV1, nothing for VP9 |
| `player(backend=None, ahead=2, mipmaps=False, **options)` | a player in the current GL context |

A raw H.264 or HEVC stream has no timestamps. Its pictures are timed in
display order, at the frame rate its parameter sets state. A stream that states none
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

A key frame is an H.264 IDR picture or a picture coded entirely of I slices,
an HEVC intra random access picture (IDR, CRA or BLA), or an AV1 or VP9 key
frame that is shown. In an H.264 or HEVC stream with open groups of pictures,
some pictures follow a key frame in decoding order and precede it in display
order. They refer to the group before it, so after a seek the decoder drops
them.

### Colour, orientation and size

The output texture is `GL_SRGB8_ALPHA8`, holding non-linear RGB. A shader
sampling it reads linear colour, as it does from an image texture. To draw it
straight to the screen, enable `GL_FRAMEBUFFER_SRGB` on an sRGB-capable
framebuffer (`glfw.SRGB_CAPABLE`). BT.709's transfer curve is treated as
sRGB's.

The matrix and range come from the stream's own colour description: the
video usability information of H.264 and HEVC, the colour configuration of
AV1 and the colour space of VP9. BT.601, BT.709 and BT.2020 are converted, at
limited or full range, from 8-bit and 10-bit pictures alike. A 10-bit stream
still gives a `GL_SRGB8_ALPHA8` texture, and its transfer curve is not
converted: PQ and HLG pictures come out with their code values as they are. A stream
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
    decoder = open_decoder(track.codec, headers=track.parameter_sets())
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
  `track.codec` is the sample entry's four-character code, which
  `open_decoder` accepts as it stands.
  `track.sample(index, parameter_sets=False)` returns a `Sample` whose `data`
  is what the codec's decoder takes: an Annex-B access unit for H.264 and
  HEVC, a temporal unit of OBUs for AV1, a frame or superframe for VP9. The
  track's configuration is put in front when asked. Its `timestamp` is the
  presentation time after the edit list, in `track.timescale` units.
  `display_order()`, `sync_samples` and `keyframe_before(index)` plan a seek.
  Fragmented MP4 is not read.
- `AnnexBReader` reads a raw H.264 or HEVC stream the same way, and its
  `codec` says which it found. It scans the stream once for access units and
  their display order.
- `IVFReader` reads an IVF file of AV1 or VP9 the same way.
- `decoder.decode(sample)` takes one sample and returns a list of the
  frames now ready, in display order. The decoder holds pictures back until
  their order is known, so the list may be empty or hold several. `flush()`
  returns the rest at the end of a stream. `reset()` drops what is held,
  before decoding from a key frame.
- A `DecodedFrame` is lent by the decoder. Its `planes` are the decoder's
  textures: a luma texture and a chroma texture, Cb in red and Cr in green.
  An 8-bit stream gives the `'nv12'` layout, `GL_R8` and `GL_RG8`. A 10-bit
  stream gives `'p010'`, `GL_R16` and `GL_RG16`, each sample in the high ten
  bits of its channel. The `vpl` backend gives `'rgba'`: one texture, already
  converted, held for OpenGL until the frame is released. The picture sits at `visible`, first row at the top, inside
  textures of `texture_size`. Release every frame: a decoder with all its
  textures lent raises `DecoderError` naming the frames held. A caller may
  hold `output_slots` frames at once, 4 by default. On VA-API the textures
  are the decode surfaces themselves; on NVDEC and oneVPL each frame is a copy
  of the picture in textures of its own.
- `NV12Converter.convert(frame, target)` draws a frame of any layout into
  an RGBA texture of the frame's size. For `'nv12'` and `'p010'` frames one
  pass applies colour, crop and the vertical flip; for an `'rgba'` frame it
  applies the crop and the flip. It restores every piece of GL state it
  changes, so it can run inside a renderer's frame.

`MP4Reader.audio` reports each audio track's codec, sample rate, channels and
configuration, and reads its samples. No audio is decoded yet; see *Audio* in
the [plan](../plans/GPU-VIDEO-DECODE.md).

## What is decoded

Every codec is 4:2:0. A stream outside what is listed is refused with an
error naming the feature.

| Codec | Decoded | Refused by name |
| --- | --- | --- |
| H.264 | Constrained Baseline, Main and High at 8 bits. Several slices a picture, CAVLC and CABAC, B-frames and pyramids, weighted prediction, long-term references, scaling matrices. Interlaced coding on the NVDEC backends. | High 10, 4:2:2 and 4:4:4 profiles; interlaced coding and slice groups on `vaapi` |
| HEVC | Main at 8 bits and Main 10. Several slices with wavefront entry points, tiles, weighted prediction, coded and default scaling lists, long-term references, temporal sub-layers, open groups (CRA with RASL pictures). | 4:2:2, 4:4:4 and monochrome, 12 bits, luma and chroma of different depths, the range, multilayer, 3D and screen-content extensions |
| AV1 | Main profile at 8 and 10 bits, and monochrome. Hidden and alt-ref frames, `show_existing_frame`, intra-only and switch frames, tiles, segmentation, quantiser matrices, loop restoration, global motion, super-resolution, film grain. | High and Professional profiles, a frame size that changes between key frames, layers above the base, tile list OBUs |
| VP9 | Profile 0, and profile 2 at 10 bits. Superframes, hidden alt-ref frames, `show_existing_frame`, tile columns and rows, segmentation, lossless. | profiles 1 and 3, 12 bits, a frame whose size or depth differs from its key frame's |

HEVC, AV1 and VP9 decode through `vaapi`. H.264 decodes through every
backend, and they differ in this:

| H.264 | `vaapi` | `nvdec`, `nvdec-copy` |
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

- Each fixture stream under `tests/data/decode/` decodes to the bytes
  `ffmpeg`'s software decoder gives, picture for picture, on AMD VCN, and
  each H.264 one on an RTX 3060 Ti. The
  features x265 does not emit (tiles, long-term references, dependent slice
  segments) are parsed and tested without a driver, and no hardware decode
  has exercised them.
- HEVC, AV1 and VP9 sizes are the driver's: 64x64 to 8192x4352 on AMD VCN. A
  larger picture is refused by its size. An HEVC picture smaller than the
  minimum is refused, since the AMD driver takes the picture size from the
  surface.
- An AV1 picture is its coded frame. A stream whose render size differs from
  its frame size is shown at the frame size.

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

`Movie`, `MP4Reader`, `IVFReader` and `AnnexBReader` need no GL and work from
any thread. A
decoder, a converter and a player belong to the thread whose OpenGL context
was current when they were made.
