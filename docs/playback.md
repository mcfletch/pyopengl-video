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
| `nvdec` | NVIDIA | Linux | H.264, HEVC, VP9, AV1 | CUDA memory copied on the GPU into two textures | working |
| `nvdec` | NVIDIA | Windows | H.264, HEVC, VP9, AV1 | the same | H.264 working, where the OpenGL context is on the NVIDIA GPU; the others the same code, untested |
| `vaapi` | AMD, VCN | Linux | H.264, HEVC, AV1, VP9 | two DMA-BUF planes imported as textures | working |
| `vaapi` | Intel, Gen9 and later | Linux | H.264, HEVC, AV1, VP9 | the same | the same code, untested |
| `nvdec-copy` | NVIDIA | Linux, Windows | H.264, HEVC, VP9, AV1 | CUDA memory copied through host memory into two textures | H.264 working; the others working on Linux |
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
8192x4352. On an RTX 3060 Ti it is H.264 to 4096x4096, and HEVC, VP9 and AV1
to 8192x8192, through `nvdec` and `nvdec-copy` alike. An NVIDIA GPU is asked
which codecs it decodes the first time `decoders()` is called, which makes
the CUDA context a decoder would otherwise make when it opens.

## Movies and players

`Movie.open(path)` reads what the first bytes of the file say it is:

| File | Codecs | Reader |
| --- | --- | --- |
| MP4 (`.mp4`, `.mov`, `.m4v`) | H.264 (`avc1`, `avc3`), HEVC (`hvc1`, `hev1`), AV1 (`av01`), VP9 (`vp09`) | `MP4Reader` |
| IVF (`.ivf`) | AV1, VP9 | `IVFReader` |
| A raw Annex-B stream (`.h264`, `.h265`) | H.264, HEVC | `AnnexBReader` |

An AVI file, an MPEG program or video stream, a Matroska or WebM file and an
Ogg file are refused with a `ContainerError` naming the container, and an MP4
whose video is in another codec with an `MP4Error` naming the codec. Opening
an MP4 reads the index and none of the media, so a two-hour movie costs the
size of its sample table. It needs no GL, so a game can open movies on a
loading thread.

Every length, count and offset in a file is whatever its writer chose. Each
is compared with the box, the record or the file it describes before it is
used, and one that does not fit raises the reader's error: `MP4Error`, `IVFError`, or `StreamError` for a raw stream. Opening
takes memory in proportion to the file's size, whatever counts it states. An
MP4 track's index is about 30 bytes a sample and is made when the track's
samples are first asked for, so an audio track beside the video costs nothing
to a player. A file that cannot be opened is closed before the error is
raised.

| `Movie` | |
| --- | --- |
| `codec` | `'h264'`, `'hevc'`, `'av1'` or `'vp9'` |
| `size` | the picture size the container states, `(width, height)`; see [Colour, orientation and size](#colour-orientation-and-size) |
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
| `size` | the size of the frame shown and of `texture`, `(width, height)`, or None |
| `ready_ahead` | converted frames waiting beyond the one shown |
| `close()` | closes the converter and the decoder and deletes the textures, each whatever the others do; call it with the player's OpenGL context current. `frame_at` then raises `DecoderError`. |

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

Two sizes are in play, and they need not agree:

- `movie.size` is what the container states. An MP4 sample entry and an IVF
  header each hold a width and a height from 0 to 65535 that nothing ties to
  the stream. A raw stream's size is read from its parameter sets.
- `player.size` is the size of the frame shown, which is the size of the
  picture the decoder returned and of the texture it was converted into.

The player makes each output texture at the size of the picture converted
into it, so a texture holds that picture and nothing else, whatever the
container states. Size a quad or a window from `player.size`. `movie.size` is
there before any GL exists, for a layout made while the movie loads, and it
chooses the decoder backend: one whose largest picture is smaller than the
stated size is passed over. A stated width or height of 0 leaves every
backend in, and so does a stated size that no backend takes, since the stream
may be smaller than its container says; a decoder refuses a picture larger
than it decodes when the stream's first picture arrives. Where a stream changes size at a key frame, the textures of the
old size are deleted as they fall out of use and `player.size` changes with
the frame. A picture the decoder returns with a width or a height of 0 or
less raises `DecoderError` before any texture is made.

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
  A sample the file does not hold whole raises `MP4Error`, as does one whose
  NAL unit lengths do not add up to it. `track.size` is the size the sample
  entry states.
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
  their display order. A path is read a megabyte at a time and each sample is
  read from the file when it is asked for, so the index is what a long stream
  costs in memory; a file object is read whole. A file shortened by another
  process after it was opened raises `StreamError` from `sample()`.
- `IVFReader` reads an IVF file of AV1 or VP9 the same way. A frame that
  states more bytes than the file holds ends the stream before it, as the end
  of the file does, so a recording cut short plays up to its last whole frame.
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
  hold `output_slots` frames at once, 4 by default and 1 to 64. On
  VA-API the textures are the decode surfaces themselves; on NVDEC and oneVPL
  each frame is a copy of the picture in textures of its own.
- `NV12Converter.convert(frame, target)` draws a frame of any layout into
  an RGBA texture of the frame's size. For `'nv12'` and `'p010'` frames one
  pass applies colour, crop and the vertical flip; for an `'rgba'` frame it
  applies the crop and the flip. It restores every piece of GL state it
  changes, so it can run inside a renderer's frame. A frame whose `visible`
  rectangle is empty or reaches outside its `texture_size` is refused with
  `DecoderError`.
  `NV12Converter.draw()` takes the two plane textures themselves, with
  `visible`, an optional `texture_size` to check it against, and an optional
  `size`: how much of the target is drawn, from its lower left, one output
  texel for each luma texel. It is the visible size by default and may be
  smaller, never larger.
- A VA-API decoder raises only `DecoderError`: whatever a stream or the
  driver brings about is translated, with the original on `__cause__`. A
  sequence that asks for more decode surfaces than any codec needs (18) is
  refused. When a stream changes to a size the device refuses, the decoder
  holds no surfaces and refuses each picture until a sequence begins that the
  device takes.
- An NVDEC decoder copies each picture into textures of the OpenGL context
  it was opened in. `decode()` and `flush()` raise `DecoderError` with
  another context current, or none, and decode again once its own is.
- An NVDEC decoder reads each picture NVIDIA's parser builds before the
  hardware is handed it. A picture that states more bitstream than the
  parser was fed, names a surface the decoder was not made with, or predicts
  from a picture the hardware has not decoded raises `DecoderError` and is
  not decoded. A stream that changes to a sequence the GPU refuses leaves
  the decoder with no hardware decoder until a sequence begins that it takes.
- An AV1 sample is read by this package's own parser before NVIDIA's on the
  NVDEC backends. A sample that breaks the AV1 specification raises
  `DecoderError` and is not handed over. Decoding then waits for the next
  key frame that is shown, and the samples before it return no frames; so do
  the samples before the first key frame of a stream, and those after a
  `reset()`. An OBU with no size field, a tile list OBU and a frame that
  ends in another sample than it began in are refused. NVIDIA's parser is
  handed the temporal delimiter, the sequence headers, the frame headers and
  the tile groups of a sample; its metadata and padding OBUs are left out,
  and so are those of an MP4's `av1C` configuration.
- Close a decoder and an `NV12Converter` with the OpenGL context they were
  made in current. A VA-API decoder releases its libva objects, the display
  and the render node whichever context is current, and an NVDEC decoder its
  parser and its hardware decoders; a decoder's textures, and the
  converter's program and framebuffers, are objects of that one context. With
  another current, or none, they are kept, `DecoderError` says so, and
  closing again with the right context current gives them back.

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
| VP9 | Profile 0, and profile 2 at 10 bits. Superframes, hidden alt-ref frames, `show_existing_frame`, tile columns and rows, segmentation, lossless. | profiles 1 and 3, 12 bits, a frame whose size, depth or profile differs from its key frame's |

Every backend decodes all four codecs, where the GPU has the decoder. On the
NVDEC backends NVIDIA's parser reads the stream, so the features in the table
are the hardware's, and what is refused is refused when the stream's first
picture arrives: anything but 4:2:0 or monochrome, anything but 8 or 10 bits,
and a codec, depth or size the GPU has no decoder for. The backends differ in
this:

| | `vaapi` | `nvdec`, `nvdec-copy` |
| --- | --- | --- |
| Interlaced H.264 (field pictures, MBAFF) | refused by name | decoded, one frame a sample, deinterlaced as `deinterlace` says |
| H.264 slice groups | refused by name | passed to the hardware, not tested |
| H.264 sizes | 64x64 to 4096x4096 on AMD VCN, as the driver reports them | 48x16 to 4096x4096 on an RTX 3060 Ti, as the driver reports them |
| HEVC, VP9 and AV1 sizes | 64x64 to 8192x4352 on AMD VCN | 144x144 (HEVC) or 128x128 (VP9, AV1) to 8192x8192 on an RTX 3060 Ti |
| A picture under the smallest size | decodes into a minimum-size surface, and only its visible part is converted; refused by its size in HEVC | refused by its size |
| A picture over the largest size | refused by its size | refused by its size |
| A VP9 or AV1 picture of an odd size | not tested | decoded into textures of the even size above it; `visible` is the picture |
| When pictures come back | when their order is known | two samples later, so the hardware has finished them; VP9 with its sample |
| A VP9 or AV1 frame whose size differs from its key frame's | refused by name | not tested |
| When a stream is refused | at the first picture that uses the sequence | at the first picture that uses the sequence |

`deinterlace='weave'` gives the two fields interleaved as they were coded,
which is exact and shows combing where the picture moves. `'bob'` scales one
field to the frame's height. `'adaptive'` chooses between the two for each part
of the picture by its motion. A progressive stream is not deinterlaced,
whichever is named.

- Each fixture stream under `tests/data/decode/` decodes to the bytes
  `ffmpeg`'s software decoder gives, picture for picture, on AMD VCN. On an
  RTX 3060 Ti each H.264 one does, and so does each stream of the other
  codecs that is at least 144 samples each way: B-frame pyramids and open
  groups in HEVC, hidden frames and frames shown again in VP9 and AV1, film
  grain, super-resolution and monochrome in AV1, 10 bits in all three, and a
  picture of an odd size in VP9 and AV1. The
  features x265 does not emit (tiles, long-term references, dependent slice
  segments) are parsed and tested without a driver, and no hardware decode
  has exercised them.
- An HEVC picture smaller than the minimum is refused on `vaapi`, since the
  AMD driver takes the picture size from the surface.
- NVIDIA's parser shows no picture for a VP9 frame that shows a reference
  slot again, so the NVDEC backends follow the slots themselves and show it.
  A stream decoded from a sample that is not a key frame has slots nothing
  was read into; a frame shown again from one of those is left out, with a
  warning in the log.
- A sample that is not AV1 at all raises `DecoderError` on the NVDEC
  backends, as does one that breaks the AV1 specification: NVIDIA's AV1
  parser is handed only what this package's own has read. One that is not
  H.264, HEVC or VP9 produces no picture.
- When a stream changes to a sequence of another size with no flush, the
  pictures NVIDIA's parser still held of the sequence before are shown after
  their decoder has gone. The NVDEC backends leave those out, with a warning
  in the log.
- An AV1 picture is its coded frame. A stream whose render size differs from
  its frame size is shown at the frame size.

## Errors

A decoder raises `DecoderError`. On VA-API it carries libva's, the DMA-BUF
shim's or the parser's error as its cause. On NVDEC a failed driver call is a
`CUDAError`, a `DecoderError` with the driver's `status`, its `name` (such as
`CUDA_ERROR_INVALID_VALUE`) and the `call` that returned it. A picture or an
AV1 sample the NVDEC backends refuse before the driver sees it is a plain
`DecoderError`, with the parser's `StreamError` as its cause for a sample.
`DecoderUnavailable` means no backend here
decodes the stream. Its message includes each unavailable backend's reason.
`EncoderError` and `DecoderError` share the base `VideoError`, so one
`except VideoError` covers recording and playback.

A file that is not what a reader reads raises `MP4Error` or `IVFError`. Both
are `ContainerError`s, and a `ContainerError` is a `VideoError` and a
`ValueError`. `Movie.open` raises `ContainerError` itself for a container it
names and does not read. A sample index outside a track is an `IndexError`.

A coded stream's syntax is whatever its writer chose, as a file's lengths
are. The H.264 and HEVC parsers hold every element to the range its
specification gives it, and one outside it raises `StreamError` naming the
element, its value and the range: `max_num_ref_frames is 100000, outside 0 to
16`. From a VA-API decoder that is the cause of a `DecoderError`, and the
picture is not submitted. `StreamError` is a `VideoError` and a `ValueError`.

The AV1 and VP9 parsers do the same for what their headers state: a frame
larger than its sequence header allows, more than 64 tile columns or rows, a
tile that runs past its frame, film grain points out of their order, a
reference slot that holds no frame. An AV1 sequence header that differs from
the one before starts a new sequence, which begins at a key frame that is
shown; any other frame after it is a `StreamError` while a frame of the
sequence before is held as a reference. A VP9 frame that states
another profile than its key frame's is refused, and so is one of 65536
samples a side: VA-API is told each dimension in 16 bits.

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

The other codecs at 1920x1080 and 60 fps, decoded and converted on the same
card, in milliseconds a frame:

| Stream | `nvdec` | `nvdec-copy` |
| --- | --- | --- |
| HEVC Main, 6 Mbit/s | 0.6 | 2.4 |
| HEVC Main 10, 6 Mbit/s | 0.6 | 4.5 |
| VP9 profile 0, 6 Mbit/s | 1.0 | 2.7 |
| AV1 Main, 6 Mbit/s | 0.6 | 2.6 |

A 10-bit picture is two bytes a sample, so `nvdec-copy` moves twice as much
through host memory for it.

The first use of NVDEC in a process initialises CUDA and makes its context,
which is most of the 250 ms the first `decoders()` call takes on a machine
with an NVIDIA GPU. The context then stays for the life of the process, the
first decoder opens in about 25 ms and each later one in about 1 ms. A stream's first picture takes about 30 ms more,
which is the driver creating its hardware decoder. Closing a decoder returns
in about 1 ms: the driver's own teardown takes up to 240 ms, and runs on a
thread of its own so the thread that draws does not wait for it. Hardware
decoders are made and destroyed one at a time, so a stream whose first
picture arrives while another decoder is being torn down waits for that to
finish: about 260 ms for a decoder opened as the only other one in the
process closes, in any codec. The driver makes an H.264 decoder wait that long
by itself, and an HEVC one 170 ms. To change movies without the wait, decode
the new movie's first frame before closing the old player: a decoder torn
down while another is open takes 36 ms, and the new one is already made.

Reading an AV1 sample before NVIDIA's parser does takes about 0.06 ms a
sample for a 150x146 stream, on the thread that decodes.

## Threading

`Movie`, `MP4Reader`, `IVFReader` and `AnnexBReader` need no GL and work from
any thread. Several threads may read samples through one reader at once: no
read moves a file position another is using. A
decoder, a converter and a player belong to the thread whose OpenGL context
was current when they were made.
