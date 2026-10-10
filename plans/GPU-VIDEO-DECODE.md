# Hardware video decode into an OpenGL texture

**Status:** D1, D2 and D3 landed 2026-10-09: H.264 plays into OpenGL textures
on AMD under Linux, through `pyopengl_video.playback`, with no scenegraph.
Every picture of the x264 test streams decodes bit-exact against `ffmpeg`'s
software decoder. D4 (OpenGLContext) is next. Audio is deferred (see *Audio*).
**Scope:** `pyopengl-video`, with OpenGLContext's `MovieTexture` as the first
consumer.
**Goal:** a movie file in, an OpenGL texture out, on NVIDIA, AMD and Intel, on
Linux and Windows, with the frame staying on the GPU. Each platform decodes
through the vendor library its encoder already uses.

## Why this exists

OpenGLContext's `MovieTexture` decodes through PyAV today
(`OpenGLContext/loaders/movie.py`, the `movie` extra). PyAV links FFmpeg, and
the FFmpeg a Linux distribution ships is built with GPL components, so a game
that installs the extra ships a GPL library. The engine cannot depend on it,
optionally or otherwise. That leaves `MovieTexture` without a decoder anyone
can ship.

The decode hardware is already reachable. Every GPU this package encodes on
has a fixed-function decoder beside the encoder, driven through the same
vendor library: NVENC's sibling NVDEC, VA-API's `VAEntrypointVLD`, and
oneVPL's `MFXVideoDECODE`. The ABI harnesses, the DMA-BUF code, the Direct3D
interop shim and the MP4 writer all have a reverse direction that reuses most
of what they are.

A decoder also closes the gap the encode plan names under *Testing*: a round
trip, encoded and decoded in process, compared against the source frames. That
check currently runs through an `ffmpeg` binary where one is installed.

What it is for, beyond `MovieTexture`: cut scenes, video on in-world screens,
a replay viewer that scrubs, and any PyOpenGL application that wants to show a
video without a scenegraph.

## What is excluded, and why

- FFmpeg, PyAV and GStreamer as libraries. Distribution builds of FFmpeg are
  GPL; GStreamer's H.264 parsers and its VA-API elements are LGPL. Neither is
  linked, imported or read.
- Reading any LGPL or GPL decoder's source for the control layer described
  below. FFmpeg's `h264*.c` and `vaapi_h264.c`, and GStreamer's `codecparsers`,
  implement exactly that layer. The ITU-T recommendation is the source, and
  [CLEAN-ROOM.md](../../CLEAN-ROOM.md) applies to anyone who has read either.
- Running an `ffmpeg` binary stays available to the test suite as an
  independent cross-check, as it is now. The package never calls one.

## Verified ground truth

Measured 2026-10-09 on the AMD machine: Radeon 8060S (`gfx1151`), Mesa 25.2.8
`radeonsi`, libva 2.12 reporting VA-API 1.20, GLFW EGL context, GL 4.6 core.
The probe opened an H.264 High `VAEntrypointVLD` config and context with the
binding in `vaapi/api.py` unchanged.

| Fact | Result |
| --- | --- |
| `VAEntrypointVLD` profiles | H.264 Constrained Baseline, Main and High; HEVC Main and Main10; VP9 profiles 0 and 2; AV1 profile 0; JPEG baseline |
| `vaCreateConfig` / `vaCreateContext` for H.264 High VLD, 1920x1080, 4 NV12 surfaces | both succeed |
| `vaExportSurfaceHandle`, `DRM_PRIME_2`, read-only, separate layers | 1 object, 2 layers: `R8` at offset 0 and `GR88` at offset 2621440, both pitch 2048 |
| The same, composed layers | 1 object, 1 `NV12` layer with 2 planes, the same offsets and pitches |
| Modifier on the exported object | `0x0200000018601b04`, a tiled AMD layout |
| Luma plane height in memory | 1280 rows for a 1080-row surface (2621440 / 2048) |
| `dmabuf.import_texture` of each separate layer | succeeds: `R8` as 1920x1080, `GR88` as 960x540 |

Three consequences. A decode surface reaches OpenGL as two ordinary textures,
one per plane, with no new EGL code. The modifier is tiled, so it has to be
carried into the import, which `import_texture` already does. The surface is
padded well past its visible size, so the visible rectangle comes from the
stream's cropping, never from the allocation.

Measured since, by decoding (2026-10-09, the same machine):

| Fact | Result |
| --- | --- |
| Decode surface size limits (`vaQuerySurfaceAttributes`) | 64x64 to 4096x4096; a smaller picture decodes into a surface of the minimum size |
| Plane order of the separate layers | luma `R8`, then chroma `GR88` with Cb in the low byte |
| `VAIQMatrixBufferH264` list order | raster, as the header states; zig-zag order decodes wrongly |
| `slice_data_bit_offset` | counted from the NAL header with emulation prevention removed, as the header states |
| x264 `cqm=jvt` | a PPS with every list absent under an SPS with none, which is fall-back rule A (7.4.2.2) |
| 1080p60 High with three B-frames, decode and convert on the GL thread | 0.75 ms a frame, about 1300 frames a second; the largest part is `vaSyncSurface` waiting for the decode |

The first measurement of the last row was 2.05 ms. Two causes were removed:
NAL unit splitting scanned every byte in Python, and a surface was reused as
soon as it came back, so its fence was still unsignalled; the least recently
returned surface is now reused first.

The NVIDIA and Windows rows below are from documentation and headers, and stay
marked unverified until they have run.

## Per-platform mechanisms

| Vendor | Linux | Windows |
| --- | --- | --- |
| NVIDIA | NVDEC (`libnvcuvid.so.1`) into CUDA memory, copied on the GPU into GL textures registered with CUDA | the same code, `nvcuvid.dll` and `nvcuda.dll` |
| AMD | VA-API VLD on Mesa, surfaces exported as DMA-BUF and imported into GL | AMF decoder into a D3D11 texture, through the interop shim |
| Intel | the same VA-API code against `iHD` | oneVPL decode and VPP into a D3D11 texture, through the interop shim |

### NVIDIA: NVDEC and CUDA

NVDEC has no OpenGL device type, which is the one place decode is not the mirror
of encode. A decoded picture lands in CUDA device memory, and CUDA's own GL
interop moves it into a texture. The CUDA it needs is the driver API,
`libcuda.so.1` or `nvcuda.dll`, which every NVIDIA driver installs. The CUDA
toolkit and runtime are not needed.

1. `cuInit`, then `cuGLGetDevices` to find the CUDA device the current GL
   context runs on, and `cuCtxCreate` on it. This is the analogue of the LUID
   match on Windows: decode on the GPU that renders, or report the pairing as a
   copy.
2. `cuvidCreateVideoParser` with sequence, decode and display callbacks.
   NVIDIA's parser reads the bitstream itself, so this backend needs no H.264
   control layer.
3. On the sequence callback, `cuvidGetDecoderCaps`, then `cuvidCreateDecoder`
   with NV12 output (P016 for 10-bit) and a surface count the callback names.
4. Per sample, `cuvidParseVideoData`. The decode callback calls
   `cuvidDecodePicture`; the display callback hands over pictures in display
   order with their timestamps.
5. Per displayed picture: `cuvidMapVideoFrame64` gives a device pointer and
   pitch; `cuGraphicsMapResources` on two GL textures registered once with
   `cuGraphicsGLRegisterImage` (an `R8` luma and an `RG8` chroma texture);
   `cuMemcpy2D` device to array for each plane; unmap both.

That is one copy per frame, GPU memory to GPU memory, about 3 MB at 1080p.
The same calls exist on Windows, so this backend needs no Direct3D.

ABI facts come from `dynlink_cuviddec.h`, `dynlink_nvcuvid.h` and
`dynlink_cuda.h` in nv-codec-headers (MIT), the repository `nvEncodeAPI.h`
already comes from. `tools/record_nvdec_abi.py` records sizes and offsets the
way `tools/record_nvenc_abi.py` does.

### AMD and Intel on Linux: VA-API VLD

The frame's journey:

```
MP4 sample (length-prefixed NAL units)
  -> parsed SPS, PPS and slice headers      h264 control layer, this package
  -> picture, IQ matrix, slice parameters   vaCreateBuffer
  -> slice data                             vaCreateBuffer (VASliceDataBufferType)
  -> decoded NV12 surface                   vaBeginPicture / vaRenderPicture / vaEndPicture
  -> DMA-BUF, two layers                    vaExportSurfaceHandle, once per surface
  -> R8 and GR88 GL textures                dmabuf.import_texture, once per surface
```

VA-API decode is slice-level, as its encode is. The driver does the entropy
decoding and reconstruction. The application parses the bitstream and does the
picture management:

- Parse SPS and PPS in full, scaling lists and VUI included, into
  `VAPictureParameterBufferH264` and `VAIQMatrixBufferH264`.
- Parse each slice header up to the slice data, into
  `VASliceParameterBufferH264`, including the bit offset where the macroblock
  data begins.
- Compute picture order counts (all three types), track `frame_num` and its
  gaps, and mark references by sliding window and by memory management control
  operations.
- Build reference lists 0 and 1, the initial order and the modifications, which
  libva takes in the slice parameters.
- Manage the decoded picture buffer: which surfaces hold references, which are
  free, and the bumping process that releases pictures in display order.

This is the decode half of what `vaapi/h264.py` is for encode, and larger,
because a decoder has to accept every stream a conforming encoder may produce,
while the encoder produces only the subset it chose. It lives in its own
module, beside the encoder's control layer. The bit writer in `h264.py` gains a
reader beside it.

The surface pool is fixed when the context is created: the DPB size from the
SPS, plus the pictures the player holds for display, plus one being decoded.
So each surface is exported and imported once, at open, and decoding never
creates a GL object. That is the decode form of `register()`.

Intel is the same code against `iHD`, as it is for encode. oneVPL decode over
VA-API would hide the control layer on Intel only, and the reasoning in
[VAAPI-BACKEND.md](VAAPI-BACKEND.md) against a second Intel path applies
unchanged.

### Intel on Windows: oneVPL

oneVPL's decoder is frame-level, like NVDEC. It parses the stream itself.

1. `MFXVideoDECODE_DecodeHeader` on the first sample with parameter sets, for
   the frame size and the surface count.
2. `IOPattern = MFX_IOPATTERN_OUT_VIDEO_MEMORY`, with the D3D11 frame allocator
   `vpl/encoder.py` already implements.
3. A VPP stage in the same session converts NV12 to `RGB4` into a D3D11 texture
   created through `windows/interop.py`, so OpenGL reads it as an ordinary RGBA
   texture inside a `for_drawing()`-style lock.
4. Per sample, `MFXVideoDECODE_DecodeFrameAsync`, `MFXVideoVPP_RunFrameVPPAsync`
   and `MFXVideoCORE_SyncOperation`.

The conversion runs on the GPU in the vendor library, where on Linux it runs in
a shader. The reason is the interop: `WGL_NV_DX_interop2` is specified for
colour formats, and whether a driver will register an NV12 texture is not
something to depend on before it is measured. If the Intel laptop registers
NV12 and its planes as GL textures, the VPP stage goes and the shader is used
here too.

### AMD on Windows: AMF

AMF's decoder component, `AMFVideoDecoderUVD_H264_AVC`, with
`AMF_MEMORY_DX11` output and the `AMFVideoConverter` component to BGRA, reaches
the same interop texture. It waits on the AMF encode backend, which builds the
C++ COM-style binding both share, and on AMD Windows hardware.

### Other options on Windows

Media Foundation ships a Microsoft H.264 decoder MFT on every Windows install,
which the encode plan's enumeration found (17 video decoders, against no
encoders). It takes a D3D11 device manager and returns D3D11 textures, on any
vendor. That makes it a vendor-neutral floor for decode where it was no use for
encode. It is not in the first phases, since the request is for each vendor's
own library, and it is recorded here so the option is not lost.

macOS is out of scope for the reason it is out of scope for encode. The decode
path there would be VideoToolbox into an IOSurface, read by
`CGLTexImageIOSurface2D`, and is a different design.

## Colour, orientation and size

The decoders produce NV12. A pass in this package converts it to RGBA:
a GL 3.3 core program that samples the `R8` and `RG8` planes and draws a single
triangle into the output texture's framebuffer. It lives in `convert.py` and is
the only conversion the Linux and NVIDIA paths have, so all three backends
produce the same RGB from the same stream.

- Matrix and range come from the stream's VUI: BT.709 or BT.601 coefficients,
  limited or full range. A stream that does not say is BT.601 below 720 rows and
  BT.709 at or above, which is the rule players use.
- Orientation is fixed in the same pass. A decoded surface's first row is the
  top of the picture, and a GL texture's first row is its bottom, so the shader
  samples with its vertical coordinate reversed. The output texture is the right
  way up for OpenGL, and `MovieTexture`'s `[::-1]` goes away.
- The output size is the stream's cropped size. Coded sizes are rounded up to
  whole macroblocks (1088 rows for 1080p) and drivers pad further (1280 rows
  here), so the shader samples only the visible rectangle.
- The output texture is `GL_SRGB8_ALPHA8`, so the PBR pass linearises a movie
  frame when it samples it, as it does an 8-bit colour image. Video's BT.709
  transfer curve is treated as sRGB, which is how browsers display it. The
  OpenGLContext phase confirms this matches what `ImageTexture` produces.
- Mipmaps, where the sampler wants them, are made after the conversion, once a
  frame.
- 10-bit streams decode to P010/P016 and convert through the same shader with
  `R16` and `RG16` planes. HDR transfer functions (PQ, HLG) are out of the first
  phases.

## Containers

`mp4.py` gains `MP4Reader`, the reverse of `MP4Writer`: `moov` and the sample
tables (`stsd` with `avcC`, `stts`, `ctts`, `stss`, `stsc`, `stsz`, `stco` and
`co64`), read without reading `mdat`. A sample is read when it is asked for, so
opening a two-hour movie costs the size of its index.

Each sample comes out as Annex-B, length prefixes replaced by start codes, with
the parameter sets from `avcC` placed in front of the first sample and in front
of the first sample after every seek. All three decoders take that form.

- Raw `.h264` Annex-B files are read too. They have no index, so seeking scans
  for IDR pictures once and keeps the positions.
- The reader works from a path or a binary file object, with no GL, so it can be
  opened on OpenGLContext's loader pool.
- Fragmented MP4 (`moof`), WebM and Matroska are later phases. WebM is where
  VP9 and AV1 usually live, and Matroska is specified in RFC 9559.
- Audio tracks are reported (codec, sample rate, channels, the codec's own
  configuration) and their samples are readable with their timestamps. What
  happens to them next is under *Audio*.

The container specification is ISO/IEC 14496-12, which is on ISO's list of
freely available standards. `avcC` is defined in ISO/IEC 14496-15, which is
not free; the fields the reader needs are the ones `MP4Writer` already writes.
Both the reader and the writer cite the two specifications once the reader
lands, and NOTICES.md records them.

## Audio

**Deferred** (2026-10-09): the video phases go first. The A phases below wait,
and so does the choice of decoder.

The candidate to evaluate first is OxideAV
(<https://github.com/OxideAV/oxideav-workspace>), MIT-licensed and pure Rust,
wrapped for Python with PyO3 or through a C ABI. As its README stated on
2026-10-09, it links no C codec library and decodes AAC (LC with SBR and PS,
about 96% of conformance), Opus, Vorbis, FLAC, MP3 and AC-3, and demultiplexes
MP4, Matroska/WebM and Ogg. That would close the AAC gap below on every
platform. Its H.264 decoder could also be the software floor that *Open
questions* asks for. Before it is adopted: its full dependency tree is checked
for copyleft crates, and its decoders are measured against the conformance
streams. It is at an early stage (42 stars, many codecs marked partial).

The design below assumes omi_audio's existing decoders, and holds with OxideAV
put in their place.

A movie's sound is demultiplexed here and decoded by omi_audio, which already
owns the audio decoders and the mixer. This package does not decode audio. The
GPU has no audio decoder, so nothing here would be gained by doing it in
`pyopengl-video`, and omi_audio's decoders are the ones a game already ships.

What omi_audio has, all permissively licensed:

| Codec | In MP4 as | omi_audio decoder | Licence |
| --- | --- | --- | --- |
| Opus | `Opus` sample entry, `dOps` | `libopus` through `_opus.py` | BSD-3-Clause |
| FLAC | `fLaC` sample entry, `dfLa` | miniaudio (dr_flac) | MIT / public domain |
| MP3 | `mp4a`, object type 0x6B | miniaudio (dr_mp3) | MIT / public domain |
| Vorbis | rare in MP4; WebM | miniaudio (stb_vorbis) | MIT / public domain |
| AAC | `mp4a`, object type 0x40, `esds` | none | see below |

AAC is the audio in nearly every MP4 a game developer will be handed, and no
decoder under an MIT or BSD licence covers it:

- FFmpeg's and GStreamer's AAC decoders are LGPL or GPL, and excluded.
- Fraunhofer's FDK AAC library is not copyleft, but its licence grants no
  patent rights and is not OSI-approved. Loading it by name at run time where
  the user installed it, as the vendor video libraries are loaded, is possible.
  Whether to offer that is the maintainer's decision, and is open below.
- The operating system's own decoder is the other route: Media Foundation's AAC
  decoder MFT ships with Windows, and AudioToolbox with macOS. Linux has none in
  its base system.

So the work splits in two:

| Phase | Content |
| --- | --- |
| A1 | `MP4Reader` reports audio tracks and reads their samples with timestamps (part of D1) |
| A2 | omi_audio gains a streaming source fed from compressed packets, so a two-hour soundtrack is not decoded whole (2 hours of 48 kHz stereo float32 is 2.7 GB); Opus, FLAC and MP3 from MP4 decode through its existing decoders |
| A3 | omi_audio gains multichannel playback for a soundtrack that is not placed in the world; its `Clip` is mono today, which suits an in-world screen and not a cut scene |
| A4 | AAC: Media Foundation on Windows, AudioToolbox on macOS, and FDK AAC by name on Linux if the maintainer accepts its licence |
| A5 | The recorder writes Opus into MP4 beside the video, so every movie this package produces plays its sound with the permissive decoders |

The player's clock is the audio's clock when a movie has sound: `frame_at()`
is asked for the time the audio device has played to, so the picture follows
the sound and drift between the two clocks never accumulates.

## The interface

A `Decoder` sits beside `Encoder`, in `decoder.py`, with a registry built the
same way. `EncoderError` and the new `DecoderError` share a base, `VideoError`,
so one `except` covers a program that does both.

```python
from pyopengl_video import decoders, open_decoder

for backend in decoders():
    print(backend.name, backend.vendor, sorted(backend.codecs), backend.max_size)
```

```python
class Decoder:
    """Decodes a compressed video stream into OpenGL textures."""

    codec: str                     # 'h264' first
    size: tuple[int, int]          # the cropped picture size
    timescale: int                 # of the timestamps samples carry
    colour: Colour                 # matrix, range, primaries, transfer
    zero_copy: bool                # frames never pass through host memory
    output_slots: int              # frames a caller may hold at once

    def decode(self, sample: Sample) -> list[DecodedFrame]: ...   # display order
    def flush(self) -> list[DecodedFrame]: ...
    def reset(self) -> None: ...   # drop every reference, before a seek
    def close(self) -> None: ...
```

`decode()` returns a list for the encode side's reason, reversed: a decoder
holds pictures back until it knows their display order, so a sample in may
produce no frame, or several.

A `DecodedFrame` carries its timestamp and duration, its layout and its GL
textures: two planes for NV12, one texture where a backend converted already.
It belongs to the decoder's pool until `release()`, and a decoder with no free
surface raises an error that names the frames still held. That is the decode
form of the encoder's `input_slots` rule.

Above the decoder, `playback.py` holds what any player needs, with no
scenegraph:

```python
movie = Movie.open('intro.mp4')          # any thread: reads the index, no GL
player = movie.player()                  # GL thread: opens a decoder, a converter, a ring
texture = player.frame_at(seconds)       # the RGBA texture shown at that time
```

`frame_at()` keeps the meaning `MovieFile.frameAt` has in OpenGLContext: the
last frame at or before the time, the first frame before the start, the last
past the end. A time behind the current frame, or more than a set distance
ahead of it, resets the decoder and starts from the key frame before it (from
`stss`). Playing forward decodes each frame once. Playing backward decodes from
the key frame before each frame shown, which costs up to one group of pictures
per frame. A reverse cache of decoded frames would remove that cost, and it
waits until a game needs reverse playback.

The player decodes ahead into a ring of output textures, so `frame_at()` for
the next frame finds it already converted. Its depth is `output_slots` less
the decoder's own reorder depth.

The engine's clock decides what time is asked for, so a fixed-step capture
gets every frame in order and never drops one. When real time runs ahead of
decoding, the player shows the latest frame it has and skips ahead to the next
key frame once it is more than a group of pictures behind.

## Threading

Phase D2 decodes on the GL thread, inside `frame_at()`. That is correct and
measurable. Hardware decode is asynchronous on every backend, so submission
returns before the picture exists and the per-frame Python work is the slice
header parse and the buffer fills.

Decoding ahead on a worker thread follows once that is measured. What may move
off the GL thread differs by backend: VA-API submission needs no GL, so only
the conversion stays; NVDEC runs its parser and decoder under a CUDA context
pushed on the worker, and the CUDA-GL map stays on the GL thread; oneVPL's
session is used from one thread at a time. The X3D plan lists "decoding ahead
of the drawing thread" as open, and this is the work that closes it.

Several movies at once are expected: a game with screens on its walls. Each
movie is its own decoder session. A test plays four 1080p streams in one
context and holds the frame time.

## OpenGLContext

- `MovieTexture` uses `Movie` and its player. `loaders/movie.py` becomes the
  adapter: `Movie.open` on the loader pool, the player made in `cached()` on the
  first draw in a context, `frame_at(self.position())` each frame.
- The PyAV `movie` extra and its import go. The `video` extra, which already
  names this package, covers recording and playback.
- Without a decoder for the movie's codec on this machine, the shape draws
  untextured and the reason from `explain()` is logged once, as it is for a
  missing PyAV now.
- `docs/x3d.rst` and the capture and loading pages say which decoders exist and
  how a game learns which one it got. `X3D-SUPPORT.md` and `PROJECT-PLAN.md`
  are updated where they name PyAV.

## Testing

- The container reader is tested against `MP4Writer`'s own output: every
  sample's bytes, timestamp, composition offset and key flag back as written.
  No GPU.
- The control layer is tested against the parameter sets and slice headers our
  three encoders write, parsed back field by field. No GPU.
- Conformance: the ITU-T H.264.1 conformance bitstreams that fall in scope
  (progressive Main and High) decode to the published checksums of their
  decoded frames. They are fetched by a tool and cached outside the
  repository, and their terms are checked before anything is fetched in CI.
- Round trip: encode known frames on each backend, decode on the same machine,
  compare colour and position within a tolerance. This is the check the encode
  plan wanted and wrote down as owed.
- Cross-vendor: a stream from NVENC decodes on VA-API and the reverse, which is
  where a parser that matches only its own encoder fails.
- The conversion pass is tested with NV12 planes uploaded from host memory
  under every matrix and range, against values computed in NumPy. It needs a GL
  context and no decoder.
- Hardware gating follows the encode suite: each backend's tests skip with the
  reason when its hardware is absent, and the README table says "untested"
  until a run says otherwise.
- The ABI harnesses grow tables for the libva decode structures and for the
  NVDEC and CUDA structures, and for the oneVPL decode fields.

## Phases

| Phase | Content | Needs |
| --- | --- | --- |
| D1 | `MP4Reader` and the Annex-B reader; the H.264 bit reader and SPS, PPS and slice header parser | landed |
| D2 | `Decoder`, `DecodedFrame`, the registry and `VideoError`; the NV12 conversion pass; `Movie` and the player | landed |
| D3 | VA-API H.264 decode with its control layer: POC, DPB, reference lists | landed on AMD; Intel/Linux is the same code, untested |
| D4 | OpenGLContext: `MovieTexture` on the player, the PyAV extra removed, docs | D3 |
| D5 | NVDEC with CUDA-GL interop, Linux and Windows | an NVIDIA machine |
| D6 | oneVPL decode with VPP into the interop texture | Intel/Windows, the laptop |
| D7 | Decoding ahead on a worker thread, after D3 to D5 are measured | D5 |
| D8 | HEVC and AV1 on VA-API (control layers of their own); AMF decode; fragmented MP4 and WebM | hardware per row |

D1 and D2 go first because every backend needs them and neither needs a
particular GPU. D3 is on this machine, so the first movie plays here, and D4
makes `MovieTexture` shippable on AMD and Intel Linux before the other vendors
land. NVIDIA owners keep an untextured shape until D5.

## What landed, and where

- `mp4.py`: `MP4Reader`, `VideoTrack`, `AudioTrack`; `annexb.py`: `AnnexBReader`.
- `vaapi/h264.py` gained `BitReader` and `rbsp_of`; `vaapi/h264_parse.py` and
  `vaapi/h264_dpb.py` are the parser and the decoded picture buffer.
- `backend.py` holds `Backend[T]`, discovery and `VideoError` for both
  directions; `decoder.py` the interface; `vaapi/device.py` the libva display
  the encoder and the decoder share.
- `colour.py`, `convert.py` (with `glstate.preserved_state`, the project's
  sanctioned OGC151 state manager) and `playback.py` (`Movie`, `Timeline`,
  `Player`).
- `examples/play_movie.py`: a GLFW player in plain PyOpenGL.
- PyOpenGL: `glGetTexImage` sizes `GL_RG`, `GL_RG_INTEGER`, `GL_BGR_INTEGER`
  and `GL_BGRA_INTEGER` without the extension module imported.

Not yet done from the sections above: the ITU-T H.264.1 conformance streams
(the fetch tool and its terms check), the cross-vendor test (an NVENC stream on
VA-API), four 1080p movies in one context, and P010 for 10-bit streams.

## Open questions

- A floor for machines with no hardware decoder: CI runners, llvmpipe, a GPU
  older than its codec. Cisco's OpenH264 is BSD-licensed and can be loaded by
  name at run time as the vendor libraries are. Which profiles its decoder
  handles in full, and whether Cisco's binary licence suits a game, are both to
  be checked before it is proposed. OxideAV's H.264 decoder (see *Audio*) is
  the other candidate.
- Whether WGL_NV_DX_interop2 registers an NV12 texture on the Intel laptop. Yes
  removes the VPP stage from D6.
- Interlaced streams (field pictures, MBAFF, PAFF). The first VA-API control
  layer refuses them with an error that names the stream's structure. Games
  ship progressive video, and a recording from this package is always
  progressive.
- AAC: OxideAV wrapped for Python (see *Audio*), or FDK AAC loaded by name
  where the user installed it, or AAC left silent until one of those lands.
  Either licence is read in full before this is decided.
- Output format for HDR video (PQ or HLG), where the engine would want a float
  texture rather than an sRGB one.

## Licensing

- nv-codec-headers (MIT) for the NVDEC and CUDA driver ABI, read and cited as
  the NVENC binding is.
- libva headers (MIT), libvpl headers (MIT), AMF (MIT).
- ITU-T H.264 and H.265, ISO/IEC 14496-12 and 14496-15, and RFC 9559 are the
  specifications the parsers and readers are written from.
- `WGL_NV_DX_interop2` and the CUDA driver API's GL interop are documented
  interfaces.
- Excluded as in *What is excluded*: FFmpeg, PyAV, GStreamer, and the source of
  any copyleft decoder. NOTICES.md records each source as the binding that uses
  it lands.
