# Hardware video encode from an OpenGL colour buffer

**Status:** In progress -- NVIDIA on Linux, Intel on Windows, and AMD on Linux
through VA-API all record end to end. NVIDIA on Windows is next, over the
interop shim the Intel backend already uses; Intel on Linux is the VA-API
backend against a different driver and needs only testing. Platform and vendor
coverage is the table in *Phases*.
**Scope:** `pyopengl-video`, a standalone library on top of PyOpenGL, consumed by
OpenGLContext's frame recorder
**Goal:** turn what the engine just rendered into an H.264 stream at full frame
rate, on NVIDIA, AMD and Intel, without the frame making a round trip through
host memory.

## Why this exists

Anyone building a game on OpenGLContext wants to record it: a trailer, a bug
report, a regression video, a replay. Doing that today means a per-frame
`glReadPixels` into Pillow, which stalls the pipeline on every frame and caps out
long before a game's frame rate. Every GPU we support has a dedicated video
encoder sitting idle on the same die as the renderer, reachable from the same
process, and the frame is already in GPU memory where the encoder wants it.

The capability is a library feature, not a demo feature: `pyopengl-video` gives
any PyOpenGL application an encoder that takes a texture and returns a video
file, and OpenGLContext wraps it in a recorder that knows about frame pacing and
the engine's clock.

## Where this lives, and why it is not in PyOpenGL

`pyopengl-video` is its own repository and its own distribution, depending on
PyOpenGL and nothing else at runtime.

It adds nothing to PyOpenGL's API. Everything it needs from GL is already
bound — framebuffer blits, texture objects, Pixel Buffer Objects, fences — and
the one extension the zero-copy path turns on, `EGL_MESA_image_dma_buf_export`,
already ships as `OpenGL.EGL.MESA.image_dma_buf_export`. What is left is a
pattern of calls into three vendor encoder libraries that happen to accept GL
object names, which is a consumer of PyOpenGL rather than a part of it.

Keeping it separate also keeps three kinds of churn out of a library that many
projects pin: vendor SDK versions move on their own schedule, the encoder
backends need hardware that PyOpenGL's CI does not have, and a video library
grows containers and colour management that have no business in a bindings
package. PyOpenGL stays what it is; a recording application installs one more
distribution.

## Verified ground truth

Measured on the development machine (RTX 3060 Ti, driver 580.173.02, Ubuntu
24.04 container, GLFW/Wayland, GL 4.6 core):

| Fact | Result |
| --- | --- |
| `libnvidia-encode.so.1` present, `NvEncodeAPICreateInstance` succeeds | yes |
| Driver's maximum NvEncodeAPI version | 13.0 |
| `nvEncOpenEncodeSessionEx` with `NV_ENC_DEVICE_TYPE_OPENGL`, `device = NULL` | succeeds, no CUDA context in the process |
| Codecs advertised by the session | 2 (H.264, HEVC — this part has no AV1 encoder) |
| H.264 caps | 4096x4096 maximum, 983040 macroblocks/s (~1080p at 120 fps) |
| Throughput through a host-memory NV12 input, 1280x720 | 120 frames in 0.161 s (744 fps), including frame synthesis |
| `libva.so.2`, `libva-drm.so.2`, `libvpl.so.2` | present |
| `/dev/dri/renderD128` | the NVIDIA card; no AMD or Intel GPU is present here |

Two consequences. First, the encoder is never the bottleneck — at 744 fps
through the *slow* path, encoding costs well under a millisecond per frame, and
the whole design question is how the frame reaches it. Second, the AMD and Intel
backends cannot be tested on that machine; they need hardware or a CI runner
that has it.

Measured on the AMD machine (Radeon 8060S, `gfx1151`, Mesa 25.2.8 `radeonsi`,
Ubuntu 24.04 container, GLFW/Wayland, GL 4.6):

| Fact | Result |
| --- | --- |
| `libva.so.2` with `mesa-va-drivers` on `/dev/dri/renderD128` | VA-API 1.20 |
| H.264 Main and High, `VAEntrypointEncSlice` | yes |
| `VAProfileNone` / `VAEntrypointVideoProc`, for the RGB to NV12 pass | yes |
| `VAConfigAttribRateControl` | CBR, VBR and CQP |
| `VAConfigAttribEncMaxRefFrames` | 1 for reference list 0 |
| `VAConfigAttribEncPackedHeaders` | 0x1f -- every kind, including the slice header |
| `EGL_MESA_image_dma_buf_export` on the GLFW EGL context | yes |
| Modifier reported for an exported `GL_RGBA8` texture | `DRM_FORMAT_MOD_INVALID` |

The last two decided the shape of the backend. The export works and the
imported surface reads correctly despite the driver naming no modifier, so the
plan's original direction -- GL allocates, libva imports -- holds on AMD, and
`new_input()` does not have to allocate through the encoder as it does on
Windows.

The reference-frame limit of one is not a restriction to work around: a stream
of I and P pictures needs exactly one, which is what this backend produces.

## The interop object is a texture, not a PBO

A Pixel Buffer Object is a GL buffer object in GPU memory, and no encoder API
accepts one. NVENC's OpenGL input resource type takes a texture name and a
texture target. VA-API takes a DMA-BUF file descriptor, and the only standard way
to produce one from GL is to wrap a *texture or renderbuffer* in an `EGLImage`
and export that with `EGL_MESA_image_dma_buf_export`. There is no equivalent
export for buffer objects.

So the frame path has three tiers, and a PBO is what tier 2 is built from:

1. **Zero-copy texture handoff.** The recorder owns a ring of RGBA8 textures and
   blits the finished frame into one with `glBlitFramebuffer`. The texture is
   handed to the encoder by name (NVIDIA) or as a DMA-BUF (AMD, Intel). Nothing
   crosses PCIe; the frame is read by the encoder from the memory the renderer
   wrote it to.
2. **PBO readback into a hardware encoder.** The same blit, then an asynchronous
   `glReadPixels` into a fenced ring of Pixel Buffer Objects, and the mapped
   bytes go to the encoder's host-memory input. One PCIe crossing each way,
   never a pipeline stall, and it works on any driver that has an encoder but no
   working interop path.
3. **PBO readback into a software encoder.** The portable floor: llvmpipe,
   macOS, a machine whose vendor SDK is missing. Left as a hook — see
   *Licensing* for why nothing GPL ships in the box.

Tier 1 and tier 2 differ only in how the encoder's input is fed, which is why
they sit behind one interface. Tier 2 is worth building for its own sake: it is
the fallback that keeps the feature working everywhere, and it is the reference
the zero-copy paths are checked against, because both must produce the same
picture.

## What the build settled

The NVENC backend, the MP4 muxer and the encoder interface are written and
tested against real hardware. Four things came out of building it that were not
knowable from the header, and that the rest of the work depends on.

**The encoder reads a texture in memory order.** The first row of texture data
becomes the top row of the picture. A frame *rendered* into a framebuffer is
therefore upside down by the time it reaches the encoder, because OpenGL's
framebuffer starts at the bottom left. The fix is free -- a `glBlitFramebuffer`
with its destination Y coordinates reversed flips the frame as it copies -- but
it has to be there, and the recorder is where it belongs. Verified by decoding a
recording of a known shape: apex-up triangle, apex up in the video, nothing
outside the rows it was drawn in.

**Colour needs no conversion pass, only a declaration.** Handing NVENC an RGBA
texture and letting the hardware convert produces limited-range BT.709 exactly
as the video usability information declares it -- checked against the decoded
luma and chroma of known colours. No shader, no second pass.

**One input texture is not enough, and the driver will not say so.** A
registered resource cannot be mapped twice at once, so an encoder holding a
frame back for reordering rejects the texture it is still reading -- with a bare
`MAP_FAILED`. The encoder now reports `input_slots`, computed from the
configuration the driver settled on rather than from what was asked for, since a
preset can turn lookahead on by itself; handing back a texture still in flight
raises an error that names the cause.

**A hand-written ABI needs a machine to check it.** `tools/record_nvenc_abi.py`
compiles a probe against the real header and records every structure's size and
every field's offset; the recorded table is checked in and tested against on
machines with no compiler and no driver. It found two layout faults on its first
run, one of them a union whose alignment came from an arm the binding does not
declare -- neither of which any runtime error would have pointed at.

The output was checked from outside as well as inside: an independent
NVDEC-based decoder demuxes the written MP4, returns every frame in order, and
finds the recorded content where it was drawn.

## Per-vendor mechanisms

### NVIDIA — NVENC, OpenGL device type

`libnvidia-encode.so.1` exposes exactly two symbols:
`NvEncodeAPIGetMaxSupportedVersion` and `NvEncodeAPICreateInstance`. The latter
fills a struct of function pointers, and everything else is called through it.
The sequence for the zero-copy path:

1. `NvEncodeAPICreateInstance` once per process, with the struct version built
   from `min(header version, driver version)`.
2. `nvEncOpenEncodeSessionEx` with `deviceType = NV_ENC_DEVICE_TYPE_OPENGL` and
   `device = NULL`. A GL context must be current on the calling thread, and it
   must be one on the NVIDIA GPU that holds the encoder; the session is bound
   to it for its lifetime. Both conditions come back as
   `NV_ENC_ERR_UNSUPPORTED_DEVICE`, so the encoder asks `GL_RENDERER` which of
   them it met before it raises.
3. `nvEncGetEncodePresetConfigEx` for the chosen preset and tuning, then adjust
   GOP length, rate control and `repeatSPSPPS` on the returned config.
4. `nvEncInitializeEncoder`.
5. `nvEncRegisterResource` per texture in the ring, with
   `resourceType = NV_ENC_INPUT_RESOURCE_TYPE_OPENGL_TEX`, `resourceToRegister`
   pointing at a `{texture, target}` pair, `bufferFormat = ABGR` and
   `pitch = width * 4` (the header specifies pitch as texture width times
   component count for GL textures).
6. Per frame: `nvEncMapInputResource`, `nvEncEncodePicture`,
   `nvEncLockBitstream` / `nvEncUnlockBitstream`, `nvEncUnmapInputResource`.

`NV_ENC_BUFFER_FORMAT_ABGR` is word-ordered with blue in the low byte, which on a
little-endian host is the byte order of a `GL_RGBA8` texture — so the ring
textures are ordinary RGBA and NVENC does the colour conversion in hardware.

Details that shape the code:

- **Deferred output.** With B-frames or lookahead enabled, `nvEncEncodePicture`
  returns `NV_ENC_ERR_NEED_MORE_INPUT` and produces nothing. The encoder must
  keep a queue of (bitstream buffer, mapped input, timestamp) and drain it in
  order when a later call succeeds; the mapped input is released only once its
  frame has come out.
- **Ring of inputs.** A single input texture would be overwritten by the next
  frame while the encoder still refers to it. Three or four textures, rotated,
  with the ring length driven by the encoder's queue depth.
- **Session limits.** GeForce drivers cap concurrent NVENC sessions per process
  and per system, so the recorder opens one session and closes it deterministically.
- **Alignment and size.** H.264 tops out at 4096x4096 on this part; 4K UHD fits,
  anything wider needs HEVC. Odd dimensions are the caller's problem to round.
- **Linux is synchronous.** `enableEncodeAsync` is a Windows event mechanism;
  on Linux the lock call blocks until the frame is ready.

ABI facts come from `nvEncodeAPI.h` as published in
[nv-codec-headers](https://github.com/FFmpeg/nv-codec-headers) (MIT, header
version 13.1). The binding cites the header and its version; struct layouts are
checked against it by a test (see *Testing*).

The concrete call sequences, the surface import and the size of the job are in
[VAAPI-BACKEND.md](VAAPI-BACKEND.md).

### Intel — VA-API, the same as AMD

`libva` is the interface every Linux Intel part has had since Gen9, driven by the
`iHD` media driver. The zero-copy handoff:

1. Wrap the ring texture in an `EGLImage` (`eglCreateImage` with
   `EGL_GL_TEXTURE_2D`), export it with `EGL_MESA_image_dma_buf_export` to get a
   DMA-BUF fd, a modifier, an offset and a stride.
2. `vaCreateSurfaces` with `VASurfaceAttribExternalBuffers` and
   `VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2`, describing the imported RGBA plane.
3. VA-API encoders consume NV12, so convert with a VPP pass
   (`VAEntrypointVideoProc`, `vaCreateContext` + `vaBeginPicture` on a
   `VAProcPipelineParameterBuffer`) from the imported RGBA surface into an
   NV12 surface the driver allocated. The conversion runs on the GPU's fixed
   function blocks; nothing reaches host memory.
4. Encode with `VAEntrypointEncSlice` for `VAProfileH264Main`/`High`: build the
   sequence, picture and slice parameter buffers, `vaRenderPicture`,
   `vaEndPicture`, then `vaMapBuffer` on the coded buffer to read the bitstream.

Intel also has **oneVPL** (`libvpl.so.2`, already present here), a smaller
interface that hides the sequence and picture parameter assembly and sits on
VA-API underneath. It is the pleasanter binding to write, and it is not the one
being written: what it saves has to be built for AMD in any case, and once it
exists, Intel is that same code against a different driver. One interface, one
control layer, both vendors — the reasoning, and what would reopen it, is in
[VAAPI-BACKEND.md](VAAPI-BACKEND.md).

### AMD — VA-API on Mesa, AMF where it exists

Mesa's VA-API state tracker exposes the VCN encoder on `radeonsi`, so **the
Intel path above is the AMD path**, unchanged apart from which driver `libva`
loads. That is why there is one backend rather than two vendor SDKs: one
binding, one control layer, two vendors, and it is the configuration most Linux
users are actually in.

AMD's own SDK, [AMF](https://github.com/GPUOpen-LibrariesAndSDKs/AMF) (MIT), is
the alternative. It has a genuine OpenGL memory type (`AMF_MEMORY_OPENGL`) and
its `AMFData` interface converts between DX, OpenCL, OpenGL, Vulkan and host
memory, which would give a texture-in interface much like NVENC's. Its Linux
runtime ships with the Pro driver and, from driver 25.20, as a separate package.
The catch is that it is present only on that stack — a Mesa user does not have
it — and it is a C++ COM-style interface, which is a far heavier ctypes binding
than libva's flat C. AMF is therefore phase 4, valuable mostly for Windows,
where it is the native path.

### Windows -- one shim, then each vendor's own encoder

**NVIDIA's OpenGL device type does not exist there**, so a texture cannot be
handed to NVENC by name on Windows however good the hardware is. What every
Windows encoder does understand is an `ID3D11Texture2D`, and `WGL_NV_DX_interop2`
makes one object that is simultaneously a D3D11 texture and a GL texture. That is
the same shape the DMA-BUF export has on Linux: a handle, resolved by the driver,
rather than a pointer.

The extension's name is historical -- Intel's Windows driver exposes it, as does
AMD's -- so **one interop mechanism serves all three vendors**, and each vendor's
encoder then reads the D3D11 texture: NVENC through
`NV_ENC_DEVICE_TYPE_DIRECTX`, Intel through oneVPL with `MFX_IMPL_VIA_D3D11`,
AMD through AMF's `AMF_MEMORY_DX11`.

Two things measured on real hardware shape the design, and both are recorded with
the rest in [WINDOWS-SUPPORT.md](WINDOWS-SUPPORT.md). **Zero-copy needs the
encoder and the GL context on the same adapter**, which on a switchable-graphics
laptop they are not by default; discovery therefore matches the DXGI adapter to
the GL context's `GL_DEVICE_LUID_EXT` and offers the encoders that live there.
And **Media Foundation is not a dependable floor**: a machine with working
QuickSync hardware registered no video encoder MFT at all.

### What does not work

NVIDIA cannot use the VA-API path: `nvidia-vaapi-driver` is a decode-only shim
over NVDEC, and NVIDIA's EGL does not implement `EGL_MESA_image_dma_buf_export`.
Two backends are genuinely required — this is not a case where one abstraction
covers every vendor.

## Architecture

### `pyopengl-video`

```
pyopengl-video/
    pyproject.toml            requires: PyOpenGL
    src/pyopengl_video/
        __init__.py           encoders(), open_encoder() -- discovery and construction
        encoder.py            Encoder ABC, Packet, EncoderUnavailable, capability records
        readback.py           fenced PBO ring: the portable frame source (tier 2)
        inputs.py             InputHandle: the texture an encoder reads, and its
                              framebuffer
        mp4.py                MP4 muxer: Annex-B in, a playable file out
        linux/                what Linux backends share, as windows/ is for Windows
            dmabuf.py         EGLImage export and import, over
                              OpenGL.EGL.MESA.image_dma_buf_export
        windows/              the Windows interop shim, shared by every backend there
            com.py            GUIDs, vtable calls, HRESULT checking
            d3d11.py          DXGI adapter identity (by LUID), D3D11 device, textures
            interop.py        WGL_NV_DX_interop2: one object, a D3D11 and a GL texture
        nvenc/
            api.py            ctypes NvEncodeAPI: structs, GUIDs, function list, errors
            encoder.py        NVENCEncoder -- GL texture (Linux) or D3D11 (Windows) in
        vpl/
            api.py            ctypes oneVPL: the packed mfx structures, the allocator
            encoder.py        VPLEncoder -- D3D11 texture in, Annex-B out (Intel)
        vaapi/
            api.py            ctypes libva: display, config, context, surfaces, buffers
            h264.py           the control layer libva leaves to the caller: the
                              bitstream writer, the parameter sets, the slice
                              headers, and the GOP and DPB bookkeeping
            encoder.py        VAAPIEncoder -- dmabuf in, Annex-B out (Intel and AMD)
    tests/
    plans/
```

Discovery returns what this machine can actually do, so a caller can choose
rather than guess:

```python
from pyopengl_video import encoders, open_encoder

for backend in encoders():
    print(backend.name, backend.vendor, backend.codecs, backend.max_size)

encoder = open_encoder(1920, 1080, codec='h264', fps=60, bitrate=12_000_000)
```

The uniform interface every backend implements:

```python
class Encoder:
    """Encodes GL colour buffers to a compressed video stream."""

    codec: str                  # 'h264', 'hevc'
    size: tuple[int, int]
    zero_copy: bool             # True when frames never reach host memory
    allocates_inputs: bool      # True when only new_input() can make an input

    def new_input(self) -> InputHandle: ...
    def register(self, texture, target=GL_TEXTURE_2D) -> InputHandle: ...
    def encode(self, handle, timestamp, duration, force_idr=False) -> list[Packet]: ...
    def flush(self) -> list[Packet]: ...
    def headers(self) -> bytes            # SPS/PPS, for containers that want them up front
    def close(self) -> None
```

`Packet` carries the compressed bytes, presentation timestamp, duration and a
keyframe flag — everything a muxer needs and nothing about how it was made.
Backends that reorder frames return packets in decode order with correct
timestamps, and say so through a `reorders_frames` attribute so a muxer knows it
must track composition offsets.

`encode()` returning a *list* is the deferred-output rule made part of the
interface: zero packets is normal, two is normal, and no caller may assume one
frame in means one packet out.

**The encoder allocates its own inputs.** On Linux a caller can make an RGBA
texture and hand it over; on Windows the texture has to be a D3D11 resource
before it can be a GL one, so only the backend can create it. `new_input()` is
therefore the way to get an input on every backend, and the handle it returns
carries the GL texture name and a framebuffer with that texture attached.
Drawing into one happens inside `handle.for_drawing()`, which is where a shared
texture changes hands between D3D and GL and where a backend that needs no such
scope does nothing. `register()` remains for a caller bringing its own texture to
a backend that accepts one.

### OpenGLContext: the recorder

Policy, not mechanism, lives here:

- `OpenGLContext/video/recorder.py` — `VideoRecorder`, driven from the context's
  `SwapBuffers` exactly as `SettleCapture` is: owns the texture ring, does the
  blit, calls the encoder, feeds the muxer.
- A fixed-step recording clock. `OpenGLContext.events.systemtime.systemTime()`
  is already the single wall-clock source the timer subsystem reads, so it grows
  a settable source; the recorder installs a clock that advances by exactly
  `1/fps` per rendered frame. Every animation driven by a `TimeSensor` then
  advances in lockstep with the frames being written, and the video is smooth and
  reproducible however long a frame took to render. Demos that keep their own
  `time.time()` call switch to the engine clock so recording works for them too.

The split is deliberate: `pyopengl-video` knows how to talk to encoders,
OpenGLContext knows when to. `pyopengl-video` is usable from any PyOpenGL
application with no scenegraph in sight, and OpenGLContext depends on it only
when a recording is asked for.

## Colour, timing and correctness

- **Colour conversion.** Handing the encoder RGBA and letting it convert is one
  less pass and one less thing to get wrong, and every backend can do it (NVENC
  natively, VA-API through VPP). The cost is that the conversion matrix is the
  driver's choice, so the stream must *say* what it contains: the H.264 VUI
  fields for colour primaries, transfer characteristics and matrix coefficients
  are set explicitly rather than left unspecified, and the recorder documents
  that it writes limited-range BT.709. A GL-side conversion shader remains an
  option if a caller needs full-range or a different matrix.
- **Timestamps.** The encoder is given presentation timestamps in a 90 kHz
  timescale derived from the fixed-step clock, not from the wall clock. Constant
  frame rate is the default because it is what editors and browsers handle best.
- **Reordering.** With B-frames the packet order is decode order and the muxer
  must write composition offsets. The first implementation sets
  `frameIntervalP = 1` (no B-frames) so packet order is frame order, and turns
  B-frames on only once the muxer tracks composition time properly.
- **Pacing.** The recorder must not let the encoder's queue grow without bound;
  if the encoder falls behind the renderer the recorder blocks on drain rather
  than dropping frames, because a dropped frame in a fixed-step recording is a
  visible stutter, not a lost millisecond.

## Container

The encoders emit Annex-B elementary streams. Writing `.h264` directly is
useful for tests and for piping, but the deliverable people want is an `.mp4`,
so `pyopengl-video` carries a small MP4 muxer of its own — a video library that
can only emit elementary streams is half a library. `ftyp`, a progressively
written `mdat` whose size is patched on close, and a `moov` assembled from the
sample table held in memory. NALUs are rewritten from start-code delimited to
length-prefixed, and the SPS/PPS go into an `avcC` box.

This is a few hundred lines and it is ours, which is the point — see below.

## Licensing

Everything named here is permissive, and that is a requirement rather than a
happy accident:

- `nvEncodeAPI.h` via nv-codec-headers — MIT. Read for ABI facts, cited in the
  binding's docstring.
- The oneVPL headers via [intel/libvpl](https://github.com/intel/libvpl) — MIT.
  Read the same way, for structure layouts and packing, and cited rather than
  vendored. No vendor sample code is copied from either.
- `libva` — MIT. AMF — MIT.
- `WGL_NV_DX_interop2`, and D3D11/DXGI/Media Foundation — a registry
  specification and documented platform interfaces.
- No clean-room procedure is needed for any of them, because none is copyleft.
  If a vendor's only documentation of some behaviour turns out to be a GPL
  implementation, [CLEAN-ROOM.md](../../CLEAN-ROOM.md) applies and a spec goes in
  `specs/` before any code is written.

Two exclusions, recorded so they are not re-litigated later:

- **x264 is GPL**, so it is never a dependency. A software-encode fallback, if
  it ever ships, has to be an openly licensed encoder or nothing.
- **PyNvVideoCodec**, NVIDIA's own Python bindings, are MIT but the wheel bundles
  LGPLv3 FFmpeg (`libavformat`, `libavutil`) for its muxer and demuxer. It is a
  fine way to prototype and it is what proved the throughput numbers above, but
  it is not what ships: a hard dependency on it would put LGPL binaries in the
  install of every user who wanted to record a video. Our own binding and our own
  muxer avoid the question entirely.

Calling an external `ffmpeg` binary stays available to *users* as a pipe target,
which is a separate program and not a dependency of ours.

## Testing

- **ABI conformance.** The ctypes structs are checked against sizes and field
  offsets recorded from the C header, so a layout mistake fails a fast test
  rather than corrupting a struct at runtime. The recorded table is generated
  from the header with a compiler and checked in; the test needs no compiler.
- **Bitstream structure.** Encode a synthetic gradient, then parse the output:
  NAL boundaries, SPS/PPS present, dimensions in the SPS equal to what was
  requested, IDR at the frames where one was demanded, frame count as submitted.
- **Container structure.** Parse the written MP4 back into its box tree and
  check the sample table against the packets that went in.
- **Round trip.** `libnvcuvid` is present, so a later test can decode the stream
  and compare against the source frames within a tolerance — the only check that
  proves the picture is the right way up and the colours have not been swapped.
  Until then, one decoded reference frame is checked in as a fixture.
- **Hardware gating.** Every backend test skips cleanly when its hardware is
  absent, following the existing OpenGLContext GL-test conventions. AMD and
  Intel backends stay marked unverified in this document until they have run on
  real parts.
- **A machine with no graphics driver at all.** EGL ships with the driver, so
  a machine with none has no `libEGL` and PyOpenGL's EGL bindings do not import
  — a case that stands in for an EGL entry point has nothing to stand in for.
  Those cases take the `egl` fixture and skip; what the package does on such a
  machine is `tests/test_dmabuf.py::TestWithNoEGLLibraryAtAll`, which needs no
  EGL to run. Everything here reaches EGL through `dmabuf._egl`, so the absence
  arrives as "this context cannot export" rather than as an exception out of an
  import. To reproduce it on a machine that does have EGL: a `-p` plugin that
  sets `OpenGL.platform.PLATFORM.EGL` to None before collection.
- **The runner installs a software EGL** — `libegl1` and `libgl1-mesa-dri`, in
  `.github/workflows/test.yml` — so the layer above is exercised there rather
  than skipped wholesale. It is five cases, the two that matter being the
  modifier attributes an import is built from, which is otherwise checked only
  on a machine with a GPU. It does not buy a context: GLFW needs a display
  server, so the export and round-trip cases still skip with "GLFW will not
  initialise here". Xvfb is the next step for anyone who wants those, and has
  not been tried.

## Phases

| Phase | Content | Verified on |
| --- | --- | --- |
| 1a | `pyopengl-video`: `Encoder` interface and registry, NVENC backend on the GL texture path, MP4 muxer with composition offsets, ABI conformance harness, worked example | **landed**, NVIDIA/Linux |
| 1b | OpenGLContext: `VideoRecorder` over the texture ring and the flipping blit, fixed-step recording clock | **landed**, NVIDIA/Linux |
| W1 | `windows/`: DXGI adapter identity, D3D11 device and textures, `WGL_NV_DX_interop2` shared textures | in progress, this machine |
| W2 | `Encoder.new_input()` and `for_drawing()`, and `OpenGLContext`'s recorder following | in progress |
| W3 | oneVPL backend: Intel on Windows, D3D11 video memory, RGB straight in | in progress, this machine |
| W4 | NVENC on the DirectX device type, reusing the shim | next; needs a GL context on the NVIDIA adapter |
| 2 | PBO readback tier, so every driver with an encoder can record, and the reference the zero-copy paths are compared against | after W4 |
| 3 | `linux/dmabuf.py` EGL export, VA-API backend with its own H.264 control layer | **landed**, AMD/Linux; Intel/Linux is the same code and untested |
| 4 | AMF for AMD, HEVC and AV1 where the part has them, audio track | needs hardware |
| 5 | Decode into an OpenGL texture -- the reverse of everything above | low priority; see *Decoding* |

Phases W1-W3 make the Windows machine record without an NVIDIA part in the path
at all, which is what makes this a library for everyone rather than for NVIDIA
owners; phase 3 does the same for Linux.

## Decoding: a compressed stream into an OpenGL texture

**Low priority.** The library is named for what it does now, which is encode,
and nothing waiting on it needs the reverse. It is written down because the
shape is already here and the cost of adding it later grows if the interface
forgets about it.

The capability is a video file or stream in, and an OpenGL texture out, with the
frame never leaving the GPU -- the same claim the encoder makes, in the other
direction. What it would be for: video textures in a scene, a cut-scene player,
a replay viewer that scrubs, and the round-trip test this suite currently
cannot write without an outside decoder.

Every mechanism it needs already exists in this package, reversed:

- **libva** decodes through `VAEntrypointVLD`, and `vaExportSurfaceHandle`
  hands the decoded surface out as a DMA-BUF with real modifiers.
  `pyopengl_video.linux.dmabuf.import_texture` already turns one of those into
  a texture, and is written for this as much as for the encoder.
- **NVDEC** (`libnvcuvid`) is NVIDIA's side, and has no OpenGL device type: the
  decoded frame arrives in CUDA memory and reaches GL through
  `cuGraphicsGLRegisterImage`, which is a CUDA dependency the encoder path
  deliberately avoids. That asymmetry is the main reason this is not a small
  job.
- **The NV12 to RGB pass** is the VPP pass the encoder runs, backwards, or a
  shader sampling the two planes as separate textures -- which is the cheaper
  and more portable of the two, and the one a renderer would want anyway
  because it can convert straight into whatever the scene needs.
- **Demuxing** is what the encoder's muxer does backwards, and `mp4.py` would
  grow a reader. That is the piece with the least to learn and the most tedium
  in it: a container is a large specification, and only a small part of it is
  needed to find the samples.

What to decide before starting: whether a `Decoder` interface sits beside
`Encoder` in `encoder.py` or in a module of its own; whether the decoded frame
is handed over as a texture the decoder owns and recycles (matching
`InputHandle`) or as one the caller supplies; and whether seeking is in scope at
all, since a decoder that only plays forward is a great deal simpler than one
that scrubs.

## Before the first release

`0.2.0a1` is built and green on AMD/Linux and is not to be published until
NVIDIA on Linux and Intel on Windows have both run against it. `main` is the
release branch and `release.yml` triggers on nothing else, so the merge from
`develop` is the decision and nothing goes out before it.

Three changes in this version are shared by every backend and can only be
checked on hardware this machine does not have. They are what the runs are for;
the rest of each suite is its own gate.

**The muxer writes the parameter sets the stream carries**, not the ones the
encoder advertised before it had coded anything. On AMD this was a fix — the
driver amends them, and the sample description described a stream that did not
exist. NVENC and oneVPL are affected wherever their streams repeat the sets
in-band, which is a configuration NVENC has. So on each machine: record, and
decode the `.mp4` with an outside decoder, requiring silence. That check is what
found the bug, and `tests/test_vaapi_encode.py::test_a_muxed_recording_decodes_without_complaint`
runs it wherever `ffmpeg` is on the path.

**The test fixture and the examples ask for an EGL context on Linux.** GLFW on
Wayland already gave one, which is what every NVENC measurement in this document
was taken on, so this is only a change for an X11 session. It needs a run to be
a fact rather than an argument: `nvEncOpenEncodeSessionEx` with the OpenGL
device type must still open under EGL on X11.

**`Backend` grew an optional `explain`.** Neither the NVENC nor the oneVPL
record sets one, so both keep their existing messages; what to confirm is that
`open_encoder` still says something useful on a machine where the backend
declines — a Windows box with no Intel adapter is the case worth reading.

Two more that are not about the code:

- **`pytest` is invoked as a bare command by tox and CI**, which it was not
  before. It works on Linux; a Windows run is what says the `pythonpath`
  setting does the same there.
- **A pending trusted publisher** has to exist on PyPI against the name before
  the first push to `main`. `pyopengl-video` is unregistered, so there is no
  project to attach an ordinary publisher to yet.

### What the NVIDIA/Linux run answered

An RTX 3060 Ti under driver 580.173.02, Wayland, GLFW's EGL context. The suite
is green and the three shared changes come out as follows.

**The muxer's parameter sets are a no-op here, and now have a gate.** NVENC
repeats the sets in-band and what the stream carries is byte-identical to what
`headers()` advertised, so the sample description is the same either way. That
is a measurement rather than a property of the interface, so the checks that
make it one live in `tests/test_nvenc_encode.py::TestMuxing`: a recording, a
reordered recording, and `ffmpeg` decoding both in silence. `decode_errors` is
in `tests/conftest.py`, where either backend's suite can reach it.

**EGL on X11 could not be answered on this machine, and the attempt found
something else.** There is an X server here, but no EGL display on it that the
NVIDIA driver will open: an EGL context on X11 lands on llvmpipe, and forcing
the NVIDIA vendor library refuses the display outright. So the question stands
for a machine with a real X session. What the attempt did establish is the
matrix around it — Wayland/EGL, Wayland/GLX and X11/GLX all open the session
and encode — and that NVENC answers a context on the wrong GPU with the same
`NV_ENC_ERR_UNSUPPORTED_DEVICE` it answers a missing context with. The message
told the reader a context was not current while one was. It now names the
renderer it found.

**`open_encoder` explains a decline here.** With NVENC absent the message
carries the VA-API backend's own reason, which names the driver packages to
install; with it present, a size or codec past what it does lists the backend
with its codecs and its maximum. Both are messages a caller can act on.

The run also crossed the DMA-BUF path, which is the VA-API backend's and not
NVENC's, and found that the two drivers disagree about layouts — see
[VAAPI-BACKEND.md](VAAPI-BACKEND.md). `import_texture` now answers a refusal
from either of the two calls with one `DMABufError`, and gives back the image
and the texture it made rather than leaking them.

The run also reported that a caller catching `EncoderError` around a VA-API
recording did not catch a texture that would not export. `VAAPIEncoder`
translates `DMABufError` at its own boundary now, so one `except EncoderError`
covers a recording on every backend; `tests/test_errors.py` holds all three to
that without hardware.

## Open questions

- **Distribution name — settled.** `pyopengl-video` is unregistered on PyPI, as
  is `pyopengl_video`, so the name the package already carries is the name it
  publishes under. Nothing has been uploaded yet: the first push to `main` with
  a version PyPI does not have is what cuts the release, so the project needs a
  *pending* trusted publisher created against the name before that happens.
- **How OpenGLContext depends on it.** An optional extra
  (`openglcontext[video]`) keeps the install of anyone who never records a video
  unchanged, at the cost of a `--record` flag that can fail at runtime with a
  "not installed" message. That is the same shape as the existing optional
  dependencies and is the default unless recording becomes core enough to
  justify a hard requirement.
- **Ring depth** is currently a guess (three or four). It should be derived from
  the encoder's reported queue depth once the deferred-output behaviour is
  measured.
- **HDR.** The engine renders in HDR before tonemapping; encoding the tonemapped
  LDR result is the obvious first target, but HEVC Main10 from the pre-tonemap
  buffer is the more interesting one, and the interface should not make it hard.
