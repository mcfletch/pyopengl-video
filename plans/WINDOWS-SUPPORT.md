# Windows

**Status:** The interop shim and the Intel backend have landed: OpenGLContext
records a playable MP4 on this machine's Intel GPU, with the frame never leaving
it. NVIDIA on Windows is next, over the same shim. Every claim below is measured
rather than assumed, and the measurements are here.

**Goal:** the same thing Linux has — the renderer's finished frame reaching the
video encoder without leaving the GPU — reached by the mechanism Windows
provides for it.

## There is no framebuffer pointer to redirect, here or anywhere

A framebuffer object is a container of attachment points, not memory. The pixels
live in a texture or renderbuffer whose allocation is the driver's, typically
tiled and colour-compressed, with no address a program can hand to anything else.

The Linux path is not pointer redirection either. NVENC takes a GL *texture
name*; VA-API takes a *DMA-BUF file descriptor*. Both are handles the driver
resolves on its own side, detiling if it must. So the Windows question is which
handle both a GL driver and an encoder understand there, and the answer is an
`ID3D11Texture2D`. **The D3D11 shim is the analogue of the DMA-BUF export, not a
substitute for a missing one.**

Three ready-made Windows paths exist and none of them replaces it:

| Path | What it gives | Why it is not this |
| --- | --- | --- |
| Windows.Graphics.Capture, DXGI Desktop Duplication | a D3D11 texture of a window or the desktop | captures after composition and at the compositor's pace: no offscreen rendering, no size other than the window's, and frames that do not line up with a fixed-step recording clock |
| Media Foundation `IMFSinkWriter`, encoder MFTs | an encoder that accepts D3D11 textures | solves the encode half only; the frame still has to get from GL into D3D11 |
| NVIDIA NVFBC | frame capture at the display level | licence-gated off GeForce, and deprecated on Windows |

## Verified ground truth

Measured 2026-09-03 on the development laptop: Intel UHD 630 (driver
31.0.101.2114) and NVIDIA GeForce GTX 1650 with Max-Q (driver 32.0.15.7322),
Windows 10 Pro 19045, Python 3.13, PyOpenGL from this workspace.

| Fact | Result |
| --- | --- |
| `WGL_NV_DX_interop` and `WGL_NV_DX_interop2` in the WGL extension string | **present on Intel's driver** — the NV name is historical, not a vendor restriction |
| `GL_EXT_memory_object_win32`, `GL_EXT_semaphore_win32` | present |
| PyOpenGL bindings for both | already shipped; the entry points live in `OpenGL.WGL.NV.DX_interop`, and `DX_interop2` carries only the availability flag |
| DXGI adapter matching the GL context by `GL_DEVICE_LUID_EXT` | works; GL reports LUID `6334010000000000`, which is DXGI adapter 0 |
| Which adapter a GL context lands on | **Intel**, with the NVIDIA part as adapter 1 — see *Adapter identity* |
| D3D11 device on a chosen adapter, from ctypes | works, feature level 11_0 |
| `wglDXOpenDeviceNV`, `wglDXRegisterObjectNV`, lock/unlock | all succeed |
| A registered texture as a GL colour attachment | `GL_FRAMEBUFFER_COMPLETE`; clears and blits into it |
| Byte order through the interop | **faithful to the format, not swizzled**: red written by OpenGL arrives in the texture's red component. A `B8G8R8A8_UNORM` surface holds BGRA bytes and an `R8G8B8A8_UNORM` one holds RGBA, so each backend names the pixel format matching the surface it asked for |
| NVENC `NV_ENC_DEVICE_TYPE_DIRECTX`, through the existing `nvenc/api.py` unchanged | session opens, H.264 encoder initialises |
| NVENC `enableEncodeAsync = 0` on Windows | accepted, so the existing synchronous encode loop carries over |
| NVENC registering a D3D11 texture as `NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX` | succeeds for `R8G8B8A8_UNORM`+`ABGR`, `B8G8R8A8_UNORM`+`ARGB` and `B8G8R8A8_UNORM`+`ABGR` |
| Media Foundation video encoder MFTs on this machine | **none** — 17 video decoders, 1 video processor, 10 audio encoders, 0 video encoders |
| oneVPL `MFXInit(MFX_IMPL_HARDWARE_ANY \| MFX_IMPL_VIA_D3D11)` | succeeds; API 1.35, implementation `0x0302` |
| `MFXVideoCORE_QueryPlatform` | succeeds; Coffee Lake, device `0x3E9B` |
| QuickSync H.264 `MFXVideoENCODE_Init` from **video memory**, with a D3D11 device and a frame allocator set | **succeeds** for `NV12`, `RGB4` and `BGR4` |
| Throughput, 1280x720, render and flipping blit and encode and synchronise | 300 frames in 1.49 s — 201 fps, 4.97 ms a frame |

Two consequences shape everything below. **The encoder accepts RGB directly**,
as NVENC does, so no colour-conversion pass is needed on either vendor — the
hardware does it. And **Media Foundation cannot be relied on**: a machine with
working QuickSync hardware registered no video encoder MFT at all, so the
vendor libraries are the path and MF is at best an extra.

### The structures are packed, and it is not optional

`mfxFrameInfo` carries a `mfxU64` in one arm of its geometry union. Declared
with natural alignment it is 80 bytes; the header wraps it in
`MFX_PACK_BEGIN_USUAL_STRUCT()`, which is `pack(4)`, making it **68**. Every
call taking an `mfxVideoParam` fails with `MFX_ERR_INVALID_VIDEO_PARAM` on the
80-byte layout and succeeds on the 68-byte one. The packing differs per
structure and the header states which each gets:

| Structure | Directive | Pack | Size (x64) |
| --- | --- | --- | --- |
| `mfxFrameInfo`, `mfxInfoMFX`, `mfxInfoVPP`, `mfxFrameAllocRequest`, `mfxExtBuffer`, `mfxVersion` | `MFX_PACK_BEGIN_USUAL_STRUCT` | 4 | 68, 136, 168, 92, 8, 4 |
| `mfxVideoParam`, `mfxFrameAllocResponse`, `mfxEncodeCtrl` | `MFX_PACK_BEGIN_STRUCT_W_PTR` | 8 | 208, 32, — |
| `mfxFrameData`, `mfxFrameSurface1`, `mfxBitstream` | `MFX_PACK_BEGIN_STRUCT_W_L_TYPE` | 8 | — |

This is exactly the class of fault `tools/record_nvenc_abi.py` exists to catch,
and there is no compiler on this machine to run the equivalent for oneVPL. So
`tools/record_vpl_abi.py` is written to the same shape and run where a compiler
is — the Linux side of this work — producing `tests/vpl_abi.json`. Until then
the sizes above are asserted directly by `tests/test_vpl_abi.py`, and the
binding is held to them.

## Adapter identity is the organising fact

Zero-copy needs the encoder and the GL context on the **same adapter**. On a
laptop with switchable graphics they are not the same by default: the GL context
here lands on the Intel part while the NVIDIA part sits idle as DXGI adapter 1.

So discovery asks GL which adapter it is on, and offers the encoders that live
there:

1. Read `GL_DEVICE_LUID_EXT` from the current context.
2. Enumerate DXGI adapters and match the LUID.
3. Build the D3D11 device on **that** adapter.
4. Offer the backends whose vendor owns it — NVENC for `0x10DE`, oneVPL for
   `0x8086`, AMF for `0x1002`.

**Choose the encoder to match the context, not the other way round.** On this
laptop the correct zero-copy answer is the Intel encoder, which is a hardware
H.264 encoder on the same die the renderer is already using. A cross-adapter
pairing is not zero-copy at all and is reported as such rather than silently
producing a copy.

Which GPU a GL context gets is a per-executable driver decision, and the
executable is `python.exe`. The `NvOptimusEnablement` exported symbol is not
available to us; the Windows per-app graphics preference
(`HKCU\Software\Microsoft\DirectX\UserGpuPreferences`) and the vendor control
panel are. A packaged game sets it for its own executable at install time. The
engine's part is to *report* the adapter it got, not to change the machine.

## The frame path

Identical in shape to Linux, with one extra scope:

```
render  ->  glBlitFramebuffer (flipping)  ->  interop texture  ->  encoder
```

The interop texture is one object seen two ways: an `ID3D11Texture2D` the
encoder reads, and a GL texture name the renderer draws into. `wglDXLockObjectsNV`
and `wglDXUnlockObjectsNV` say which side owns it, and the driver puts the
synchronisation in. Registered with `WGL_ACCESS_WRITE_DISCARD_NV`, because the
recorder overwrites the whole frame every time.

Per frame: lock, blit, unlock, encode. Locking is not a copy.

### Why this mechanism

`WGL_NV_DX_interop2` over the alternatives:

- **Over `GL_EXT_memory_object_win32`.** The external-objects route — a shared
  NT handle imported into GL, with explicit `EXT_semaphore_win32`
  synchronisation — is the more modern shape and is available here too. It costs
  explicit fences and D3D11 fence objects where lock/unlock costs a scope. It
  stays the upgrade if lock/unlock shows up in a profile, or where interop2 is
  missing.
- **Over CUDA.** `cuGraphicsGLRegisterImage` works on both platforms, but it is
  NVIDIA-only and makes a CUDA runtime a dependency of recording a video.
- **Over a PBO readback.** That tier is still wanted — see the phases — as the
  floor for drivers with no interop, for cross-adapter pairings, and as the
  reference the zero-copy output is compared against. It is not the fast path.

## Per-vendor, per-platform

The whole matrix, so a reader can see where any machine lands.

| Vendor | Linux | Windows |
| --- | --- | --- |
| **NVIDIA** | NVENC, `NV_ENC_DEVICE_TYPE_OPENGL`, texture by name. Landed. | NVENC, `NV_ENC_DEVICE_TYPE_DIRECTX`, D3D11 texture through the shim. The binding is verified; the encoder needs its session and registration split behind an input strategy. |
| **Intel** | VA-API (`iHD`) with a DMA-BUF imported from an `EGLImage`, or oneVPL over the same driver. Designed, needs hardware. | oneVPL (`libvpl.dll`) with a D3D11 texture, `MFX_IMPL_VIA_D3D11`. **Verified here; in progress.** |
| **AMD** | VA-API on Mesa's `radeonsi` — the same backend as Intel, a different driver. Designed, needs hardware. | AMF (`amfrt64.dll`), `AMF_MEMORY_DX11`, through the same shim. Designed, needs hardware; absent on this machine. |
| **Any** | PBO readback into whichever hardware encoder is present. | The same, plus Media Foundation where a video encoder MFT exists — which is not everywhere. |

**NVIDIA cannot use the VA-API path on Linux:** `nvidia-vaapi-driver` is a
decode-only shim over NVDEC, and NVIDIA's EGL does not implement
`EGL_MESA_image_dma_buf_export`. Two backends are genuinely required.

## What this changes in the interface

On Linux the caller makes a texture and hands it over. On Windows the texture
must be **created by the interop layer**, because it has to be a D3D11 resource
before it can be a GL one. So allocation becomes the encoder's job on both
platforms:

```python
handles = [encoder.new_input() for _ in range(encoder.input_slots)]

with handles[slot].for_drawing():        # GL owns the texture inside this scope
    copy_frame(handles[slot].framebuffer, size)
packets = encoder.encode(handles[slot], timestamp)
```

`new_input()` returns a handle carrying the GL texture name and a framebuffer
with it attached. `for_drawing()` is the lock/unlock scope, and does nothing on
backends that need none, so a recorder written once works on both platforms.
`register()` stays for a caller that already has a texture and a backend that
can take a foreign one; `Encoder.allocates_inputs` says which is which.

`OpenGLContext`'s `CaptureTarget` becomes a wrapper around a handle rather than
the owner of a texture, and `VideoRecorder.capture()` gains the `with` scope.
Nothing else in the recorder changes: the flipping blit, the ring, the fixed-step
clock and the muxer are all platform-independent already.

## Phases

| Phase | Content | State |
| --- | --- | --- |
| W1 | `windows/`: COM helpers, DXGI adapter identity, D3D11 device and textures, `WGL_NV_DX_interop2` shared textures | **landed** |
| W2 | `Encoder.new_input()` / `for_drawing()`, and `OpenGLContext` following | **landed** |
| W3 | oneVPL backend: Intel, D3D11 video memory, RGB in | **landed** — OpenGLContext records a playable MP4 on the Intel GPU here |
| W4 | NVENC DirectX device type reusing the shim | next, once a GL context can be put on the NVIDIA adapter |
| W5 | PBO readback tier — the floor, and the reference for the zero-copy output | after W4 |
| W6 | AMF for AMD; Media Foundation where an encoder MFT exists | needs hardware |

### What W3 left open

- **No recording has been decoded.** The picture is provably the right way up,
  provably varies with what was drawn, and the surface's byte order is checked
  against the format the encoder is told it holds; the stream also declares
  limited-range BT.709 the way the NVENC backend does, and the runtime reports
  that declaration back. What none of that shows is a red pixel coming back red
  through a decoder. The Linux side settled it with NVDEC; Windows has a Media
  Foundation H.264 *decoder* — the enumeration found one — which is the natural
  way to close it.
- **The ABI is asserted rather than recorded.** There is no C compiler on this
  machine, so `tests/test_vpl_abi.py` holds the binding to a table written by
  hand from the headers. `tools/record_vpl_abi.py` produces the real thing and
  is meant to be run on the Linux side, where it writes `tests/vpl_abi.json`
  and the test prefers it.
- **One picture at a time.** Each frame is synchronised before `encode()`
  returns, so `AsyncDepth` is 1 and the encoder is never pipelined. At 720p the
  whole loop costs 4.97 ms a frame, which carries a game at 60 fps with room
  over but is not the headroom this part has: the renderer waits on the encoder
  every frame instead of running alongside it. Taking it back means a queue of
  bitstream buffers and syncing a frame or two behind — the same shape the NVENC
  backend's deferred output already has, and the reason `encode()` returns a
  list. Worth doing before the bar is a heavier game at 1080p.

## Licensing

Everything used here is permissive. The oneVPL headers
([intel/libvpl](https://github.com/intel/libvpl)) are MIT and are read for ABI
facts exactly as `nvEncodeAPI.h` is, and cited in the binding rather than
vendored — see NOTICES.md. `WGL_NV_DX_interop2` is an OpenGL registry
specification. D3D11, DXGI and Media Foundation are documented Windows
interfaces. No vendor sample code is copied.

## macOS

Out of scope. There is no NVENC and no VA-API; the encoder is VideoToolbox,
reached from OpenGL through an IOSurface, and Apple's OpenGL is deprecated and
capped at 4.1. A recorder there is a different design, not a backend of this one.
