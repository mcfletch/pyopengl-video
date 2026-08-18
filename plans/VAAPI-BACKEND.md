# The VA-API backend: Intel and AMD

**Status:** Planned. Nothing here has run — the development machine has an
NVIDIA card, and this backend needs Intel or AMD hardware to write against.
**Covers:** phase 3 of [GPU-VIDEO-ENCODE.md](GPU-VIDEO-ENCODE.md).

One backend serves both vendors. `libva` is the encoder interface on Linux for
Intel parts through the `iHD` media driver and for AMD parts through Mesa's
`radeonsi`, so the difference between them is which driver `libva` loads, not
what this code does.

## What it is not: how libva differs from NVENC

This matters more than any single call, so it comes first.

**NVENC is a frame-level interface.** Hand it a picture and a target bitrate and
the driver decides picture types, manages reference frames, writes the parameter
sets and hands back a compressed frame. The backend that exists is small because
of that.

**libva is a slice-level interface, and the caller is the control layer.** For
H.264 encoding the application:

- fills in the sequence, picture and slice parameters itself, which means
  deciding `frame_num`, picture order counts, reference list contents and slice
  types per picture;
- **builds the packed SPS and PPS by hand** — as bitstreams, start code and NAL
  header included — whenever the driver reports the packed-header attributes
  through `vaGetConfigAttributes`, which the common drivers do;
- manages the decoded picture buffer: which surfaces are reference frames, which
  are free, and what each picture predicts from.

So this backend contains an H.264 *encoder control layer*: a bitstream writer
for the parameter sets, a GOP structure, and DPB bookkeeping. That is the bulk
of the work, and none of it is shared with the NVENC backend. Budget for it
accordingly, and keep it in its own module — `vaapi/h264.py` — so the libva
plumbing and the codec logic are not tangled.

**oneVPL is the way out of that for Intel.** Intel's newer interface
(`libvpl.so.2`, the runtime is `libmfx-gen` over VA-API) does the parameter
sets, the GOP and the DPB itself, the way NVENC does, and takes
`VASurfaceID`s as input. If the control layer proves the expensive part, the
sequence to consider is: write the surface-import half against libva, then reach
the encoder through oneVPL on Intel and through libva on AMD, sharing the import
and the interface but not the control layer.

## The frame's journey

```
GL texture (RGBA8)
  -> EGLImage                              eglCreateImage
  -> DMA-BUF fd + modifier/offset/stride   eglExportDMABUFImageMESA
  -> VASurfaceID (RGBA)                    vaCreateSurfaces, DRM_PRIME_2 import
  -> VASurfaceID (NV12)                    VPP: VAEntrypointVideoProc
  -> coded buffer                          VAEntrypointEncSlice
  -> Annex-B packets                       vaMapBuffer
```

Nothing crosses the bus. The conversion to NV12 is a pass on the GPU's fixed
function blocks, needed because VA-API encoders take NV12 and the renderer
produces RGBA.

### 1. The context must be EGL

`EGL_MESA_image_dma_buf_export` is an EGL extension, so the OpenGL context has
to be an EGL one. GLFW creates a GLX context by default on X11 and has to be
told otherwise:

```python
glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
```

This is the first thing to check when the export fails, and a good thing for
`probe()` to notice: `eglQueryString(display, EGL_EXTENSIONS)` must list
`EGL_MESA_image_dma_buf_export`, and the current context must be an EGL context.
PyOpenGL already binds what is needed, in
`OpenGL.EGL.MESA.image_dma_buf_export` and `OpenGL.EGL.KHR.image_base`.

### 2. Export the texture

```python
image = eglCreateImageKHR(display, context, EGL_GL_TEXTURE_2D_KHR,
                          ctypes.c_void_p(texture), None)
eglExportDMABUFImageQueryMESA(display, image, fourcc, num_planes, modifiers)
eglExportDMABUFImageMESA(display, image, fds, strides, offsets)
```

Query before export: the plane count and the modifier decide how the surface is
described on the other side, and a tiled or compressed layout has a modifier
that must be carried across rather than assumed linear. Keep the `EGLImage`
alive as long as the surface that was made from it, and close the file
descriptors when the surface goes.

### 3. Import as a VA surface

`vaCreateSurfaces` with `VASurfaceAttribMemoryType` set to
`VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2` and
`VASurfaceAttribExternalBufferDescriptor` pointing at a
`VADRMPRIMESurfaceDescriptor`: fourcc, size, one object per file descriptor with
its size and modifier, one layer per plane with its format, object index, offset
and pitch.

Both drivers open the device the same way — `open('/dev/dri/renderD128')`,
`vaGetDisplayDRM(fd)`, `vaInitialize` — which is what makes this a headless path
as well as a windowed one.

### 4. Convert and encode

A VPP context (`VAProfileNone`, `VAEntrypointVideoProc`) converts the imported
RGBA surface into an NV12 surface the driver allocated, through a
`VAProcPipelineParameterBuffer` between `vaBeginPicture` and `vaEndPicture`.

The encode context (`VAProfileH264Main` or `High`, `VAEntrypointEncSlice`) then
takes, per picture: the sequence parameters once and on every IDR, the picture
parameters, the slice parameters, the packed SPS/PPS when required, and any
misc parameter buffers for rate control, frame rate and HRD. `vaMapBuffer` on
the coded buffer gives the compressed bytes, with a status field that says
whether the picture was coded at all.

## Fitting the existing interface

Nothing about the shape above conflicts with `Encoder`, which is the point of
having written the interface first:

- **`register(texture)`** does the export and the import, and returns a handle
  holding the `EGLImage`, the file descriptors, the RGBA surface and the NV12
  surface it converts into. Expensive, done once, exactly as the interface
  intends.
- **`input_slots`** is the GOP's reordering depth plus one, from the structure
  this backend chose rather than from one the driver picked.
- **`encode()`** runs the VPP pass, submits the picture, and returns packets.
  When the control layer reorders, it returns them in decode order carrying
  display timestamps and sets `reorders_frames`, and `MP4Writer` needs no
  changes.
- **Colour** is declared the same way: limited-range BT.709, written into the
  VUI of the SPS this backend builds. The VPP conversion must be told to match
  what the VUI says, which is a place the two can silently disagree — see the
  testing note below.
- **Orientation** must match: first row of the texture is the top of the
  picture. The export preserves the texture's layout, so it should follow, and
  a decode of a known frame is what proves it.

## probe()

Cheap, quiet, and honest about all four things that have to be true: `libva.so.2`
loads, a DRM render node opens, the EGL display advertises
`EGL_MESA_image_dma_buf_export`, and `vaQueryConfigEntrypoints` reports
`VAEntrypointEncSlice` for an H.264 profile. The last one is the only reliable
answer to "does this machine have an encoder", because `libva` is present on
plenty of machines whose GPU has none.

## Testing

The suite's conventions carry over, and `tests/test_nvenc_encode.py` is the
pattern to copy: packet counts, key frame positions, timestamps in and out,
content sensitivity at a constant quantiser, and reordering behaviour.

Three checks matter more here than they did for NVENC, because this backend
writes the bitstream headers itself:

- **Parse back what was written.** The SPS this backend builds must say the
  right resolution, level, frame rate and colour description. A parser in the
  tests reading its own encoder's SPS catches a bitstream writer that packs a
  field one bit off, which no player will report clearly.
- **Check colour end to end.** Encode known colours, decode, and compare against
  the BT.709 limited-range values. The VPP conversion and the VUI declaration
  are set in different places and nothing but this notices when they part.
- **Check both drivers.** Intel `iHD` and Mesa `radeonsi` are different
  implementations of the same interface, and the packed-header and DPB
  requirements are exactly where they differ. A backend tested on one is a
  backend untested on the other.

Hardware is the blocker: CI needs an Intel machine and an AMD machine, or the
tests skip and the backend stays unverified, which the README's support table
must keep saying until it is not true.

## Licensing

`libva` is MIT and `oneVPL` is MIT, so both may be read and bound directly, and
no clean-room procedure applies. Mesa's VA-API driver is the implementation
underneath on AMD; its source is MIT as well, but there is no need to read it —
the interface is documented, and behaviour that is not can be observed. Record
in `NOTICES.md` where any ABI facts came from, as the NVENC binding does.
