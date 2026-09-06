# The VA-API backend: Intel and AMD

**Status:** Landed on AMD. The backend records end to end on a Radeon 8060S
(`gfx1151`, Mesa 25.2.8 `radeonsi`): colour, orientation and timing are checked
against the surface the encoder is handed, and the muxed result decodes without
complaint. Intel is the same code against the `iHD` driver and has not been run.
**Covers:** phase 3 of [GPU-VIDEO-ENCODE.md](GPU-VIDEO-ENCODE.md).

## What building it settled

Five things came out of the work that were not knowable from the interface, and
that the Intel bring-up will meet in the same order.

**The driver need not write the slice header, and Mesa does not.** libva's
documentation describes packed headers as an option a driver may ask for.
Mesa's H.264 encoder writes the coded macroblocks and *nothing in front of
them*: no NAL header, no slice header. Without `VAEncPackedHeaderSlice` supplied
every picture, the stream carries a zero byte where the NAL header belongs, no
player finds a picture, and no call fails. So the control layer here writes
slice headers as well as parameter sets, and `h264.py` is correspondingly
larger than the plan below assumed.

**Ask only for the packed headers you write.** `VAConfigAttribEncPackedHeaders`
reports every kind the driver *can* take. Requesting that mask is a promise to
supply all of them, and a driver told to expect a packed slice header stops
writing its own — which is the failure above, arrived at by trying to be
accommodating. Request exactly what the backend produces.

**The parameter sets in the stream are not the ones handed over.** Mesa amends
what it is given: it clears `transform_8x8_mode_flag` in the picture parameter
set, because its encoder writes no per-macroblock transform flag, and a stream
claiming otherwise is one a decoder mis-parses. So the sets that describe the
stream are the ones the *stream* carries, and both `Encoder.headers()` and the
MP4 muxer take them from there rather than from what was advertised before
anything was coded.

**One reference frame is what the hardware offers and what this needs.**
`VAConfigAttribEncMaxRefFrames` reports 1 for reference list 0 on this part.
Asking for more is refused rather than quietly reduced, so the count is clamped
to what the driver states — and a stream of I and P pictures needs exactly one.

**A texture exported with no modifier still imports correctly.** Mesa reports
`DRM_FORMAT_MOD_INVALID` for an exported `GL_RGBA8` texture, which reads as a
warning that the layout cannot be described. Importing it anyway produces the
right picture: the export gives the buffer a layout the importing driver
resolves. The modifier is still carried across wherever the driver names one,
because a tiled surface imported as linear reads as noise and nothing reports
it.

**A driver that names the layout requires it back.** NVIDIA's proprietary
driver reports `0x0300000000E08013` — a block-linear layout — for the same
export, and takes it back only when the import names it. An import that leaves
the modifier out, or claims linear, is refused, which is correct: the pixels
are not where either description says. So the modifier is not a hint to pass on
where convenient but the half of the description that says how to read the
memory, and only a driver that would not name a layout gives an import the
freedom to omit one.

Which of the two calls does the checking is also the driver's choice. Mesa
answers at `eglCreateImageKHR`; NVIDIA hands back an image and refuses at
`glEGLImageTargetTexture2DOES`, so `import_texture` turns a failure at either
into one `DMABufError` and gives the image and the texture back before it
raises.

## What is not done

- **Intel.** The same code against `iHD`. The packed-header requirements are
  exactly where the two drivers are most likely to differ, so the bring-up is
  testing rather than implementation — as this plan predicted.
- **B-frames.** `bframes` above zero is refused by name. Pictures are coded in
  display order, so nothing reorders and a container needs no composition
  offsets; adding them means picture order counts for a reordered GOP and a
  second reference list, and is the natural next piece of `h264.py`.
- **HEVC and AV1.** Both are advertised by this part's `EncSlice`. Each needs
  its own parameter sets and slice headers, which is most of what `h264.py` is.

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

**One control layer, both vendors.** Intel has a newer interface, oneVPL
(`libvpl.so.2`, over VA-API underneath), which does the parameter sets, the GOP
and the DPB itself the way NVENC does, and it would avoid the control layer
entirely — for Intel. It buys nothing here, because the control layer has to be
written for AMD in any case, and once it exists Intel costs no more code than
choosing a different driver. Splitting the vendors across two encoder interfaces
would mean two control paths to maintain, two sets of bugs, and a difference in
behaviour between vendors that callers would eventually notice.

So: **libva for both**, and the marginal cost of supporting Intel is testing
rather than implementation.

What would reopen the question is an Intel-specific capability that libva does
not reach — a rate control mode, a lookahead, or a codec that the media driver
exposes only through oneVPL. That is a reason to add a second Intel backend
later, beside this one, not a reason to start with two.

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
