# Windows

**Status:** Unsupported, and the reason is a design decision of NVIDIA's rather
than a gap in this package.

## The short answer

The zero-copy path this package is built on does not exist on Windows. NVIDIA's
header states it plainly: `NV_ENC_DEVICE_TYPE_OPENGL` is *"supported only on
Linux"*. There is no way to open an NVENC session against an OpenGL context on
Windows, so a texture cannot be handed over by name there.

`pyopengl_video.nvenc.probe()` therefore reports the backend unavailable off
Linux, and `encoders()` returns an empty list rather than an encoder that would
fail when a session is opened.

## What would work unchanged

Rather more than the headline suggests, which is why the ground is worth
recording.

| Part | On Windows |
| --- | --- |
| `mp4.py` | Works. Pure Python, no platform in it. |
| `encoder.py` — interface, registry, `Packet` | Works. |
| `nvenc/api.py` — structures, GUIDs, function table | Expected to work; unverified. The library is `nvEncodeAPI64.dll`, which `library_name()` already returns, and the interface is `__stdcall`, which `FUNCTYPE` already accounts for. The two are the same ABI on 64-bit Windows and differ on 32-bit. |
| `nvenc/encoder.py` — session, registration, encode loop | The encode loop is fine; **opening the session and registering a texture are not**, because both name the OpenGL device type. |

So what is missing is one step: getting the frame from an OpenGL texture into
something NVENC will accept on Windows. Everything on either side of that step
is written.

## The three ways across

### Read the frame back through a Pixel Buffer Object

The portable tier from [GPU-VIDEO-ENCODE.md](GPU-VIDEO-ENCODE.md): a fenced ring
of Pixel Buffer Objects, an asynchronous `glReadPixels`, and NVENC's
host-memory input. The frame crosses the bus and comes back, which costs real
bandwidth at high resolutions, and the encode is still done in hardware.

This is the **cheapest route to a working Windows recording** and it is worth
building for its own sake — it is the fallback for any driver with an encoder
and no interop, and it is the reference the zero-copy paths are checked against.
It needs no new vendor interface, and it makes Windows work for AMD and Intel
too, through their own host-memory paths.

### Share the texture with Direct3D 11

`WGL_NV_DX_interop2` registers a D3D11 texture with OpenGL, so the renderer can
blit into a surface that is simultaneously a D3D11 resource; NVENC then takes it
through `NV_ENC_DEVICE_TYPE_DIRECTX`. AMD's AMF and Intel's oneVPL both take
D3D11 textures as well, so **one interop mechanism serves all three vendors on
Windows** — which makes it the strategically right answer, and the same shape as
the DMA-BUF export that serves Intel and AMD on Linux.

The cost is a D3D11 device to create and manage from Python, a second set of
bindings, and a WGL extension whose availability has to be probed.

### Go through CUDA

`cuGraphicsGLRegisterImage` on the texture, then `NV_ENC_DEVICE_TYPE_CUDA` with
a device pointer. This works on Windows *and* Linux, and would replace the
OpenGL device type everywhere with one mechanism. It costs a CUDA dependency,
which is a large thing to require of someone who wants to record a video, and it
is NVIDIA-only — so it buys less than the D3D11 route for more.

## Recommendation

If Windows becomes a goal, build the PBO tier first: it is portable, it is
wanted anyway, and it makes every vendor work on every platform at the cost of
bandwidth. Add the D3D11 interop path afterwards for the machines where the
bandwidth matters, because that one mechanism reaches NVENC, AMF and oneVPL
alike.

Until one of those is written, the honest statement — the one in the README and
in `docs/usage.md` — is that this package records on Linux.

## macOS

Out of scope. There is no NVENC and no VA-API; the encoder is VideoToolbox,
reached from OpenGL through an IOSurface, and Apple's OpenGL is deprecated and
capped at 4.1. A recorder there is a different design, not a backend of this one.
