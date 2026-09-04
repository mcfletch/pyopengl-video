# Third-party notices

`pyopengl-video` links against video encoder libraries that ship with the
graphics driver. It contains no third-party code.

## NVIDIA NvEncodeAPI

The ctypes declarations in `pyopengl_video/nvenc/api.py` describe the ABI of
`libnvidia-encode.so.1`, which is part of the NVIDIA display driver. The
structure layouts, enumerations and interface GUIDs are taken from
`nvEncodeAPI.h` as published in
[nv-codec-headers](https://github.com/FFmpeg/nv-codec-headers), header version
13.1, which carries the following notice:

> Copyright (c) 2010-2026 NVIDIA Corporation
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the software, and to permit persons to whom the software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## Intel oneVPL

The ctypes declarations in `pyopengl_video/vpl/api.py` describe the ABI of
`libvpl.dll` and the Media SDK runtime it dispatches to, both part of the Intel
graphics driver. The structure layouts, their packing and the enumerations are
taken from the oneVPL headers as published in
[intel/libvpl](https://github.com/intel/libvpl), which carry the following
notice:

> Copyright (C) Intel Corporation
>
> SPDX-License-Identifier: MIT
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## Microsoft Direct3D 11, DXGI, and the OpenGL registry

`pyopengl_video/windows/` calls documented Windows interfaces (D3D11, DXGI) and
the `WGL_NV_DX_interop2` extension as specified in the Khronos OpenGL registry.
No code from any vendor sample or SDK is reproduced.

## libva

The ctypes declarations in `pyopengl_video/vaapi/api.py` describe the ABI of
`libva.so.2` and `libva-drm.so.2`, which are the Video Acceleration API runtime
on Linux. The structure layouts, the enumerations and the constant values are
taken from the libva headers as published in
[intel/libva](https://github.com/intel/libva) — `va.h`, `va_vpp.h`,
`va_enc_h264.h`, `va_drmcommon.h` and `va_drm.h` — and are recorded by machine
in `tests/va_abi.json` rather than vendored. The headers carry the following
notice:

> Copyright (c) 2007 Intel Corporation. All Rights Reserved.
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES
> OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
> ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
> DEALINGS IN THE SOFTWARE.

## ITU-T H.264 and the Khronos EGL registry

`pyopengl_video/vaapi/h264.py` writes the sequence parameter set, picture
parameter set and slice headers of an H.264 stream. The syntax it follows is
that of the ITU-T H.264 specification; the field names in the code are the
specification's own, so the two can be read side by side. No encoder
implementation was consulted.

`pyopengl_video/linux/dmabuf.py` calls `EGL_MESA_image_dma_buf_export` and
`EGL_EXT_image_dma_buf_import` as specified in the Khronos EGL registry. Mesa is
the implementation underneath on AMD, and its behaviour where the specification
is silent — that an exported texture reports `DRM_FORMAT_MOD_INVALID`, and that
such a buffer still imports correctly — was established by observation rather
than by reading its source.
