# Third-party notices

`pyopengl-video` links against video encoder and decoder libraries that ship
with the graphics driver. It contains no third-party code.

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

## NVIDIA NVDEC and the CUDA driver API

The ctypes declarations in `pyopengl_video/nvdec/api.py` describe the ABI of
`libnvcuvid.so.1` and `libcuda.so.1` (`nvcuvid.dll` and `nvcuda.dll`), both
part of the NVIDIA display driver. The structure layouts, the enumerations and
the exported symbol names are taken from `dynlink_cuviddec.h`,
`dynlink_nvcuvid.h`, `dynlink_cuda.h` and `dynlink_loader.h` as published in
[nv-codec-headers](https://github.com/FFmpeg/nv-codec-headers), header version
13.1, and are recorded by machine in `tests/nvdec_abi.json` rather than
vendored. The four headers carry the MIT notice quoted above for
`nvEncodeAPI.h`: `dynlink_cuviddec.h` and `dynlink_nvcuvid.h` under
"Copyright (c) 2010-2026 NVIDIA Corporation", and `dynlink_cuda.h` and
`dynlink_loader.h` under "Copyright (c) 2016".

The order of the calls and what each callback is for are as those headers
document them. That chroma follows luma by the decoder's target height in
rows, and that the parser reuses a picture once its display callback has
returned, were established by decoding: every picture of the streams under
`tests/data/decode/` comes out with the bytes `ffmpeg`'s software decoder
recorded. No decoder's source was consulted, FFmpeg's `cuviddec.c` included.

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
[intel/libva](https://github.com/intel/libva), decode and encode alike: `va.h`, `va_vpp.h`,
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

## ITU-T H.264, ISO/IEC 14496 and the Khronos EGL registry

`pyopengl_video/vaapi/h264.py` writes the sequence parameter set, picture
parameter set and slice headers of an H.264 stream. The syntax it follows is
that of the ITU-T H.264 specification; the field names in the code are the
specification's own, so the two can be read side by side. No encoder
implementation was consulted.

`pyopengl_video/vaapi/h264_parse.py` reads the same syntax.
`pyopengl_video/vaapi/h264_dpb.py` implements picture order count (8.2.1),
reference marking (8.2.5), reference picture lists (8.2.4) and output order
(annex C.4) from Rec. ITU-T H.264 alone. No decoder's source was consulted,
FFmpeg's and GStreamer's included. libva takes scaling lists in raster order,
as its header states; decoding confirmed it.

`pyopengl_video/mp4.py` reads and writes the ISO base media file format of
ISO/IEC 14496-12, with the `avcC` record of ISO/IEC 14496-15 and the `esds`
descriptors of ISO/IEC 14496-1. `pyopengl_video/colour.py` uses the code points
and coefficients of Rec. ITU-T H.273.

`tools/make_decode_fixtures.py` runs `ffmpeg` with `libx264` on `ffmpeg`'s
test pattern to write the streams under `tests/data/decode/`, the interlaced
one included. It also records
the checksum of each picture `ffmpeg` decodes. The streams and checksums are
test data. `ffmpeg` and `x264` run as programs, and no part of either is in
this package.

`pyopengl_video/linux/dmabuf.py` calls `EGL_MESA_image_dma_buf_export` and
`EGL_EXT_image_dma_buf_import` as specified in the Khronos EGL registry. Mesa is
the implementation underneath on AMD, and its behaviour where the specification
is silent — that an exported texture reports `DRM_FORMAT_MOD_INVALID`, and that
such a buffer still imports correctly — was established by observation rather
than by reading its source.
