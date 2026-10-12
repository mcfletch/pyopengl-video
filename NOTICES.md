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

The ctypes declarations in `pyopengl_video/vaapi/api.py`, `hevc_api.py`,
`av1_api.py` and `vp9_api.py` describe the ABI of `libva.so.2` and
`libva-drm.so.2`, which are the Video Acceleration API runtime on Linux. The
structure layouts, the enumerations and the constant values are taken from the
libva headers as published in [intel/libva](https://github.com/intel/libva),
decode and encode alike: `va.h`, `va_vpp.h`, `va_enc_h264.h`, `va_dec_hevc.h`,
`va_enc_hevc.h`, `va_dec_av1.h`, `va_enc_av1.h`, `va_dec_vp9.h`,
`va_drmcommon.h` and `va_drm.h`. They are recorded by machine in
`tests/va_abi.json` rather than vendored. The headers carry the following
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

`tools/make_decode_fixtures.py` runs `ffmpeg` with `libx264`, `libx265`,
`libaom-av1`, `libsvtav1` and `libvpx-vp9` on `ffmpeg`'s test pattern to write
the streams under `tests/data/decode/`, the interlaced H.264 one included. It
also records the checksum of each picture `ffmpeg` decodes. The streams and checksums are test data. `ffmpeg`
and the encoders run as programs, and no part of any of them is in this
package. The parsers' tests hold header values that `ffmpeg`'s `trace_headers`
bitstream filter printed for those streams, which is its output and not its
source.

## ITU-T H.265, the AV1 and VP9 specifications, and their container bindings

`pyopengl_video/vaapi/hevc_parse.py`, `hevc_dpb.py`, `hevc_decode.py`,
`hevc.py` and `hevc_encode.py` read and write HEVC from Rec. ITU-T H.265: the
syntax of clause 7, picture order count and reference picture sets (8.3), and
output order (annex C.5). The field names are the specification's.

`pyopengl_video/vaapi/av1_parse.py`, `av1_decode.py`, `av1.py`,
`av1_encode.py` and `pyopengl_video/obu.py` follow the AV1 Bitstream and
Decoding Process Specification of the Alliance for Open Media.
`pyopengl_video/vaapi/vp9_parse.py`, `vp9_tables.py` and `vp9_decode.py`
follow the VP9 Bitstream and Decoding Process Specification, whose
dequantisation tables `vp9_tables.py` holds as the facts they are.

`pyopengl_video/mp4.py` writes and reads the `hvcC` record of ISO/IEC
14496-15, the `av1C` record of the AV1 Codec ISO Media File Format Binding and
the `vpcC` record of the VP Codec ISO Media File Format Binding.
`pyopengl_video/ivf.py` reads and writes the IVF layout: a 32-byte header and
each frame behind its size and timestamp.

No decoder's or encoder's copyleft source was consulted for any of these,
FFmpeg's, GStreamer's and x265's included. Where a driver's behaviour is not
stated by libva's headers, Mesa's source (MIT) was read for it: which packed
headers its AV1 and HEVC encoders take and what they write themselves, and
how its encoder treats a colour conversion that precedes an encode.

`pyopengl_video/linux/dmabuf.py` calls `EGL_MESA_image_dma_buf_export` and
`EGL_EXT_image_dma_buf_import` as specified in the Khronos EGL registry. Mesa is
the implementation underneath on AMD, and its behaviour where the specification
is silent — that an exported texture reports `DRM_FORMAT_MOD_INVALID`, and that
such a buffer still imports correctly — was established by observation rather
than by reading its source.
