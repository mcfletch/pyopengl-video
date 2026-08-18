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
