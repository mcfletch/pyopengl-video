"""What Linux backends share: exporting an OpenGL texture as a DMA-BUF.

A DMA-BUF file descriptor is how a buffer moves between drivers on Linux
without being copied. Every hardware encoder here takes one -- libva imports it
as a surface, and the V4L2 memory-to-memory encoders on ARM parts take it as a
buffer -- so the export belongs beside the backends rather than inside any one
of them, the way :mod:`pyopengl_video.windows.interop` serves every Windows
backend.

:mod:`pyopengl_video.linux.dmabuf` is that export.
"""
