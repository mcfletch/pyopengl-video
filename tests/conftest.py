"""Fixtures for the tests that need a live OpenGL context or a real encoder.

The context is a hidden GLFW window: encoding reads a texture and never presents
anything, so there is nothing to show, and an unmapped window keeps a test run
from flashing over whatever the person running it is doing.
"""
import hashlib
import shutil
import subprocess
import sys

import numpy as np
import pytest
from OpenGL.GL import (
    GL_CLAMP_TO_EDGE,
    GL_COLOR_ATTACHMENT0,
    GL_COLOR_BUFFER_BIT,
    GL_DRAW_FRAMEBUFFER,
    GL_LINEAR,
    GL_NEAREST,
    GL_PACK_ALIGNMENT,
    GL_READ_FRAMEBUFFER,
    GL_RED,
    GL_RG,
    GL_RGBA,
    GL_RGBA8,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_UNSIGNED_BYTE,
    GL_UNSIGNED_SHORT,
    glBindFramebuffer,
    glBindTexture,
    glBlitFramebuffer,
    glDeleteFramebuffers,
    glDeleteTextures,
    glFramebufferTexture2D,
    glGenFramebuffers,
    glGenTextures,
    glGetTexImage,
    glPixelStorei,
    glTexImage2D,
    glTexParameteri,
    glTexSubImage2D,
)

from pyopengl_video import decoders, nvdec, nvenc, vaapi, vpl
from pyopengl_video.linux import dmabuf
from pyopengl_video.windows import interop


@pytest.fixture(scope='session')
def gl_context():
    """A current OpenGL context for the whole session, hidden from the display.

    An EGL context where the platform has one. Handing a texture to a Linux
    encoder means exporting it as a DMA-BUF, which is an EGL extension, and
    GLFW makes a GLX context by default on X11 -- so a suite that did not ask
    would skip every VA-API test on a machine that can run them.
    """
    glfw = pytest.importorskip('glfw')
    if not glfw.init():
        pytest.skip('GLFW will not initialise here')
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    if sys.platform.startswith('linux'):
        glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
    window = glfw.create_window(64, 64, 'pyopengl-video tests', None, None)
    if not window:
        glfw.terminate()
        pytest.skip('no OpenGL context available')
    glfw.make_context_current(window)
    yield window
    glfw.terminate()


@pytest.fixture
def egl():
    """PyOpenGL's EGL bindings, or a skip where this machine has no EGL.

    EGL ships with the graphics driver, and a machine with no driver installed
    -- a virtual machine, a container, most CI runners -- has no library for
    PyOpenGL to bind to, so importing the bindings raises. A case that stands
    in for an EGL entry point has nothing to stand in for there; what this
    package does on such a machine is covered by the cases that need no EGL at
    all, in ``TestWithNoEGLLibraryAtAll``.
    """
    module = dmabuf._egl()  # noqa: SLF001 - the bindings the module under test loads, which the fixture hands to stand-ins
    if module is None:
        pytest.skip('PyOpenGL has no EGL library to bind to on this machine, '
                    'so there is no EGL entry point to stand in for')
    return module


@pytest.fixture
def vaapi_available(gl_context):  # noqa: ARG001 - requested so a GL context is current first
    """Skip unless libva here can encode H.264 from this OpenGL context."""
    if not sys.platform.startswith('linux'):
        pytest.skip('libva is the Linux encoder interface')
    reason = dmabuf.unavailable_because()
    if reason:
        pytest.skip(reason)
    vaapi.forget_probe()
    if not vaapi.probe():
        pytest.skip('no VA-API H.264 encoder here: libva must be installed '
                    '(libva2 and a driver such as mesa-va-drivers or '
                    'intel-media-va-driver), and the GPU must have an encoder')
    return True


@pytest.fixture
def vaapi_decoder_available(gl_context):  # noqa: ARG001 - requested so a GL context is current first
    """Skip unless libva here can decode H.264 into this OpenGL context."""
    if not sys.platform.startswith('linux'):
        pytest.skip('libva is the Linux decoder interface')
    if not dmabuf.import_available():
        pytest.skip('this OpenGL context cannot import a DMA-BUF '
                    '(EGL_EXT_image_dma_buf_import)')
    vaapi.forget_probe()
    if not vaapi.decoder_probe():
        pytest.skip('no VA-API H.264 decoder here: libva must be installed with '
                    'a driver such as mesa-va-drivers or intel-media-va-driver')
    return True


@pytest.fixture
def nvdec_available(gl_context):  # noqa: ARG001 - requested so a GL context is current first
    """Skip unless NVDEC here can decode into this OpenGL context's textures."""
    nvdec.forget_probe()
    if not nvdec.probe():
        pytest.skip('no NVDEC decoder writes this OpenGL context\'s textures: '
                    + (nvdec.explain() or 'there is no NVIDIA driver here'))
    return True


@pytest.fixture
def decoder_available(gl_context):  # noqa: ARG001 - requested so a GL context is current first
    """Skip unless some backend here decodes H.264 into this OpenGL context."""
    vaapi.forget_probe()
    nvdec.forget_probe()
    if not decoders():
        pytest.skip('no hardware H.264 decoder reaches this OpenGL context')
    return True


@pytest.fixture
def nvenc_available(gl_context):  # noqa: ARG001 - requested so a GL context is current first
    """Skip unless this machine has an NVIDIA encoder to talk to."""
    if not nvenc.probe():
        # Naming only the library sends a reader looking for a file that is
        # very often sitting right there: the usual cause on Windows is a
        # context on the integrated GPU of a hybrid laptop, and elsewhere than
        # Linux and Windows there is no way in at all.
        pytest.skip('no NVENC encoder reachable from OpenGL here: '
                    + (nvenc.explain() or 'the NVIDIA driver library is absent, '
                       'or this is neither Linux nor Windows'))
    return True


@pytest.fixture
def vpl_available(gl_context):  # noqa: ARG001 - requested so a GL context is current first
    """Skip unless Intel's encoder is reachable from this OpenGL context."""
    if not vpl.probe():
        pytest.skip('no Intel encoder library here (oneVPL is reached through '
                    'Direct3D, so this backend is Windows-only for now)')

    if not interop.available():
        pytest.skip('this OpenGL driver does not offer WGL_NV_DX_interop2')
    adapter = interop.adapter_for_context()
    if adapter is None or adapter.vendor != 'Intel':
        pytest.skip(f'this OpenGL context is on {adapter}, not an Intel GPU, so '
                    'the Intel encoder could not read what it draws')
    return adapter


@pytest.fixture
def vpl_decoder_available(gl_context):  # noqa: ARG001 - requested so a GL context is current first
    """Skip unless Intel's decoder can write this OpenGL context's textures."""
    vpl.forget_probe()
    if not vpl.probe():
        pytest.skip('no Intel decoder library here (oneVPL is reached through '
                    'Direct3D, so this backend is Windows-only for now)')
    if not vpl.decoder_probe():
        pytest.skip('the Intel decoder is not reachable from this OpenGL context: '
                    + (vpl.decoder_explain() or 'there is no Intel GPU'))
    return vpl.decoder_capabilities()


def decode_errors(path):
    """What an outside decoder complains about, or None where there is none.

    ``ffmpeg`` is not a dependency of this package and is not needed to run the
    suite. Where it happens to be installed it is worth asking, because a
    stream can be well formed, correctly timed, and still describe itself in a
    way that sets a decoder up wrongly -- which nothing inside this package can
    notice.
    """
    if shutil.which('ffmpeg') is None:
        return None
    found = subprocess.run(
        ['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'null', '-'],
        capture_output=True, text=True)
    return [line for line in found.stderr.splitlines() if line.strip()]


def blit(handle, texture, size):
    """Copy a `size` texture into an encoder input, inside its drawing scope."""
    width, height = size
    source = int(glGenFramebuffers(1))
    with handle.for_drawing():
        glBindFramebuffer(GL_READ_FRAMEBUFFER, source)
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, int(texture), 0)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, handle.framebuffer)
        glBlitFramebuffer(0, 0, width, height, 0, 0, width, height,
                          GL_COLOR_BUFFER_BIT, GL_NEAREST)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
    glDeleteFramebuffers(1, [source])


def put_frame(handle, frame):
    """Replace the picture in an encoder input with an RGBA array, inside its drawing scope."""
    height, width = frame.shape[:2]
    with handle.for_drawing():
        glBindTexture(GL_TEXTURE_2D, handle.texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                        GL_RGBA, GL_UNSIGNED_BYTE, frame)
        glBindTexture(GL_TEXTURE_2D, 0)


def gradient_frame(width, height, phase=0):
    """An RGBA frame with a diagonal gradient that moves with `phase`.

    Something with structure in both directions and motion between frames, so a
    stream made of it exercises intra and inter prediction rather than
    compressing to nothing.
    """
    x = (np.arange(width, dtype=np.int32)[None, :] + phase * 4) % 256
    y = (np.arange(height, dtype=np.int32)[:, None] + phase * 2) % 256
    frame = np.empty((height, width, 4), dtype=np.uint8)
    frame[..., 0] = x
    frame[..., 1] = y
    frame[..., 2] = (x + y) % 256
    frame[..., 3] = 255
    return frame


@pytest.fixture
def upload_texture(gl_context):  # noqa: ARG001 - requested so a GL context is current first
    """Return a helper that puts an RGBA array into a new GL texture."""
    made = []

    def make(frame):
        height, width = frame.shape[:2]
        texture = int(glGenTextures(1))
        made.append(texture)
        glBindTexture(GL_TEXTURE_2D, texture)
        for parameter, value in (
            (GL_TEXTURE_MIN_FILTER, GL_LINEAR), (GL_TEXTURE_MAG_FILTER, GL_LINEAR),
            (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE),
        ):
            glTexParameteri(GL_TEXTURE_2D, parameter, value)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, frame)
        return texture

    def update(texture, frame):
        height, width = frame.shape[:2]
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                        GL_RGBA, GL_UNSIGNED_BYTE, frame)

    make.update = update
    yield make
    if made:
        glDeleteTextures(made)


# ------------------------------------------------------------ decoded frames


def read_plane(texture, format, kind=GL_UNSIGNED_BYTE):
    """A plane texture's samples, rows packed with no padding."""
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    glBindTexture(GL_TEXTURE_2D, texture)
    data = glGetTexImage(GL_TEXTURE_2D, 0, format, kind)
    glBindTexture(GL_TEXTURE_2D, 0)
    glPixelStorei(GL_PACK_ALIGNMENT, 4)
    return np.frombuffer(data, dtype=np.uint16 if kind == GL_UNSIGNED_SHORT else np.uint8)


def frame_bytes(frame):
    """The visible picture as its packed planes, first row at the top.

    NV12 for an ``'nv12'`` frame and little-endian P010 for a ``'p010'`` one,
    which are the bytes ``ffmpeg`` writes for those pixel formats and so what
    the fixture tables hold the MD5 of.
    """
    kind = GL_UNSIGNED_SHORT if frame.layout == 'p010' else GL_UNSIGNED_BYTE
    width, height = frame.texture_size
    x, y, w, h = frame.visible
    luma = read_plane(frame.planes[0], GL_RED, kind).reshape(height, width)[y:y + h, x:x + w]
    chroma = read_plane(frame.planes[1], GL_RG, kind).reshape(height // 2, width // 2, 2)[
        y // 2:(y + h + 1) // 2, x // 2:(x + w + 1) // 2]
    return luma.astype('<u2' if kind == GL_UNSIGNED_SHORT else np.uint8).tobytes() + chroma.astype(
        '<u2' if kind == GL_UNSIGNED_SHORT else np.uint8).tobytes()


def md5s(frames):
    """The MD5 of each frame's visible picture; every frame is released."""
    found = []
    for frame in frames:
        found.append(hashlib.md5(frame_bytes(frame)).hexdigest())
        frame.release()
    return found


def decode_all(decoder, samples):
    """The MD5 of every picture `samples` decode to, in display order."""
    found = []
    for sample in samples:
        found += md5s(decoder.decode(sample))
    return found + md5s(decoder.flush())


@pytest.fixture
def vaapi_decodes(vaapi_decoder_available):  # noqa: ARG001 - requested to skip where there is no VA-API decoder
    """A function that skips the test unless this machine decodes the codec named."""
    def need(codec):
        if codec not in vaapi.decoder_codecs():
            pytest.skip(f'no VA-API {codec} decoder on this machine')
    return need


@pytest.fixture
def vaapi_encodes(vaapi_available):  # noqa: ARG001 - requested to skip where there is no VA-API encoder
    """A function that skips the test unless this machine encodes the codec named."""
    def need(codec):
        if codec not in vaapi.encoder_codecs():
            pytest.skip(f'no VA-API {codec} encoder on this machine')
    return need
