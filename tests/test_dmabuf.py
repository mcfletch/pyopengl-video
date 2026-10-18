"""Exporting an OpenGL texture as a DMA-BUF, and importing one back.

The handle every Linux hardware encoder takes. The tests that need a GPU say
what is missing when there is not one; the rest run anywhere.
"""
import os
import sys

import numpy as np
import pytest
from OpenGL import error
from OpenGL.GL import (
    GL_COLOR_ATTACHMENT0,
    GL_FRAMEBUFFER,
    GL_NEAREST,
    GL_R8,
    GL_RED,
    GL_RGBA,
    GL_TEXTURE_2D,
    GL_TEXTURE_BINDING_2D,
    GL_TEXTURE_MIN_FILTER,
    GL_UNSIGNED_BYTE,
    glBindFramebuffer,
    glBindTexture,
    glDeleteFramebuffers,
    glDeleteTextures,
    glFinish,
    glFramebufferTexture2D,
    glGenFramebuffers,
    glGenTextures,
    glGetIntegerv,
    glIsTexture,
    glReadPixels,
    glTexImage2D,
    glTexParameteri,
)
from OpenGL.GLES2.OES import EGL_image

from pyopengl_video.linux import dmabuf
from pyopengl_video.vaapi import api
from tests.conftest import gradient_frame

#: The `contexts` fixture: two more OpenGL contexts beside the suite's.
pytest_plugins = ('tests.other_context',)

SIZE = (320, 240)

#: The low and high halves of each plane's modifier, as
#: :data:`~pyopengl_video.linux.dmabuf._IMPORT_ATTRIBUTES` names them.
MODIFIER_ATTRIBUTES = tuple(plane[3:] for plane in dmabuf._IMPORT_ATTRIBUTES)  # noqa: SLF001 - the EGL attribute table under test

#: A single colour plane, which is what an RGBA frame exports as.
ONE_PLANE = (dmabuf.Plane(fd=7, offset=0, stride=1280),)

pytestmark = pytest.mark.skipif(not sys.platform.startswith('linux'),
                                reason='DMA-BUF is the Linux frame handle')


@pytest.fixture
def attributes_of(egl, monkeypatch):  # noqa: ARG001 - requested to skip where there is no EGL
    """Return what an import would hand ``eglCreateImageKHR``, as a mapping.

    Whether a display accepts a buffer whose layout was not named is the
    driver's business and differs between them; what the import asked for is
    this package's, and is the same everywhere.
    """
    from OpenGL.EGL.KHR import (  # noqa: PLC0415 - needs libEGL, which the egl fixture checks for
        image_base,
    )

    def ask(modifier, planes=ONE_PLANE):
        seen = {}

        def createImage(display, context, target, buffer, attributes):  # noqa: ARG001 - eglCreateImageKHR's signature
            # name/value pairs, with a lone EGL_NONE closing the list
            values = list(attributes)[:-1]
            seen.update(dict(zip(values[::2], values[1::2], strict=True)))
            return None                      # refused, so nothing is left over

        monkeypatch.setattr(dmabuf, 'import_available', lambda: True)
        monkeypatch.setattr(image_base, 'eglCreateImageKHR', createImage)
        with pytest.raises(dmabuf.DMABufError):
            dmabuf.import_texture(api.DRM_FORMAT_ABGR8888, *SIZE, modifier, planes)
        return seen

    return ask


@pytest.fixture
def refusing_import(egl, monkeypatch, gl_context):  # noqa: ARG001 - requested to skip where there is no EGL, with a context current
    """Import a buffer from a driver that takes the image and then refuses it.

    Both halves are stood in for, so the case is the same on any driver: EGL
    hands back an image, and the call that gives it to a texture fails. What
    the fixture records is whether that image was given back.
    """
    from OpenGL.EGL.KHR import (  # noqa: PLC0415 - needs libEGL, which the egl fixture checks for
        image_base,
    )

    destroyed: list = []
    image = object()

    def refuse(_target, _image):
        raise error.GLError(err=1282,
                            baseOperation='glEGLImageTargetTexture2DOES')

    monkeypatch.setattr(dmabuf, 'import_available', lambda: True)
    monkeypatch.setattr(image_base, 'eglCreateImageKHR',
                        lambda *_arguments: image)
    monkeypatch.setattr(EGL_image, 'glEGLImageTargetTexture2DOES', refuse)
    monkeypatch.setattr(dmabuf, '_destroy',
                        lambda _display, given: destroyed.append(given))

    def run():
        return dmabuf.import_texture(api.DRM_FORMAT_ABGR8888, *SIZE, 0,
                                     ONE_PLANE)

    run.destroyed = destroyed
    return run


@pytest.fixture
def exportable(gl_context):  # noqa: ARG001 - requested so a GL context is current first
    """Skip unless this OpenGL context can export a texture."""
    reason = dmabuf.unavailable_because()
    if reason:
        pytest.skip(reason)
    return True


@pytest.fixture
def exported(exportable, upload_texture):  # noqa: ARG001 - requested to skip where the context cannot export
    """A gradient texture, exported."""
    width, height = SIZE
    texture = upload_texture(gradient_frame(width, height))
    image = dmabuf.export_texture(texture, width, height)
    yield image
    image.close()


class TestWhatTheContextCanDo:
    @pytest.mark.usefixtures('exportable')
    def test_a_context_that_can_export_says_nothing_is_wrong(self):
        assert dmabuf.unavailable_because() == ''
        assert dmabuf.available() is True

    def test_the_reason_names_the_extension_or_the_context_kind(self):
        """Whatever the answer, it is a sentence a reader can act on."""
        reason = dmabuf.unavailable_because()
        assert reason == '' or 'EGL' in reason

    def test_a_context_that_is_not_egl_is_told_how_to_be_one(self, egl,
                                                             monkeypatch):
        monkeypatch.setattr(egl, 'eglGetCurrentDisplay', lambda: None)
        reason = dmabuf.unavailable_because()
        assert 'not an EGL context' in reason
        assert 'CONTEXT_CREATION_API' in reason, 'says what to do about it'
        assert dmabuf.available() is False

    @pytest.mark.usefixtures('exportable')
    def test_a_display_missing_the_extension_says_which_one(self, monkeypatch):
        monkeypatch.setattr(dmabuf, '_extensions', lambda: 'EGL_KHR_image_base')
        assert 'EGL_MESA_image_dma_buf_export' in dmabuf.unavailable_because()


@pytest.mark.usefixtures('without_egl')
class TestWithNoEGLLibraryAtAll:
    """EGL ships with the graphics driver, and plenty of machines have neither.

    A virtual machine, a container built without one, a CI runner: there is no
    libEGL to load, and PyOpenGL's EGL bindings raise on import rather than
    becoming a module whose entry points answer no. Nothing here may turn that
    into an error of its own -- the machine simply cannot export a texture, and
    that is one of the answers this module exists to give.
    """

    @pytest.fixture
    def without_egl(self, monkeypatch):
        monkeypatch.setattr(dmabuf, '_egl', lambda: None)

    def test_the_reason_says_there_is_no_egl_library(self):
        reason = dmabuf.unavailable_because()
        assert 'EGL' in reason
        assert 'driver' in reason, 'says where an EGL library comes from'

    def test_the_context_is_not_available_and_does_not_raise(self):
        assert dmabuf.available() is False
        assert dmabuf.import_available() is False

    def test_an_export_refuses_with_the_reason(self):
        with pytest.raises(dmabuf.DMABufError, match='EGL'):
            dmabuf.export_texture(1, *SIZE)

    def test_an_import_refuses_with_the_reason(self):
        with pytest.raises(dmabuf.DMABufError):
            dmabuf.import_texture(api.DRM_FORMAT_ABGR8888, *SIZE, 0, ())

    def test_releasing_nothing_is_still_harmless(self):
        dmabuf.release_imported_texture(0, None)


class TestExport:
    def test_a_texture_comes_back_as_a_buffer_of_the_same_size(self, exported):
        assert (exported.width, exported.height) == SIZE

    def test_an_rgba8_texture_exports_as_abgr8888(self, exported):
        """The DRM name reads from the low byte up, so RGBA8 is ABGR8888."""
        assert exported.fourcc == api.DRM_FORMAT_ABGR8888
        assert api.fourcc_name(exported.fourcc) == 'AB24'

    def test_a_colour_surface_is_one_plane(self, exported):
        assert len(exported.planes) == 1

    def test_the_descriptor_is_a_real_open_file(self, exported):
        assert exported.planes[0].fd >= 0
        assert os.fstat(exported.planes[0].fd).st_size > 0

    def test_the_stride_is_carried_across_rather_than_assumed(self, exported):
        """A driver pads rows out, so the stride is not the row's width."""
        width, _ = SIZE
        assert exported.planes[0].stride >= width * 4

    def test_it_describes_itself(self, exported):
        assert 'AB24' in repr(exported)
        assert '320x240' in repr(exported)

    @pytest.mark.usefixtures('exportable')
    def test_closing_gives_the_descriptor_back(self, upload_texture):
        width, height = SIZE
        image = dmabuf.export_texture(upload_texture(gradient_frame(*SIZE)),
                                      width, height)
        fd = image.planes[0].fd
        image.close()
        assert image.planes == ()
        with pytest.raises(OSError):
            os.fstat(fd)

    @pytest.mark.usefixtures('exportable')
    def test_closing_twice_is_harmless(self, upload_texture):
        image = dmabuf.export_texture(upload_texture(gradient_frame(*SIZE)),
                                      *SIZE)
        image.close()
        image.close()

    @pytest.mark.usefixtures('exportable')
    def test_a_texture_that_is_not_there_is_refused(self):
        with pytest.raises(dmabuf.DMABufError):
            dmabuf.export_texture(9999, *SIZE)

    @pytest.mark.usefixtures('egl')
    def test_a_driver_that_raises_rather_than_answering_is_the_same_refusal(self, monkeypatch):
        """A driver's two ways of saying no reach the caller as one exception.

        An entry point can refuse by handing back a null handle, or by setting
        an error, which PyOpenGL raises. Mesa does the second for a texture
        that is not there. Neither is a caller's to catch: an ``EGLError``
        names an entry point they did not call.
        """
        from OpenGL.EGL.KHR import (  # noqa: PLC0415 - needs libEGL, which the egl fixture checks for
            image_base,
        )
        from OpenGL.raw.EGL._errors import (  # noqa: PLC0415 - needs libEGL, which the egl fixture checks for
            EGLError,
        )

        def refuse(*_arguments):
            raise EGLError(err=0x300C, baseOperation='eglCreateImageKHR')

        monkeypatch.setattr(dmabuf, 'unavailable_because', lambda: None)
        monkeypatch.setattr(image_base, 'eglCreateImageKHR', refuse)
        with pytest.raises(dmabuf.DMABufError, match='refused texture 9999'):
            dmabuf.export_texture(9999, *SIZE)

    @pytest.mark.usefixtures('exportable')
    def test_a_context_that_cannot_export_refuses_with_the_reason(
            self, monkeypatch, upload_texture):
        monkeypatch.setattr(dmabuf, 'unavailable_because', lambda: 'no EGL here')
        with pytest.raises(dmabuf.DMABufError, match='no EGL here'):
            dmabuf.export_texture(upload_texture(gradient_frame(*SIZE)), *SIZE)


class TestTheModifier:
    """A tiled layout carried across as linear reads as noise, silently."""

    def test_the_modifier_is_reported_with_the_buffer(self, exported):
        assert isinstance(exported.modifier, int)

    def test_an_unnamed_layout_is_not_passed_on_as_a_claim(self, attributes_of):
        """``DRM_FORMAT_MOD_INVALID`` means the driver would not describe it.

        Handing that to an importer as though it were a layout would be a claim
        rather than the absence of one, so it is left out of the attributes.
        """
        named = attributes_of(api.DRM_FORMAT_MOD_INVALID)
        for low, high in MODIFIER_ATTRIBUTES:
            assert low not in named and high not in named, (
                'an absent layout was passed on as though it were one')

    def test_a_layout_the_driver_named_is_passed_on(self, attributes_of):
        """The other half: a real modifier reaches the importer, both halves.

        A tiled layout carried across as linear reads as noise, and nothing
        about the import says so.
        """
        modifier = 0x0300000000E08013            # a real block-linear layout
        named = attributes_of(modifier)
        low, high = MODIFIER_ATTRIBUTES[0]
        assert named[low] == modifier & 0xFFFFFFFF
        assert named[high] == modifier >> 32


class TestImport:
    @pytest.mark.usefixtures('exportable')
    def test_a_buffer_comes_back_as_a_texture_with_the_same_pixels(self, upload_texture):
        """Round trip: the same memory, under two names, reads the same."""
        if not dmabuf.import_available():
            pytest.skip('this EGL display cannot import a DMA-BUF')
        width, height = SIZE
        frame = gradient_frame(width, height)
        image = dmabuf.export_texture(upload_texture(frame), width, height)
        texture = egl_image = None
        framebuffer = 0
        try:
            texture, egl_image = dmabuf.import_texture(
                image.fourcc, image.width, image.height, image.modifier,
                image.planes)
            framebuffer = int(glGenFramebuffers(1))
            glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_2D, texture, 0)
            glFinish()
            read = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            read = np.frombuffer(read, np.uint8).reshape(height, width, 4)
            # Reading a framebuffer starts at its lower left, which for a
            # texture attachment is the texture's first row -- so the rows come
            # back in the order they were uploaded, with no flip between them.
            assert np.array_equal(read, frame)
        finally:
            if framebuffer:
                glDeleteFramebuffers(1, [framebuffer])
            if texture is not None:
                dmabuf.release_imported_texture(texture, egl_image)
            image.close()

    def test_a_buffer_the_driver_will_not_name_is_refused_with_a_reason(
            self, refusing_import):
        """Where the layout is checked differs between drivers.

        Mesa answers at ``eglCreateImageKHR``; NVIDIA takes the image and
        refuses when the texture is given it, which is the ordinary path on
        that driver rather than an exotic one. Either way the caller gets one
        exception that names the buffer, not a bare ``GL_INVALID_OPERATION``
        from a call it never made.
        """
        with pytest.raises(dmabuf.DMABufError) as raised:
            refusing_import()
        assert '320x240' in str(raised.value), 'names the buffer it refused'
        assert '0x0' in str(raised.value), 'names the layout it was given'

    def test_a_refused_buffer_leaves_no_image_behind(self, refusing_import):
        """The EGLImage and the texture are the importer's until it succeeds."""
        with pytest.raises(dmabuf.DMABufError):
            refusing_import()
        assert len(refusing_import.destroyed) == 1, (
            'the image the import made was not given back')

    @pytest.mark.usefixtures('exportable')
    def test_an_unnamed_layout_is_answered_rather_than_crashed_through(self, upload_texture):
        """What a real driver does with a buffer whose layout was not named.

        Some take it and some will not -- a block-linear buffer imported as
        though it were anonymous is one that cannot be read correctly, and
        refusing is right. Both are answers; a ``GLError`` reaching the caller
        is not.
        """
        if not dmabuf.import_available():
            pytest.skip('this EGL display cannot import a DMA-BUF')
        image = dmabuf.export_texture(upload_texture(gradient_frame(*SIZE)),
                                      *SIZE)
        try:
            texture, egl_image = dmabuf.import_texture(
                image.fourcc, image.width, image.height,
                api.DRM_FORMAT_MOD_INVALID, image.planes)
        except dmabuf.DMABufError:
            pass                             # a refusal that says what and why
        else:
            dmabuf.release_imported_texture(texture, egl_image)
        finally:
            image.close()

    @pytest.mark.usefixtures('exportable')
    def test_a_display_that_cannot_import_says_so(self, monkeypatch):
        monkeypatch.setattr(dmabuf, 'import_available', lambda: False)
        with pytest.raises(dmabuf.DMABufError,
                           match='EGL_EXT_image_dma_buf_import'):
            dmabuf.import_texture(api.DRM_FORMAT_ABGR8888, 16, 16, 0, ())

    @pytest.mark.usefixtures('exportable')
    def test_releasing_nothing_is_harmless(self):
        dmabuf.release_imported_texture(0, None)


class TestTheTextureDescribed:
    """An export states a size, and OpenGL is asked whether the texture has it."""

    @pytest.mark.usefixtures('exportable')
    def test_a_texture_of_another_size_than_stated_is_refused(self, upload_texture):
        texture = upload_texture(gradient_frame(16, 16))
        with pytest.raises(dmabuf.DMABufError, match='16x16.*4096x4096'):
            dmabuf.export_texture(texture, 4096, 4096)

    @pytest.mark.usefixtures('exportable')
    def test_a_texture_of_another_format_than_stated_is_refused(self):
        texture = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, 16, 16, 0, GL_RED, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glBindTexture(GL_TEXTURE_2D, 0)
        try:
            with pytest.raises(dmabuf.DMABufError, match='GL_RGBA8'):
                dmabuf.export_texture(texture, 16, 16)
        finally:
            glDeleteTextures([texture])

    @pytest.mark.usefixtures('exportable')
    def test_a_refused_texture_leaves_no_descriptor_open(self, upload_texture):
        texture = upload_texture(gradient_frame(16, 16))
        before = set(os.listdir('/proc/self/fd'))
        with pytest.raises(dmabuf.DMABufError):
            dmabuf.export_texture(texture, 32, 32)
        assert set(os.listdir('/proc/self/fd')) == before

    @pytest.mark.usefixtures('exportable')
    def test_asking_leaves_the_texture_binding_as_it_was(self, upload_texture):
        texture = upload_texture(gradient_frame(*SIZE))
        bound = upload_texture(gradient_frame(8, 8))
        glBindTexture(GL_TEXTURE_2D, bound)
        image = dmabuf.export_texture(texture, *SIZE)
        try:
            assert int(glGetIntegerv(GL_TEXTURE_BINDING_2D)) == bound
        finally:
            glBindTexture(GL_TEXTURE_2D, 0)
            image.close()


class TestBindings:
    @pytest.mark.usefixtures('exportable')
    def test_an_import_leaves_the_texture_binding_as_it_was(self, upload_texture):
        if not dmabuf.import_available():
            pytest.skip('this EGL display cannot import a DMA-BUF')
        image = dmabuf.export_texture(upload_texture(gradient_frame(*SIZE)), *SIZE)
        bound = upload_texture(gradient_frame(8, 8))
        glBindTexture(GL_TEXTURE_2D, bound)
        try:
            texture, egl_image = dmabuf.import_texture(
                image.fourcc, image.width, image.height, image.modifier, image.planes)
            assert int(glGetIntegerv(GL_TEXTURE_BINDING_2D)) == bound
            dmabuf.release_imported_texture(texture, egl_image)
        finally:
            glBindTexture(GL_TEXTURE_2D, 0)
            image.close()

    def test_a_refused_import_leaves_the_texture_binding_as_it_was(
            self, refusing_import, upload_texture):
        bound = upload_texture(gradient_frame(8, 8))
        glBindTexture(GL_TEXTURE_2D, bound)
        try:
            with pytest.raises(dmabuf.DMABufError):
                refusing_import()
            assert int(glGetIntegerv(GL_TEXTURE_BINDING_2D)) == bound
        finally:
            glBindTexture(GL_TEXTURE_2D, 0)

    def test_an_image_is_given_back_when_no_texture_can_be_made_for_it(
            self, refusing_import, monkeypatch):
        """Any failure after the image is made releases it, not only the last call's."""
        import OpenGL.GL  # noqa: PLC0415 - the module the import reads its entry points from

        def refuse(_count):
            raise error.GLError(err=1285, baseOperation='glGenTextures')
        monkeypatch.setattr(OpenGL.GL, 'glGenTextures', refuse)
        with pytest.raises(dmabuf.DMABufError):
            refusing_import()
        assert len(refusing_import.destroyed) == 1


class TestAnotherContextCurrent:
    """An OpenGL name is deleted in the context that made it, or not at all."""

    @pytest.fixture
    def imported(self, exportable, upload_texture):  # noqa: ARG002 - requested to skip where the context cannot export
        if not dmabuf.import_available():
            pytest.skip('this EGL display cannot import a DMA-BUF')
        image = dmabuf.export_texture(upload_texture(gradient_frame(*SIZE)), *SIZE)
        texture, egl_image = dmabuf.import_texture(
            image.fourcc, image.width, image.height, image.modifier, image.planes)
        yield texture, egl_image
        dmabuf.release_imported_texture(texture, egl_image)
        image.close()

    def test_an_imported_texture_is_not_deleted_from_another_context(self, imported, contexts):
        texture, egl_image = imported
        contexts.use(contexts.second)
        victim = contexts.texture_named(texture)
        with pytest.raises(dmabuf.DMABufError, match='another OpenGL context'):
            dmabuf.release_imported_texture(texture, egl_image)
        assert glIsTexture(victim), "the other context's texture of the same name was deleted"
        contexts.use(contexts.first)
        assert glIsTexture(texture)
        dmabuf.release_imported_texture(texture, egl_image)
        assert not glIsTexture(texture)

    def test_an_imported_texture_is_not_deleted_with_no_context_current(self, imported, contexts):
        texture, egl_image = imported
        contexts.use(None)
        with pytest.raises(dmabuf.DMABufError, match='no OpenGL context'):
            dmabuf.release_imported_texture(texture, egl_image)
        contexts.use(contexts.first)
        assert glIsTexture(texture)

    def test_an_imported_texture_is_deleted_from_a_context_that_shares(self, imported, contexts):
        texture, egl_image = imported
        contexts.use(contexts.sharing)
        dmabuf.release_imported_texture(texture, egl_image)
        assert not glIsTexture(texture)
        assert egl_image.image is None
        contexts.use(contexts.first)
        assert not glIsTexture(texture)

    @pytest.mark.usefixtures('exportable')
    def test_an_export_closes_in_a_context_that_shares(self, upload_texture, contexts):
        image = dmabuf.export_texture(upload_texture(gradient_frame(*SIZE)), *SIZE)
        contexts.use(contexts.sharing)
        image.close()
        assert image.planes == ()
        assert image._image is None  # noqa: SLF001 - the EGLImage, which the close gave back

    def test_releasing_an_imported_texture_twice_deletes_one_texture(self, imported):
        texture, egl_image = imported
        dmabuf.release_imported_texture(texture, egl_image)
        reused = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, reused)
        glBindTexture(GL_TEXTURE_2D, 0)
        try:
            dmabuf.release_imported_texture(texture, egl_image)
            assert glIsTexture(reused)
        finally:
            glDeleteTextures([reused])

    @pytest.mark.usefixtures('exportable')
    @pytest.mark.parametrize('current', ['second', None])
    def test_an_export_closes_its_descriptors_and_keeps_its_image(
            self, upload_texture, contexts, current):
        image = dmabuf.export_texture(upload_texture(gradient_frame(*SIZE)), *SIZE)
        fd = image.planes[0].fd
        contexts.use(getattr(contexts, current) if current else None)
        with pytest.raises(dmabuf.DMABufError, match='OpenGL context'):
            image.close()
        assert image.planes == ()
        with pytest.raises(OSError):
            os.fstat(fd)
        contexts.use(contexts.first)
        image.close()
        image.close()


class TestClosingAnExport:
    def exported(self, fds):
        planes = tuple(dmabuf.Plane(fd=fd, offset=0, stride=64) for fd in fds)
        return dmabuf.ExportedImage('a display', 'an image', api.DRM_FORMAT_ABGR8888,
                                    16, 16, 0, planes)

    def test_every_descriptor_is_closed_when_closing_one_fails(self, monkeypatch):
        closed, destroyed = [], []

        def close(fd):
            closed.append(fd)
            if fd == 7:
                raise OSError(9, 'Bad file descriptor')
        monkeypatch.setattr(dmabuf.os, 'close', close)
        monkeypatch.setattr(dmabuf, '_destroy', lambda _display, image: destroyed.append(image))
        image = self.exported((7, 8))
        with pytest.raises(OSError):
            image.close()
        assert closed == [7, 8]
        assert destroyed == ['an image']
        assert image.planes == ()
        image.close()
        assert closed == [7, 8]
        assert destroyed == ['an image']


class TestPlanes:
    def test_a_plane_records_what_an_importer_needs(self):
        plane = dmabuf.Plane(fd=7, offset=64, stride=1536)
        assert (plane.fd, plane.offset, plane.stride) == (7, 64, 1536)
