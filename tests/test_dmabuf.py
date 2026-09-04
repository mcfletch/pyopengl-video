"""Exporting an OpenGL texture as a DMA-BUF, and importing one back.

The handle every Linux hardware encoder takes. The tests that need a GPU say
what is missing when there is not one; the rest run anywhere.
"""
import os
import sys

import numpy as np
import pytest

from pyopengl_video.linux import dmabuf
from pyopengl_video.vaapi import api
from tests.conftest import gradient_frame

SIZE = (320, 240)

pytestmark = pytest.mark.skipif(not sys.platform.startswith('linux'),
                                reason='DMA-BUF is the Linux frame handle')


@pytest.fixture
def exportable(gl_context):
    """Skip unless this OpenGL context can export a texture."""
    reason = dmabuf.unavailable_because()
    if reason:
        pytest.skip(reason)
    return True


@pytest.fixture
def exported(exportable, upload_texture):
    """A gradient texture, exported."""
    width, height = SIZE
    texture = upload_texture(gradient_frame(width, height))
    image = dmabuf.export_texture(texture, width, height)
    yield image
    image.close()


class TestWhatTheContextCanDo:
    def test_a_context_that_can_export_says_nothing_is_wrong(self, exportable):
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

    def test_a_display_missing_the_extension_says_which_one(self, monkeypatch,
                                                            exportable):
        monkeypatch.setattr(dmabuf, '_extensions', lambda: 'EGL_KHR_image_base')
        assert 'EGL_MESA_image_dma_buf_export' in dmabuf.unavailable_because()


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

    def test_the_reason_says_there_is_no_egl_library(self, without_egl):
        reason = dmabuf.unavailable_because()
        assert 'EGL' in reason
        assert 'driver' in reason, 'says where an EGL library comes from'

    def test_the_context_is_not_available_and_does_not_raise(self, without_egl):
        assert dmabuf.available() is False
        assert dmabuf.import_available() is False

    def test_an_export_refuses_with_the_reason(self, without_egl):
        with pytest.raises(dmabuf.DMABufError, match='EGL'):
            dmabuf.export_texture(1, *SIZE)

    def test_an_import_refuses_with_the_reason(self, without_egl):
        with pytest.raises(dmabuf.DMABufError):
            dmabuf.import_texture(api.DRM_FORMAT_ABGR8888, *SIZE, 0, ())

    def test_releasing_nothing_is_still_harmless(self, without_egl):
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

    def test_closing_gives_the_descriptor_back(self, exportable, upload_texture):
        width, height = SIZE
        image = dmabuf.export_texture(upload_texture(gradient_frame(*SIZE)),
                                      width, height)
        fd = image.planes[0].fd
        image.close()
        assert image.planes == ()
        with pytest.raises(OSError):
            os.fstat(fd)

    def test_closing_twice_is_harmless(self, exportable, upload_texture):
        image = dmabuf.export_texture(upload_texture(gradient_frame(*SIZE)),
                                      *SIZE)
        image.close()
        image.close()

    def test_a_texture_that_is_not_there_is_refused(self, exportable):
        with pytest.raises(dmabuf.DMABufError):
            dmabuf.export_texture(9999, *SIZE)

    def test_a_context_that_cannot_export_refuses_with_the_reason(
            self, monkeypatch, exportable, upload_texture):
        monkeypatch.setattr(dmabuf, 'unavailable_because', lambda: 'no EGL here')
        with pytest.raises(dmabuf.DMABufError, match='no EGL here'):
            dmabuf.export_texture(upload_texture(gradient_frame(*SIZE)), *SIZE)


class TestTheModifier:
    """A tiled layout carried across as linear reads as noise, silently."""

    def test_the_modifier_is_reported_with_the_buffer(self, exported):
        assert isinstance(exported.modifier, int)

    def test_an_unnamed_layout_is_not_passed_on_as_a_claim(self, exportable,
                                                           upload_texture):
        """``DRM_FORMAT_MOD_INVALID`` means the driver would not describe it.

        Handing that to an importer as though it were a layout would be a claim
        rather than the absence of one, so it is left out of the attributes.
        """
        if not dmabuf.import_available():
            pytest.skip('this EGL display cannot import a DMA-BUF')
        image = dmabuf.export_texture(upload_texture(gradient_frame(*SIZE)),
                                      *SIZE)
        try:
            texture, egl_image = dmabuf.import_texture(
                image.fourcc, image.width, image.height,
                api.DRM_FORMAT_MOD_INVALID, image.planes)
            dmabuf.release_imported_texture(texture, egl_image)
        finally:
            image.close()


class TestImport:
    def test_a_buffer_comes_back_as_a_texture_with_the_same_pixels(
            self, exportable, upload_texture):
        """Round trip: the same memory, under two names, reads the same."""
        from OpenGL.GL import (
            GL_COLOR_ATTACHMENT0,
            GL_FRAMEBUFFER,
            GL_RGBA,
            GL_TEXTURE_2D,
            GL_UNSIGNED_BYTE,
            glBindFramebuffer,
            glDeleteFramebuffers,
            glFinish,
            glFramebufferTexture2D,
            glGenFramebuffers,
            glReadPixels,
        )
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

    def test_a_display_that_cannot_import_says_so(self, monkeypatch, exportable):
        monkeypatch.setattr(dmabuf, 'import_available', lambda: False)
        with pytest.raises(dmabuf.DMABufError,
                           match='EGL_EXT_image_dma_buf_import'):
            dmabuf.import_texture(api.DRM_FORMAT_ABGR8888, 16, 16, 0, ())

    def test_releasing_nothing_is_harmless(self, exportable):
        dmabuf.release_imported_texture(0, None)


class TestPlanes:
    def test_a_plane_records_what_an_importer_needs(self):
        plane = dmabuf.Plane(fd=7, offset=64, stride=1536)
        assert (plane.fd, plane.offset, plane.stride) == (7, 64, 1536)
