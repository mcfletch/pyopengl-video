"""Which OpenGL context an object was made in, and whether it is the current one."""
import pytest

from pyopengl_video.backend import VideoError
from pyopengl_video.decoder import DecoderError
from pyopengl_video.glcontext import Owner

#: The `contexts` fixture: a second OpenGL context beside the suite's.
pytest_plugins = ('tests.other_context',)


class TestAnUnknownOwner:
    """Made where no binding named a context: nothing is checked."""

    def test_it_is_always_current(self):
        assert Owner(None).is_current()

    def test_it_requires_nothing(self):
        Owner(None).require('a texture')

    def test_it_describes_itself(self):
        assert 'unknown' in repr(Owner(None))


class TestAKnownOwner:
    def test_the_context_it_was_made_in_is_current(self, contexts):  # noqa: ARG002 - requested so a GL context is current first
        owner = Owner.current()
        assert owner.is_current()
        owner.require('a texture')
        assert 'context' in repr(owner)

    def test_an_egl_owner_keeps_its_display(self, contexts, egl):  # noqa: ARG002 - requested so an EGL context is current first
        assert Owner.current().display is not None

    def test_another_context_is_not_it(self, contexts):
        owner = Owner.current()
        contexts.use(contexts.second)
        assert not owner.is_current()
        assert Owner.current().is_current()
        with pytest.raises(VideoError, match='a texture.*another OpenGL context'):
            owner.require('a texture')

    def test_no_context_is_not_it(self, contexts):
        owner = Owner.current()
        contexts.use(None)
        assert not owner.is_current()
        with pytest.raises(DecoderError, match='no OpenGL context is current'):
            owner.require('a texture', DecoderError)

    def test_an_object_made_with_no_context_has_an_unknown_owner(self, contexts):
        contexts.use(None)
        owner = Owner.current()
        contexts.use(contexts.second)
        assert owner.is_current()

    def test_a_context_that_is_not_current_is_refused_whatever_it_is(self):
        with pytest.raises(VideoError):
            Owner(('egl', 1)).require('a texture')
