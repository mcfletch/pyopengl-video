"""Which OpenGL context an object was made in, and whether it is the current one."""
import pytest

from pyopengl_video import glcontext
from pyopengl_video.backend import VideoError
from pyopengl_video.decoder import DecoderError
from pyopengl_video.glcontext import Owner

#: The `contexts` fixture: two more OpenGL contexts beside the suite's.
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


class TestAContextThatShares:
    """A texture, a program or a sync object is the same object across a share group."""

    def test_an_unknown_owner_is_shared_with_everything(self):
        assert Owner(None).is_shared()
        Owner(None).require_own('a framebuffer')

    def test_its_own_context_shares_with_it(self, contexts):  # noqa: ARG002 - requested so a GL context is current first
        owner = Owner.current()
        assert owner.is_shared()
        owner.require_own('a framebuffer')

    @pytest.mark.parametrize(('made_in', 'released_in'),
                             [('first', 'sharing'), ('sharing', 'first')])
    def test_what_it_shares_is_released_from_either_context(
            self, contexts, made_in, released_in):
        contexts.use(getattr(contexts, made_in))
        owner = Owner.current()
        contexts.use(getattr(contexts, released_in))
        assert not owner.is_current()
        assert owner.is_shared()
        owner.require('a texture')

    def test_what_it_does_not_share_is_released_in_its_own(self, contexts):
        owner = Owner.current()
        contexts.use(contexts.sharing)
        with pytest.raises(DecoderError, match='a framebuffer.*not shared'):
            owner.require_own('a framebuffer', DecoderError)
        contexts.use(None)
        with pytest.raises(VideoError, match='no OpenGL context is current'):
            owner.require_own('a framebuffer')

    def test_a_context_outside_the_group_does_not_share(self, contexts):
        owner = Owner.current()
        contexts.use(contexts.second)
        assert not owner.is_shared()
        outsider = Owner.current()
        contexts.use(contexts.sharing)
        assert not outsider.is_shared()
        with pytest.raises(VideoError, match='another OpenGL context'):
            outsider.require('a texture')

    def test_no_context_shares_with_nothing(self, contexts):
        owner = Owner.current()
        contexts.use(None)
        assert not owner.is_shared()

    def test_a_group_is_marked_once(self, contexts):
        first = Owner.current()
        contexts.use(contexts.sharing)
        assert Owner.current().group == first.group
        contexts.use(contexts.second)
        assert Owner.current().group != first.group

    def test_a_group_that_cannot_be_marked_is_its_own_context_alone(
            self, contexts, monkeypatch):
        contexts.use(contexts.second)
        monkeypatch.setattr(glcontext, '_marks', {})

        def refuse(*_arguments):
            raise RuntimeError('this context has no shaders')
        monkeypatch.setattr('OpenGL.GL.glCreateShader', refuse)
        owner = Owner.current()
        assert owner.group is None
        assert owner.is_shared()
        contexts.use(contexts.first)
        assert not owner.is_shared()
