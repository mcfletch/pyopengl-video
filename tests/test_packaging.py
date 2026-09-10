"""What this distribution declares about itself, and why the alpha resolves.

This package needs PyOpenGL 4.0.0a5, which is a **pre-release**, and a
pre-release is not installed by default: `pip install pyopengl-video` gets one
only because PEP 440 says a specifier that explicitly names a pre-release admits
pre-releases for that requirement. `PyOpenGL>=4.0.0a5` does; `PyOpenGL>=4.0.0`
does not, and would resolve to nothing at all until 4.0.0 final exists.

So the marker in the floor is load-bearing rather than decorative, and a later
tidy-up that rounds it to `>=4.0.0` would leave a distribution nobody can
install -- silently, since the metadata still looks reasonable. That is what
these check.

The version is read from the *installed metadata* rather than from
`pyproject.toml`, because metadata is what a resolver reads. A checkout whose
metadata has gone stale is worth failing on too: it is the same staleness that
makes a locally-built wheel disagree with the source it came from.
"""
import importlib.metadata

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

import pyopengl_video

DISTRIBUTION = 'pyopengl-video'


def declared(name):
    """This distribution's declared requirement on *name*, or None"""
    wanted = canonicalize_name(name)
    for raw in importlib.metadata.requires(DISTRIBUTION) or []:
        requirement = Requirement(raw)
        if canonicalize_name(requirement.name) == wanted and not requirement.marker:
            return requirement
    return None


class TestTheDeclaredStack:
    def test_pyopengl_is_required(self):
        assert declared('PyOpenGL') is not None, (
            'the whole package is texture and framebuffer calls; PyOpenGL is '
            'not optional')

    def test_the_floor_admits_a_prerelease(self):
        """Which is what lets `pip install pyopengl-video` find 4.0.0a5.

        Without a pre-release in the specifier, PEP 440 has a resolver skip
        every pre-release of PyOpenGL, and there is no final 4.x to fall back
        to -- so the install fails rather than quietly using an older one.
        """
        requirement = declared('PyOpenGL')
        assert requirement.specifier.prereleases, (
            f'PyOpenGL is required as {str(requirement)!r}, which names no '
            f'pre-release. The release this package needs is an alpha, and a '
            f'specifier that does not mention one excludes it: write '
            f'>=4.0.0a5 rather than >=4.0.0.')

    def test_the_floor_is_at_least_the_release_this_package_needs(self):
        """4.0.0a4 brought the DMA-BUF entry points, a5 the stub for them.

        `glDeleteTextures(textures)` -- the one-argument form this package
        calls, and the one PyOpenGL's own wrapper documents -- reached the
        shipped stub in a5. Against a4 the call is an error to a type checker,
        so a floor of a4 is a floor this package does not type-check at.
        """
        requirement = declared('PyOpenGL')
        assert requirement.specifier.contains('4.0.0a5'), str(requirement)
        assert not requirement.specifier.contains('4.0.0a4'), (
            f'the floor is lower than 4.0.0a5: {requirement}')


class TestTheInstalledStack:
    def test_the_installed_pyopengl_satisfies_the_floor(self):
        """A stale environment resolves differently from a fresh one, and this
        is where that shows up rather than in a driver call three files away."""
        requirement = declared('PyOpenGL')
        installed = importlib.metadata.version('PyOpenGL')
        assert requirement.specifier.contains(installed, prereleases=True), (
            f'PyOpenGL {installed} is installed, which does not satisfy '
            f'{requirement}')

    def test_the_metadata_version_matches_the_module(self):
        """They part company when an editable install has not been refreshed
        after a version bump, which is exactly when a release is being cut."""
        assert (Version(importlib.metadata.version(DISTRIBUTION))
                == Version(pyopengl_video.__version__))


def test_the_version_is_a_release_number_a_resolver_can_order():
    version = Version(pyopengl_video.__version__)
    assert version.release, pyopengl_video.__version__


if __name__ == '__main__':  # pragma: no cover - convenience
    raise SystemExit(pytest.main([__file__]))
