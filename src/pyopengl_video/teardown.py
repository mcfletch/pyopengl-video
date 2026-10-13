"""Releasing several resources, where a failed release must not strand the rest.

A decoder or an encoder owns driver objects of several kinds: a session, the
surfaces, file descriptors, OpenGL names. Each is released by a call that can
fail, and the ones released later are the ones the earlier ones depend on.
:func:`release_all` runs every release whatever the others did.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

__all__ = ['release_all']

log = logging.getLogger(__name__)


def release_all(*steps: Callable[[], object]) -> None:
    """Call every one of `steps` in order, then raise the first failure.

    A step that raises does not stop the steps after it. The first exception
    is raised once all have run; any later one is logged with its traceback.
    Each step should clear the handle it releases, so that calling it again
    does nothing.
    """
    first: BaseException | None = None
    for step in steps:
        try:
            step()
        except BaseException as error:
            if first is None:
                first = error
            else:
                log.exception('a later release also failed: %r', error)
    if first is not None:
        raise first
