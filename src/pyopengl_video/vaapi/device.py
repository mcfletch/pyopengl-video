"""A libva display opened on a DRM render node, and the calls every session makes.

Opening a render node, initialising libva on it with the driver's own log lines
kept off a caller's stderr, creating parameter buffers and running one
begin/render/end sequence are the same for an encoder and a decoder;
:class:`Device` holds them.
"""
from __future__ import annotations

import ctypes
import logging
import os
from collections.abc import Callable, Sequence
from typing import Any

from pyopengl_video.teardown import release_all
from pyopengl_video.vaapi import api

__all__ = ['Device', 'abandon', 'worded']

log = logging.getLogger(__name__)


def worded(error: BaseException) -> str:
    """What `error` says, led by its type's name where the type is not this package's.

    This package's errors are sentences. ``list index out of range`` or an
    empty :class:`MemoryError` is not one without the name of what was raised.
    """
    if type(error).__module__.startswith('pyopengl_video'):
        return str(error)
    return f'{type(error).__name__}: {error}' if str(error) else type(error).__name__


def abandon(close: Callable[[], object]) -> None:
    """Call `close` for an object whose construction failed, logging what it raises.

    The caller is about to re-raise the failure that stopped the
    construction, which is the one a reader needs; a failure to release what
    had been made is logged with its traceback.
    """
    try:
        close()
    except Exception:
        log.exception('releasing what a failed construction had made also failed')


class Device:
    """libva initialised on one render node.

    node -- the render node, such as ``/dev/dri/renderD128``
    vendor -- the driver's own description of itself
    version -- the VA-API version libva reports, as (major, minor)

    Raises :class:`~pyopengl_video.vaapi.api.VAError` or :class:`OSError`
    when the node will not open, and leaves nothing open behind it.
    """

    def __init__(self, node: str, va: api.VA | None = None) -> None:
        self.va = va or api.VA.instance()
        self.node = node
        self.display: Any = None
        # Kept alive for as long as the display is: libva holds the pointer.
        self._silence = api.MESSAGE_CALLBACK(lambda _context, _message: None)
        self._fd = os.open(node, os.O_RDWR)
        try:
            display = self.va.vaGetDisplayDRM(self._fd)
            if not display:
                raise api.VAError(-1, 'vaGetDisplayDRM', node)
            # The display exists from here, initialised or not: libva allocates
            # it in vaGetDisplayDRM and frees it in vaTerminate, so close()
            # terminates one that would not initialise too.
            self.display = display
            # Before vaInitialize, so the driver's log lines never reach a
            # caller's stderr: discovery runs in other people's programs.
            self.va.vaSetInfoCallback(display, self._silence, None)
            self.va.vaSetErrorCallback(display, self._silence, None)
            major, minor = ctypes.c_int(), ctypes.c_int()
            status = self.va.vaInitialize(display, ctypes.byref(major), ctypes.byref(minor))
            if status != api.VA_STATUS_SUCCESS:
                raise api.VAError(status, 'vaInitialize', node)
            self.version = (major.value, minor.value)
            vendor = self.va.vaQueryVendorString(display)
            self.vendor = vendor.decode('utf-8', 'replace') if vendor else ''
        except BaseException:
            abandon(self.close)
            raise

    def __repr__(self) -> str:
        return f'<{type(self).__name__} {self.node} {self.vendor!r}>'

    @classmethod
    def first(cls, nodes: Sequence[str], accept: Callable[[Device], str],
              what: str) -> Device:
        """The first node that opens and that `accept` has no objection to.

        accept -- returns '' for a device that will do, or why it will not
        what -- what was wanted, for the error when no node will do
        """
        if not nodes:
            raise api.VAError(-1, 'open', f'no DRM render node under /dev/dri '
                                          f'to open for {what}')
        failures = []
        for node in nodes:
            try:
                device = cls(node)
            except (OSError, api.VAError) as error:
                failures.append(f'{node}: {error}')
                continue
            try:
                objection = accept(device)
            except BaseException:
                device.close()
                raise
            if not objection:
                return device
            device.close()
            failures.append(f'{node}: {objection}')
        raise api.VAError(-1, 'open', f'no render node offers {what}: '
                                      + '; '.join(failures))

    def entrypoints(self, profile: int) -> list[int]:
        """The entrypoints this device offers for `profile`, empty for none."""
        count = self.va.vaMaxNumEntrypoints(self.display)
        if count <= 0:
            return []
        found = (api.VAEntrypoint * count)()
        number = ctypes.c_int(0)
        status = self.va.vaQueryConfigEntrypoints(self.display, profile, found,
                                                  ctypes.byref(number))
        if status != api.VA_STATUS_SUCCESS:
            return []
        return list(found)[:number.value]

    def surface_limits(self, config: Any) -> dict[int, int]:
        """The size limits of a configuration's surfaces, by surface attribute type.

        The keys are ``VASurfaceAttribMinWidth``, ``MaxWidth``, ``MinHeight``
        and ``MaxHeight``; a limit the driver does not state is absent.
        """
        count = ctypes.c_uint(0)
        self.check(self.va.vaQuerySurfaceAttributes(self.display, config, None,
                                                    ctypes.byref(count)),
                   'vaQuerySurfaceAttributes', 'counting')
        attributes = (api.VASurfaceAttrib * count.value)()
        self.check(self.va.vaQuerySurfaceAttributes(self.display, config, attributes,
                                                    ctypes.byref(count)),
                   'vaQuerySurfaceAttributes', 'size limits')
        wanted = (api.VASurfaceAttribMinWidth, api.VASurfaceAttribMaxWidth,
                  api.VASurfaceAttribMinHeight, api.VASurfaceAttribMaxHeight)
        return {entry.type: int(entry.value.value.i)
                for entry in list(attributes)[:count.value]
                if entry.type in wanted and entry.value.type == api.VAGenericValueTypeInteger}

    def largest(self, profile: int, entrypoint: int) -> tuple[int, int] | None:
        """The largest surface `profile` takes at `entrypoint`, as (width, height).

        None where the driver states no limit, or will not make the
        configuration to ask about.
        """
        config = api.VAConfigID(0)
        status = self.va.vaCreateConfig(self.display, profile, entrypoint, None, 0,
                                        ctypes.byref(config))
        if status != api.VA_STATUS_SUCCESS:
            return None
        try:
            limits = self.surface_limits(config)
        except api.VAError:
            return None
        finally:
            self.va.vaDestroyConfig(self.display, config)
        width = limits.get(api.VASurfaceAttribMaxWidth)
        height = limits.get(api.VASurfaceAttribMaxHeight)
        return (width, height) if width and height else None

    def check(self, status: int, call: str, context: str = '') -> None:
        """Raise :class:`~pyopengl_video.vaapi.api.VAError` unless `status` is success."""
        self.va.check(status, call, context)

    def buffer(self, context: Any, kind: int, payload: Any) -> int:
        """A VA buffer holding `payload`, a ctypes object or bytes."""
        if isinstance(payload, (bytes, bytearray, memoryview)):
            payload = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
        buffer = api.VABufferID(0)
        self.check(self.va.vaCreateBuffer(self.display, context, kind,
                                          ctypes.sizeof(payload), 1,
                                          ctypes.byref(payload), ctypes.byref(buffer)),
                   'vaCreateBuffer', type(payload).__name__)
        return int(buffer.value)

    def picture(self, context: Any, target: int, batches: Sequence[Sequence[int]],
                what: str) -> None:
        """One begin/render/end sequence, the buffers given back after it.

        Each batch is one ``vaRenderPicture`` call. ``vaEndPicture`` is what
        submits the picture, so every buffer is held until it has returned.

        A picture that was begun is always ended: when a batch is refused,
        ``vaEndPicture`` closes the sequence, its status unread, before the
        refusal is raised, so the context is ready for the next
        ``vaBeginPicture``. What `target` holds after that is undefined.
        """
        try:
            self.check(self.va.vaBeginPicture(self.display, context, target),
                       'vaBeginPicture', what)
            try:
                for batch in batches:
                    array = (api.VABufferID * len(batch))(*batch)
                    self.check(self.va.vaRenderPicture(self.display, context, array,
                                                       len(batch)),
                               'vaRenderPicture', f'{len(batch)} buffer(s) for {what}')
            except BaseException:
                self.va.vaEndPicture(self.display, context)
                raise
            self.check(self.va.vaEndPicture(self.display, context), 'vaEndPicture', what)
        finally:
            for batch in batches:
                for buffer in batch:
                    self.va.vaDestroyBuffer(self.display, buffer)

    def destroy_surfaces(self, surfaces: Sequence[int]) -> None:
        if surfaces and self.display is not None:
            array = (api.VASurfaceID * len(surfaces))(*surfaces)
            self.va.vaDestroySurfaces(self.display, array, len(surfaces))

    def close(self) -> None:
        """Terminate libva and close the render node. Safe to call twice.

        The node is closed whatever terminating does.
        """
        release_all(self._terminate, self._close_node)

    def _terminate(self) -> None:
        display, self.display = self.display, None
        if display is not None:
            self.va.vaTerminate(display)

    def _close_node(self) -> None:
        fd, self._fd = self._fd, -1
        if fd >= 0:
            os.close(fd)
