"""VP9 frames written by hand, for the tests of the parser and the control layer.

:func:`frame` writes an uncompressed header from the values a test names and
puts filler behind it where the entropy coded data would be. Nothing decodes
the filler, so these frames suit a parser and a stand-in session and no driver.
"""
from pyopengl_video.vaapi.bits import BitWriter

SYNC_CODE = (0x49, 0x83, 0x42)
KEY, INTER = 0, 1


def _signed(writer, value, bits):
    """A magnitude of `bits` bits, then the sign."""
    writer.f(abs(value), bits)
    writer.flag(value < 0)


def _colour(writer, profile, depth, color_space, color_range, subsampling):
    if profile >= 2:
        writer.flag(depth == 12)
    writer.f(color_space, 3)
    if color_space != 7:
        writer.flag(color_range)
        if profile in (1, 3):
            writer.flag(subsampling[0])
            writer.flag(subsampling[1])
            writer.flag(0)
    elif profile in (1, 3):
        writer.flag(0)


def _size(writer, size):
    writer.f(size[0] - 1, 16)
    writer.f(size[1] - 1, 16)


def _render(writer, render):
    writer.flag(render is not None)
    if render is not None:
        _size(writer, render)


def frame(*, profile=0, frame_type=KEY, show_frame=1, error_resilient=0,
          intra_only=0, reset_frame_context=0, depth=8, color_space=2,
          color_range=0, subsampling=(1, 1), size=(64, 48), render=None,
          refresh=0xFF, refs=(0, 1, 2), sign_bias=(0, 0, 0), found_ref=0,
          high_precision=1, filter=None, refresh_context=1, parallel=0,
          context=0, level=0, sharpness=0, deltas=None, base_q=60,
          q_deltas=(0, 0, 0), segmentation=None, tile_columns=(), tile_rows=0,
          compressed=4, tiles=8, tile_data=None, sync=SYNC_CODE, marker=2):
    """One frame: the uncompressed header these values state, then filler.

    found_ref -- which of the three references gives the size, or None for a
        size the header states itself
    filter -- the raw_interpolation_filter value, or None for a switchable filter
    deltas -- None for loop filter deltas left alone, or a pair of dictionaries
        of the reference and mode deltas to write, by index
    segmentation -- None for none, or a dictionary: ``tree`` (seven
        probabilities, None for one not coded), ``pred`` (three, with a
        temporal update), ``abs`` and ``features`` ({(segment, feature): value})
    tile_columns -- the increment bits to write
    compressed/tiles -- how many filler bytes stand for the compressed header
        and for the last tile. Each tile before the last is its four-byte
        size and one byte.
    tile_data -- the bytes to put after the compressed header in place of
        that, sizes included
    """
    writer = BitWriter()
    writer.f(marker, 2)
    writer.flag(profile & 1)
    writer.flag(profile >> 1)
    if profile == 3:
        writer.flag(0)
    writer.flag(0)                              # show_existing_frame
    writer.flag(frame_type)
    writer.flag(show_frame)
    writer.flag(error_resilient)
    if frame_type == KEY:
        for byte in sync:
            writer.f(byte, 8)
        _colour(writer, profile, depth, color_space, color_range, subsampling)
        _size(writer, size)
        _render(writer, render)
    else:
        if not show_frame:
            writer.flag(intra_only)
        if not error_resilient:
            writer.f(reset_frame_context, 2)
        if intra_only:
            for byte in sync:
                writer.f(byte, 8)
            if profile > 0:
                _colour(writer, profile, depth, color_space, color_range, subsampling)
            writer.f(refresh, 8)
            _size(writer, size)
            _render(writer, render)
        else:
            writer.f(refresh, 8)
            for slot, bias in zip(refs, sign_bias, strict=True):
                writer.f(slot, 3)
                writer.flag(bias)
            for index in range(3):
                writer.flag(index == found_ref)
                if index == found_ref:
                    break
            else:
                _size(writer, size)
            _render(writer, render)
            writer.flag(high_precision)
            writer.flag(filter is None)
            if filter is not None:
                writer.f(filter, 2)
    if not error_resilient:
        writer.flag(refresh_context)
        writer.flag(parallel)
    writer.f(context, 2)
    writer.f(level, 6)
    writer.f(sharpness, 3)
    writer.flag(deltas is not None)             # loop_filter_delta_enabled
    if deltas is not None:
        writer.flag(any(deltas))
        if any(deltas):
            for count, changed in zip((4, 2), deltas, strict=True):
                for index in range(count):
                    writer.flag(index in changed)
                    if index in changed:
                        _signed(writer, changed[index], 6)
    writer.f(base_q, 8)
    for delta in q_deltas:
        writer.flag(delta != 0)
        if delta:
            _signed(writer, delta, 4)
    writer.flag(segmentation is not None)
    if segmentation is not None:
        tree = segmentation.get('tree')
        writer.flag(tree is not None)
        if tree is not None:
            for probability in tree:
                writer.flag(probability is not None)
                if probability is not None:
                    writer.f(probability, 8)
            pred = segmentation.get('pred')
            writer.flag(pred is not None)
            for probability in pred or ():
                writer.flag(probability is not None)
                if probability is not None:
                    writer.f(probability, 8)
        features = segmentation.get('features')
        writer.flag(features is not None)
        if features is not None:
            writer.flag(segmentation.get('abs', 0))
            for segment in range(8):
                for feature, (bits, signed) in enumerate(((8, 1), (6, 1), (2, 0), (0, 0))):
                    value = features.get((segment, feature))
                    writer.flag(value is not None)
                    if value is not None:
                        writer.f(abs(value), bits)
                        if signed:
                            writer.flag(value < 0)
    for bit in tile_columns:
        writer.flag(bit)
    writer.flag(tile_rows > 0)
    if tile_rows:
        writer.flag(tile_rows > 1)
    writer.f(compressed, 16)
    filler = bytes(1 + index % 255 for index in range(compressed))
    if tile_data is None:
        count = tile_count(size[0], tile_columns, tile_rows)
        tile_data = (b'\x00\x00\x00\x01\x55') * (count - 1) + b'\x55' * tiles
    return writer.bytes() + filler + tile_data


def tile_count(width, tile_columns=(), tile_rows=0):
    """How many tiles a frame `width` wide has, for the tile bits :func:`frame` writes."""
    sb64_cols = (width + 63) // 64
    log2 = 0
    while (64 << log2) < sb64_cols:
        log2 += 1
    for bit in tile_columns:
        if not bit:
            break
        log2 += 1
    return (1 << log2) << tile_rows


def show_existing(slot, profile=0):
    """A frame that shows the picture in reference slot `slot` again."""
    writer = BitWriter()
    writer.f(2, 2)
    writer.flag(profile & 1)
    writer.flag(profile >> 1)
    if profile == 3:
        writer.flag(0)
    writer.flag(1)
    writer.f(slot, 3)
    return writer.bytes()


def superframe(*frames, size_bytes=2):
    """`frames` as one sample, with the index of their sizes behind them."""
    marker = 0xC0 | (size_bytes - 1) << 3 | (len(frames) - 1)
    index = b''.join(len(data).to_bytes(size_bytes, 'little') for data in frames)
    return b''.join(frames) + bytes([marker]) + index + bytes([marker])
