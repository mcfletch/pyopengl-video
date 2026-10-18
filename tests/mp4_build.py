"""A minimal MP4 built box by box, for sample entries no writer here produces."""
import struct

from pyopengl_video.mp4 import box, full_box


def visual_entry(code, width, height, *children):
    """A VisualSampleEntry of four-character `code` holding `children` boxes."""
    return box(
        code,
        bytes(6), struct.pack('>H', 1), bytes(16),
        struct.pack('>HH', width, height),
        struct.pack('>II', 0x00480000, 0x00480000), struct.pack('>I', 0),
        struct.pack('>H', 1), bytes(32), struct.pack('>Hh', 0x0018, -1),
        *children)


def movie(entry, samples, *, timescale=30, sync=None, tables=None):
    """The bytes of an MP4 with one video track of `samples`, each one tick long.

    entry -- the sample entry box, from :func:`visual_entry`
    sync -- the indices of the sync samples; every sample when None
    tables -- boxes by four-character code that replace the sample tables built
        here, or add to them, for a file whose tables are to be malformed
    """
    ftyp = box('ftyp', b'isom', struct.pack('>I', 512), b'isom')
    mdat = box('mdat', *samples)
    first = len(ftyp) + 8
    count = len(samples)
    built = {
        'stsd': full_box('stsd', 0, 0, struct.pack('>I', 1), entry),
        'stts': full_box('stts', 0, 0, struct.pack('>III', 1, count, 1)),
        'stsc': full_box('stsc', 0, 0, struct.pack('>IIII', 1, 1, count, 1)),
        'stsz': full_box('stsz', 0, 0, struct.pack('>II', 0, count),
                         *(struct.pack('>I', len(sample)) for sample in samples)),
        'stco': full_box('stco', 0, 0, struct.pack('>II', 1, first)),
    }
    if sync is not None:
        built['stss'] = full_box('stss', 0, 0, struct.pack('>I', len(sync)),
                                 *(struct.pack('>I', index + 1) for index in sync))
    built.update(tables or {})
    minf = box('minf', full_box('vmhd', 0, 1, bytes(8)), box('stbl', *built.values()))
    mdia = box(
        'mdia',
        full_box('mdhd', 0, 0, struct.pack('>IIII', 0, 0, timescale, count),
                 struct.pack('>HH', 0x55c4, 0)),
        full_box('hdlr', 0, 0, struct.pack('>I4s', 0, b'vide'), bytes(12), b'v\x00'),
        minf)
    tkhd = full_box('tkhd', 0, 3, struct.pack('>IIIII', 0, 0, 1, 0, count), bytes(60))
    mvhd = full_box('mvhd', 0, 0, struct.pack('>IIII', 0, 0, timescale, count), bytes(80))
    return ftyp + mdat + box('moov', mvhd, box('trak', tkhd, mdia))
