"""Write the streams the decoders' tests read, and what they decode to.

Our own encoders write a narrow slice of each codec: one slice a picture, P
frames only, one reference. A decoder meets everything else a conforming
encoder may produce, so the suite carries streams from the reference software
encoders that use B-frame pyramids, weighted prediction, several slices or
tiles a picture, scaling matrices and open groups of pictures. One interlaced
H.264 stream is listed apart from those, for the decoders that take interlaced
video.

``ffmpeg`` writes them with the encoder each :class:`Stream` names, and
``ffmpeg``'s software decoder records the MD5 of each decoded picture, in
display order: as NV12 for an 8-bit stream and as little-endian P010 for a
10-bit one, which are the bytes a decoder's textures hold. Decoding these
codecs is exact, so a hardware decoder that is right produces the same bytes.
The streams and one table a codec, ``tests/data/decode/<codec>.json``, are
committed, so the suite needs neither ``ffmpeg`` nor this tool to run. The
package itself never runs ``ffmpeg``.

Run it from the project directory, for every codec or for the ones named::

    python tools/make_decode_fixtures.py
    python tools/make_decode_fixtures.py hevc av1
"""
from __future__ import annotations

import atexit
import dataclasses
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DESTINATION = Path(__file__).parent.parent / 'tests' / 'data' / 'decode'


@dataclasses.dataclass(frozen=True)
class Stream:
    """One fixture stream.

    codec -- the codec's name, which is also the table the stream is listed in
    size -- the picture, as (width, height). The sizes are not whole blocks,
        so every stream is cropped.
    frames -- how many pictures
    coding -- the ``ffmpeg`` options that choose the encoder and its settings
    suffix -- the elementary stream's file suffix
    depth -- bits a sample, 8 or 10
    mp4 -- also write the stream as an MP4 with a short audio track, so the
        container reader is tested on what ``ffmpeg`` writes: an edit list,
        signed composition offsets and a second track
    to_stream -- the options that take the MP4's video back out as the
        elementary stream, so both forms hold the same bitstream
    two_pass -- code the stream in two passes, which is what makes libvpx
        place alternate reference frames
    files -- text files the encoder reads, as (name, text) pairs; ``{name}``
        in a coding option stands for the path the file is written to
    monochrome -- the stream has no chroma planes. Its reference pictures are
        the decoded luma with chroma at the mid value, which is what a decoder
        writing 4:2:0 surfaces holds.
    group -- the part of the codec's table the stream is listed in:
        ``'streams'``, or ``'interlaced'`` for a stream a decoder of
        progressive frames refuses
    """

    codec: str
    size: tuple[int, int]
    frames: int
    coding: tuple[str, ...]
    suffix: str
    depth: int = 8
    mp4: bool = False
    to_stream: tuple[str, ...] = ()
    two_pass: bool = False
    files: tuple[tuple[str, str], ...] = ()
    monochrome: bool = False
    group: str = 'streams'

    @property
    def pixel_format(self) -> str:
        """The ``ffmpeg`` pixel format the stream is coded from."""
        return 'yuv420p' if self.depth == 8 else 'yuv420p10le'

    @property
    def reference_format(self) -> str:
        """The ``ffmpeg`` pixel format whose bytes a decoder's textures hold."""
        return 'nv12' if self.depth == 8 else 'p010le'

    @property
    def frame_bytes(self) -> int:
        width, height = self.size
        return width * height * 3 // 2 * (1 if self.depth == 8 else 2)


def x264(size: tuple[int, int], frames: int, profile: str, parameters: str,
         mp4: bool = False, interlaced: bool = False) -> Stream:
    """An H.264 stream from libx264; `interlaced` codes field macroblock pairs."""
    fields = ('-flags', '+ildct+ilme') if interlaced else ()
    return Stream('h264', size, frames,
                  ('-c:v', 'libx264', '-profile:v', profile, *fields,
                   '-x264-params', parameters),
                  '.h264', mp4=mp4, to_stream=('-bsf:v', 'h264_mp4toannexb'),
                  group='interlaced' if interlaced else 'streams')


def vpx(size: tuple[int, int], frames: int, options: str, *, depth: int = 8,
        mp4: bool = False, two_pass: bool = False) -> Stream:
    return Stream('vp9', size, frames, ('-c:v', 'libvpx-vp9', *options.split()),
                  '.ivf', depth=depth, mp4=mp4, two_pass=two_pass)


def x265(size: tuple[int, int], frames: int, parameters: str, *, depth: int = 8,
         mp4: bool = False, before: tuple[str, ...] = (),
         files: tuple[tuple[str, str], ...] = ()) -> Stream:
    """An HEVC stream from libx265; `before` are options ahead of the encoder's."""
    tag = ('-tag:v', 'hvc1') if mp4 else ()
    return Stream('hevc', size, frames,
                  (*before, '-c:v', 'libx265', *tag, '-x265-params',
                   f'log-level=error:{parameters}'),
                  '.h265', depth=depth, mp4=mp4,
                  to_stream=('-bsf:v', 'hevc_mp4toannexb'), files=files)


def av1(size: tuple[int, int], frames: int, options: str, *, depth: int = 8,
        mp4: bool = False, monochrome: bool = False) -> Stream:
    """An AV1 stream; `options` name the encoder, ahead of its settings."""
    return Stream('av1', size, frames, tuple(options.split()), '.ivf', depth=depth,
                  mp4=mp4, monochrome=monochrome)


def hevc_scaling_lists() -> str:
    """A scaling list file for libx265 in which every kind of coding occurs.

    The file gives each list in raster order. No list is symmetric, so a
    decoder that confuses raster order with the coded diagonal order decodes
    other pictures. The intra 4x4 luma list is flat, which is the default
    and so is coded as one; each Cr list repeats its Cb list, which is coded
    as a copy; the rest are coded value by value, and each 16x16 and 32x32
    list has a DC factor of its own.
    """
    lines: list[str] = []
    number = 0
    for size in (4, 8, 16, 32):
        side = min(size, 8)
        for prediction in ('INTRA', 'INTER'):
            for component in ('LUMA', 'CHROMAU', 'CHROMAV'):
                if size == 32 and component != 'LUMA':
                    continue
                name = f'{prediction}{size}X{size}_{component}'
                if component != 'CHROMAV':
                    number += 1
                flat = size == 4 and name == 'INTRA4X4_LUMA'
                lines.append(f'{name} =')
                for row in range(side):
                    lines.append(''.join(
                        f'{16 if flat else 11 + number + 2 * row + 3 * column},'
                        for column in range(side)))
                if size >= 16:
                    lines += [f'{name}_DC =', str(9 + number)]
    return '\n'.join(lines) + '\n'


#: Every fixture, by the name of its file.
STREAMS: dict[str, Stream] = {
    'high_bpyramid': x264(
        (120, 68), 40, 'high',
        'bframes=3:b-pyramid=normal:ref=4:weightp=2:weightb=1:slices=2:'
        'keyint=15:min-keyint=15:scenecut=0', mp4=True),
    'high_cqm': x264(
        (64, 48), 20, 'high',
        'bframes=2:ref=2:cqm=jvt:keyint=10:min-keyint=10:scenecut=0'),
    'baseline_cavlc': x264(
        (72, 40), 20, 'baseline', 'ref=3:keyint=8:min-keyint=8:scenecut=0'),
    'open_gop': x264(
        (96, 54), 30, 'high',
        'bframes=2:ref=3:open-gop=1:keyint=10:min-keyint=10:scenecut=0'),
    # x264 codes an interlaced stream as field macroblock pairs (MBAFF), and
    # the checksums are of the two fields woven into a frame.
    'interlaced_mbaff': x264(
        (64, 48), 12, 'high', 'tff=1:bframes=1:keyint=6:min-keyint=6:scenecut=0',
        interlaced=True),
    # VP9. Two passes give hidden alternate reference frames, each in a
    # superframe with the frame shown beside it; several layers of them give
    # show_existing_frame. A tile column needs 256 luma samples, so the tiled
    # stream is wider than the rest.
    'vp9_altref': vpx(
        (122, 70), 40,
        '-b:v 150k -g 20 -auto-alt-ref 1 -lag-in-frames 16 -frame-parallel 0',
        mp4=True, two_pass=True),
    'vp9_show_existing': vpx(
        (106, 58), 40,
        '-b:v 120k -g 20 -auto-alt-ref 6 -lag-in-frames 16 -frame-parallel 0',
        two_pass=True),
    'vp9_tiles': vpx(
        (498, 138), 8,
        '-b:v 300k -g 4 -tile-columns 1 -tile-rows 1 -row-mt 0 -threads 1'),
    'vp9_profile2': vpx((90, 50), 24, '-profile:v 2 -b:v 100k -g 12', depth=10),
    'vp9_segmented': vpx(
        (160, 90), 20, '-b:v 150k -g 10 -aq-mode 1 -frame-parallel 0'),
    'vp9_lossless': vpx((98, 54), 10, '-lossless 1 -g 5 -frame-parallel 0'),
    'vp9_resilient': vpx(
        (100, 60), 20,
        '-b:v 100k -g 10 -error-resilient default -deadline realtime -cpu-used 5'),
    # HEVC. libx265 codes every reference picture set in the slice header and
    # opens its groups of pictures unless told otherwise. The fade is what
    # makes it choose weighted prediction. No picture is under 64 samples
    # either way, the smallest an AMD decoder takes in this codec. The stream
    # with an MP4 leaves out the encoder's information SEI, which ffmpeg would
    # store with the parameter sets and so put back in front of each key
    # picture of the elementary stream.
    'hevc_bpyramid': x265(
        (120, 68), 40,
        'bframes=4:b-pyramid=1:ref=4:weightp=1:weightb=1:open-gop=0:'
        'keyint=20:min-keyint=20:scenecut=0:info=0',
        mp4=True, before=('-vf', 'fade=in:0:14')),
    'hevc_slices_wpp': x265(
        (160, 90), 30,
        'slices=3:wpp=1:ctu=16:min-cu-size=8:bframes=2:ref=2:temporal-layers=1:'
        'tskip=1:amp=1:rect=1:cu-lossless=1:constrained-intra=1:cbqpoffs=2:'
        'crqpoffs=-2:deblock=1,-1:keyint=15:min-keyint=15:scenecut=0'),
    'hevc_scaling_lists': x265(
        (100, 66), 20,
        'scaling-list={lists}:ctu=32:bframes=2:ref=2:aud=1:repeat-headers=1:'
        'hrd=1:vbv-bufsize=500:vbv-maxrate=500:keyint=10:min-keyint=10:scenecut=0',
        files=(('lists', hevc_scaling_lists()),)),
    'hevc_scaling_default': x265(
        (76, 68), 12,
        'scaling-list=default:ctu=16:bframes=1:ref=1:sao=0:signhide=0:'
        'strong-intra-smoothing=0:keyint=6:min-keyint=6:scenecut=0'),
    'hevc_open_gop': x265(
        (98, 70), 30,
        'open-gop=1:bframes=3:ref=3:keyint=10:min-keyint=10:scenecut=0'),
    'hevc_main10': x265(
        (120, 68), 30,
        'bframes=3:b-pyramid=1:ref=3:weightp=1:weightb=1:keyint=15:min-keyint=15:'
        'scenecut=0:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:'
        'range=limited:chromaloc=2', depth=10, before=('-vf', 'fade=in:0:14')),
    # AV1, from three encoders, since each uses a different part of the
    # syntax. libaom hides alternate reference frames and shows them later
    # with show_existing_frame; its film-grain-test option codes one of its
    # test sets of grain parameters. SVT-AV1 switches interpolation filters by
    # block and uses intra block copy, and its irefresh-type=1 opens each
    # group with an intra-only frame. rav1e codes segmentation, loop
    # restoration on every plane and switch frames.
    'av1_altref': av1(
        (120, 68), 40,
        '-c:v libaom-av1 -cpu-used 4 -b:v 150k -g 20 -lag-in-frames 19 -auto-alt-ref 1',
        mp4=True),
    'av1_tiles': av1(
        (160, 90), 20,
        '-c:v libaom-av1 -cpu-used 5 -b:v 200k -g 10 -tile-columns 1 -tile-rows 1 '
        '-aq-mode 1 -aom-params deltaq-mode=1:delta-lf-mode=1:enable-qm=1'),
    'av1_tile_sizes': av1(
        (160, 90), 12,
        '-c:v libaom-av1 -cpu-used 5 -b:v 200k -g 6 '
        '-aom-params tile-width=1,2:tile-height=1,1'),
    'av1_grain': av1(
        (120, 68), 20,
        '-c:v libaom-av1 -cpu-used 5 -b:v 150k -g 10 -aom-params film-grain-test=3'),
    'av1_grain10': av1(
        (96, 54), 20,
        '-c:v libaom-av1 -cpu-used 5 -b:v 150k -g 10 -aom-params film-grain-test=7',
        depth=10),
    'av1_switchable10': av1(
        (120, 68), 20,
        '-c:v libsvtav1 -preset 6 -g 10 -svtav1-params tile-columns=1', depth=10),
    'av1_intra_only': av1(
        (110, 66), 40, '-c:v libsvtav1 -preset 8 -g 12 -svtav1-params irefresh-type=1'),
    # SVT-AV1 codes a frame with super-resolution at whole blocks, so this
    # one stream is not cropped.
    'av1_superres': av1(
        (120, 72), 20,
        '-c:v libsvtav1 -preset 8 -g 10 '
        '-svtav1-params superres-mode=1:superres-denom=12:superres-kf-denom=12'),
    'av1_switch_frames': av1(
        (96, 54), 30,
        '-c:v librav1e -speed 6 -g 30 '
        '-rav1e-params switch_frame_interval=8:low_latency=true'),
    # A source that turns and grows is what makes libaom code global motion.
    'av1_global_motion': av1(
        (160, 90), 20,
        '-vf rotate=0.03*n:fillcolor=gray,zoompan=z=1+0.004*on:d=1:s=160x90 '
        '-c:v libaom-av1 -cpu-used 2 -b:v 200k -g 20 -enable-global-motion 1'),
    'av1_resilient': av1(
        (88, 50), 20,
        '-c:v libaom-av1 -cpu-used 5 -b:v 120k -g 10 -error-resilience default'),
    'av1_monochrome': av1(
        (72, 40), 12, '-pix_fmt gray -c:v libaom-av1 -cpu-used 5 -b:v 80k -g 6',
        monochrome=True),
    'av1_still': av1((64, 36), 1, '-c:v libaom-av1 -cpu-used 5 -still-picture 1'),
}


def run(command: list[str]) -> bytes:
    return subprocess.run(command, check=True, capture_output=True).stdout


def first_pass(name: str, video: list[str], coding: list[str]) -> list[str]:
    """Run the first of two passes; return the options the second reads it with."""
    folder = tempfile.mkdtemp(prefix='decode-fixture-')
    atexit.register(shutil.rmtree, folder, ignore_errors=True)
    log = str(Path(folder) / name)
    run(['ffmpeg', '-v', 'error', '-y', *video, *coding, '-pass', '1',
         '-passlogfile', log, '-f', 'null', '-'])
    return ['-pass', '2', '-passlogfile', log]


def coding_options(stream: Stream) -> tuple[str, ...]:
    """The stream's coding options, with the files they name written."""
    if not stream.files:
        return stream.coding
    folder = Path(tempfile.mkdtemp(prefix='decode-fixture-'))
    atexit.register(shutil.rmtree, folder, ignore_errors=True)
    paths = {}
    for name, text in stream.files:
        paths[name] = folder / name
        paths[name].write_text(text)  # noqa: OGC121 a new file in a folder made for it, read by the one encoder run
    return tuple(option.format(**paths) for option in stream.coding)


def encode(name: str, stream: Stream) -> Path:
    """Write the elementary stream; one with an MP4 goes through its MP4 first."""
    path = DESTINATION / f'{name}{stream.suffix}'
    width, height = stream.size
    video = ['-f', 'lavfi', '-i', f'testsrc2=size={width}x{height}:rate=30']
    coding = ['-frames:v', str(stream.frames), '-pix_fmt', stream.pixel_format,
              *coding_options(stream)]
    if stream.two_pass:
        # The source ends itself in place of ``-frames:v``, which stops a first
        # pass before it writes the statistics that close its log.
        video[-1] += f',trim=end_frame={stream.frames}'
        coding = ['-pix_fmt', stream.pixel_format, *stream.coding]
        coding += first_pass(name, video, coding)
    if not stream.mp4:
        run(['ffmpeg', '-v', 'error', '-y', *video, *coding, str(path)])
        return path
    movie = path.with_suffix('.mp4')
    run(['ffmpeg', '-v', 'error', '-y', *video,
         '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=8000:duration=2',
         '-map', '0:v', '-map', '1:a', *coding, '-c:a', 'aac', '-b:a', '16k',
         '-shortest', str(movie)])
    # The same bitstream, out of the container, so both forms decode alike.
    run(['ffmpeg', '-v', 'error', '-y', '-i', str(movie), '-map', '0:v',
         '-c:v', 'copy', *stream.to_stream, str(path)])
    return path


def decoded_md5s(path: Path, stream: Stream) -> list[str]:
    """The MD5 of each decoded picture, cropped, in display order."""
    if stream.monochrome:
        return monochrome_md5s(path, stream)
    raw = run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'rawvideo',
               '-pix_fmt', stream.reference_format, '-'])
    size = stream.frame_bytes
    if len(raw) % size:
        raise SystemExit(f'{path}: decoded {len(raw)} bytes, not whole frames')
    return [hashlib.md5(raw[start:start + size]).hexdigest()
            for start in range(0, len(raw), size)]


def monochrome_md5s(path: Path, stream: Stream) -> list[str]:
    """The MD5 of each picture of an 8-bit stream with no chroma, as NV12.

    ``ffmpeg`` treats grey as full range and rescales it on the way to NV12,
    so the luma is taken as it was decoded and the chroma plane is written
    here, every sample 128.
    """
    if stream.depth != 8:
        raise SystemExit(f'{path}: monochrome references are written at 8 bits only')
    raw = run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'rawvideo',
               '-pix_fmt', 'gray', '-'])
    width, height = stream.size
    size = width * height
    if len(raw) % size:
        raise SystemExit(f'{path}: decoded {len(raw)} bytes, not whole frames')
    chroma = bytes([128]) * (((width + 1) // 2) * ((height + 1) // 2) * 2)
    return [hashlib.md5(raw[start:start + size] + chroma).hexdigest()
            for start in range(0, len(raw), size)]


def write_table(codec: str, version: str) -> None:
    """Write every stream of `codec` and the table that lists them."""
    tables: dict[str, dict] = {'streams': {}}
    for name, stream in STREAMS.items():
        if stream.codec != codec:
            continue
        path = encode(name, stream)
        table = tables.setdefault(stream.group, {})
        table[name] = {
            'size': list(stream.size),
            'depth': stream.depth,
            'file': path.name,
            'coding': ' '.join(stream.coding),
            'md5': decoded_md5s(path, stream),
        }
        print(f'{path.name}: {path.stat().st_size} bytes, '
              f'{len(table[name]["md5"])} pictures')
    destination = DESTINATION / f'{codec}.json'
    # Renamed over the recorded table once written, so the suite never reads half of one.
    partial = destination.with_name(destination.name + '.partial')
    partial.write_text(json.dumps({'written_by': version, **tables}, indent=2) + '\n')
    partial.replace(destination)


def main(argv: list[str]) -> int:
    if shutil.which('ffmpeg') is None:
        print('ffmpeg is needed to write the fixtures')
        return 2
    known = sorted({stream.codec for stream in STREAMS.values()})
    wanted = argv[1:] or known
    unknown = sorted(set(wanted) - set(known))
    if unknown:
        print(f'no fixtures are listed for {", ".join(unknown)}; '
              f'there are some for {", ".join(known)}')
        return 2
    DESTINATION.mkdir(parents=True, exist_ok=True)
    version = run(['ffmpeg', '-version']).decode().splitlines()[0]
    for codec in wanted:
        write_table(codec, version)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
