"""Write the H.264 streams the decoder's tests read, and what they decode to.

Our own encoders write a narrow slice of H.264: one slice a picture, P frames
only, one reference. A decoder meets everything else a conforming encoder may
produce, so the suite carries streams from x264 that use B-frame pyramids,
weighted prediction, several slices a picture, scaling matrices, CAVLC and
open groups of pictures.

``ffmpeg`` with ``libx264`` writes them, and ``ffmpeg``'s software decoder
records the MD5 of each decoded picture as NV12, in display order. H.264
decoding is exact, so a hardware decoder that is right produces the same bytes.
The streams and ``tests/data/decode/fixtures.json`` are committed, so the suite
needs neither ``ffmpeg`` nor this tool to run. The package itself never runs
``ffmpeg``.

Run it from the project directory::

    python tools/make_decode_fixtures.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

DESTINATION = Path(__file__).parent.parent / 'tests' / 'data' / 'decode'

#: name -> (width, height, frames, x264 parameters, extra ffmpeg options).
#: The sizes are not whole macroblocks, so every stream is cropped.
STREAMS: dict[str, tuple[int, int, int, str, list[str]]] = {
    'high_bpyramid': (
        120, 68, 40,
        'bframes=3:b-pyramid=normal:ref=4:weightp=2:weightb=1:slices=2:'
        'keyint=15:min-keyint=15:scenecut=0',
        ['-profile:v', 'high']),
    'high_cqm': (
        64, 48, 20,
        'bframes=2:ref=2:cqm=jvt:keyint=10:min-keyint=10:scenecut=0',
        ['-profile:v', 'high']),
    'baseline_cavlc': (
        72, 40, 20,
        'ref=3:keyint=8:min-keyint=8:scenecut=0',
        ['-profile:v', 'baseline']),
    'open_gop': (
        96, 54, 30,
        'bframes=2:ref=3:open-gop=1:keyint=10:min-keyint=10:scenecut=0',
        ['-profile:v', 'high']),
}

#: The stream that is also written as MP4, with a short audio track, so the
#: container reader is tested on what ``ffmpeg`` writes: an edit list, signed
#: composition offsets and a second track.
MP4_STREAM = 'high_bpyramid'


def run(command: list[str]) -> bytes:
    return subprocess.run(command, check=True, capture_output=True).stdout


def encode(name: str, width: int, height: int, frames: int, parameters: str,
           extra: list[str]) -> Path:
    """Write the stream as Annex-B; the MP4 stream goes through its MP4 first."""
    path = DESTINATION / f'{name}.h264'
    video = ['-f', 'lavfi', '-i', f'testsrc2=size={width}x{height}:rate=30']
    coding = ['-frames:v', str(frames), '-pix_fmt', 'yuv420p', '-c:v', 'libx264',
              *extra, '-x264-params', parameters]
    if name != MP4_STREAM:
        run(['ffmpeg', '-v', 'error', '-y', *video, *coding, str(path)])
        return path
    movie = path.with_suffix('.mp4')
    run(['ffmpeg', '-v', 'error', '-y', *video,
         '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=8000:duration=2',
         '-map', '0:v', '-map', '1:a', *coding, '-c:a', 'aac', '-b:a', '16k',
         '-shortest', str(movie)])
    # The same bitstream, out of the container, so both forms decode alike.
    run(['ffmpeg', '-v', 'error', '-y', '-i', str(movie), '-map', '0:v',
         '-c:v', 'copy', '-bsf:v', 'h264_mp4toannexb', str(path)])
    return path


def decoded_md5s(path: Path, width: int, height: int) -> list[str]:
    """The MD5 of each decoded picture, as cropped NV12, in display order."""
    raw = run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'rawvideo',
               '-pix_fmt', 'nv12', '-'])
    size = width * height * 3 // 2
    if len(raw) % size:
        raise SystemExit(f'{path}: decoded {len(raw)} bytes, not whole frames')
    return [hashlib.md5(raw[start:start + size]).hexdigest()
            for start in range(0, len(raw), size)]


def main() -> int:
    if shutil.which('ffmpeg') is None:
        print('ffmpeg is needed to write the fixtures')
        return 2
    DESTINATION.mkdir(parents=True, exist_ok=True)
    table = {}
    for name, (width, height, frames, parameters, extra) in STREAMS.items():
        path = encode(name, width, height, frames, parameters, extra)
        table[name] = {
            'size': [width, height],
            'x264': parameters,
            'md5': decoded_md5s(path, width, height),
        }
        print(f'{path.name}: {path.stat().st_size} bytes, '
              f'{len(table[name]["md5"])} pictures')
    version = run(['ffmpeg', '-version']).decode().splitlines()[0]
    destination = DESTINATION / 'fixtures.json'
    # Renamed over the recorded table once written, so the suite never reads half of one.
    partial = destination.with_name(destination.name + '.partial')
    partial.write_text(json.dumps({'written_by': version, 'streams': table},
                                  indent=2) + '\n')
    partial.replace(destination)
    return 0


if __name__ == '__main__':
    sys.exit(main())
