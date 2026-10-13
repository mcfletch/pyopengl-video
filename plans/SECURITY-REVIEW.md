# Security and memory-safety review

A review of every module under `src/pyopengl_video/`, the tools and the
examples, made before the first public release. It records what was found,
what has been fixed, what is still open, and the coding standards that keep the
same classes of defect from coming back.

Date of the review: 2026-10-13.

## Threat model

- A media file is hostile. Every length, offset, count, index, size and table
  entry read from an MP4, an IVF file, an Annex-B stream or a coded bitstream
  is chosen by whoever made the file.
- A value parsed from a stream ends in one of three places: a Python
  allocation or loop, a fixed-width field of a structure a GPU driver reads,
  or a length handed to native code. The driver indexes its own arrays with
  what it is given and checks little of it, so a value that leaves a parser
  outside its specification range is a defect whether or not Python would
  later raise.
- A driver's answers are trusted for meaning and not for bounds: a length or an
  offset a driver reports is compared with the buffer this process allocated
  before anything is read through it.
- A caller's arguments are not hostile, and are still validated, because
  ctypes stores an integer that does not fit a field by dropping its high bits
  and raises nothing.

## How it was done

Seven reviews, one per area, each reading its files in full: the containers,
the H.264 and HEVC parsers and picture buffers, the AV1 and VP9 parsers, the
VA-API device and session layer, the VA-API codec control layers, the NVENC and
NVDEC bindings, and the Windows and oneVPL bindings. Each reproduced what it
could:

- the parsers and containers by mutation fuzzing of the fixtures in
  `tests/data/decode/`, with range assertions on every successful parse
  (about 430,000 container inputs, 360,000 H.264 and HEVC inputs);
- the control layers against a recording device, asserting on the buffers
  handed to `submit` (about 156,000 pictures);
- the VA-API structures against the libva 2.20.0 headers with a C compiler
  (45 sizes, 598 offsets and 390 bit-fields, no mismatch);
- EGL, OpenGL and NVDEC on the RTX 3060 Ti in this container;
- the Windows code by reading and by simulation on Linux, since nothing here
  runs it.

libva has no driver for the GPU in this container, so what a VA driver does
with a bad value was not observed. "Reaches the driver" below means the value
was read back from the ctypes buffer handed to the device.

Native crashes found: three ways to a segmentation fault through NVDEC's AV1
decoding (N1 to N3), one through an NVENC input handle (N5), one in ctypes for
a null function-table entry (N9), a hang at exit (N4), and a SIGBUS in the
Annex-B reader (C3). H.264, HEVC and VP9 through NVDEC, and every codec through
the VA-API control layers, raised the package's errors and nothing else.

## Status

Each finding has an identifier. **Fixed** findings have a test that holds them, and the commit that
fixed each names its identifiers; **Open** findings are listed again under
*Still open* where they need something this container lacks.

### D. A size a container states that the stream does not have

The question asked of the review: what happens when a container header says
one picture size and the stream is another.

The decoders ignore the stated size. `VAAPIDecoder` and `NVDECDecoder` accept
`size` for the registry and `del` it; surfaces, contexts and plane textures are
sized from the sequence header the bitstream carries, and a sequence of another
size rebuilds them. That part is sound.

| Id | Where | Finding | Status |
|---|---|---|---|
| D1 | `playback.py` `Player._new_texture` | Output textures are made at `movie.size`, the container's stated size, and `NV12Converter.draw` sets the viewport to the decoded picture's size. A 120x68 stream in an MP4 stating 640x480 gives a 640x480 texture with the picture in one corner and the other 1.2 MB never written (texture storage is undefined until written); stating 16x16 crops the picture; stating 0x0 raises a bare `GLError`. A stated 8192x8192 allocates 256 MB a texture for a thumbnail-sized stream. Reproduced on NVDEC. | Fixed |
| D2 | `playback.py` | A stream that changes size at a key frame keeps the output textures of the old size, for the same reason. | Fixed |
| D3 | `convert.py` fragment shader | `texelFetch(luma, …)` is undefined outside the texture. `visible` is not checked against `texture_size`, and `draw(size=…)` larger than `visible` fetches at negative coordinates. | Fixed |
| D4 | `h264_parse.py`, `hevc_parse.py` cropping | A cropping window larger than the picture gives a negative visible size, which reaches `session.output` (`visible=(0, 0, -136, -136)`). | Fixed |
| D5 | `nvdec/decoder.py` `Sequence.of` | `display_area` from NVIDIA's parser is not checked to lie inside the coded size before it becomes `visible`. | Fixed |
| D6 | `vaapi/av1_decode.py`, `av1_parse.py` | An AV1 frame header may state a frame size larger than the sequence header's maximum. Surfaces are 64x48 and the driver is told 4096x2304. | Fixed |
| D7 | `vaapi/hevc_decode.py` | A later slice segment of a picture may name a PPS whose SPS is another size; it is appended to the picture under way. | Fixed |
| D8 | `vaapi/vp9_decode.py` | A non-key frame may state another profile than its key frame; a 65536-wide key frame is written to a 16-bit field as 0. | Fixed |
| D9 | `vaapi/encoder.py` `register` | A caller's texture is exported at the encoder's size without asking OpenGL what size it is: a 16x16 texture is described to libva as 4096x4096. | Fixed |
| D10 | `mp4.py`, `ivf.py` | The stated width and height are never held to 1 or more; (0, 0) passes `Backend.supports`. | Fixed |

### C. Containers (`mp4.py`, `ivf.py`, `annexb.py`, `obu.py`, `playback.py`)

No path hands native code a length larger than the bytes behind it.

| Id | Where | Finding | Status |
|---|---|---|---|
| C1 | `mp4.py` sample tables | `stsz`, `stts`, `ctts`, `stsc` counts size allocations and loops with no check against the box. A 553-byte file takes 7.6 GB, or loops until killed. | Fixed |
| C2 | `annexb.py` with the parsers' unbounded ids | Every access unit snapshots a tuple of every parameter set seen: a 340 KB file indexes to 528 MB, quadratic in the file. | Fixed |
| C3 | `annexb.py` | Samples are read through an `mmap`; a file shortened by another process after opening kills this one with SIGBUS. | Fixed |
| C4 | `mp4.py` `read_at`, `ivf.py` | A file-stated size or offset goes straight to `seek` and `read`: `MemoryError`, `OverflowError` and `ValueError` escape, or 4 GB is allocated up front. | Fixed |
| C5 | `ivf.py` | A frame size is not checked against the file, and `sample()` returns a short read as a whole sample; a frame stating 4 GB swallows every later frame as picture data. Header `length` under 32 is accepted; `sample(-1)` wraps. | Fixed |
| C6 | `mp4.py` | Slices sized by a file length are taken as full-length: NAL length prefixes, `avcC`/`hvcC` records, `stz2`. | Fixed |
| C7 | `mp4.py` `stsc` | `first_chunk` 0 indexes `chunks[-1]`. | Fixed |
| C8 | `mp4.py` `stss` | Sync sample numbers are not range-checked: 0 becomes index -1. | Fixed |
| C9 | `mp4.py`, `annexb.py`, `playback.py` | `struct.error`, `IndexError`, `StopIteration`, `ValueError`, `OverflowError`, `MemoryError` and `ZeroDivisionError` escape the readers. `MP4Error`, `IVFError` and `StreamError` do not derive from `VideoError`, which the documentation calls the base of every error raised. | Fixed |
| C10 | `mp4.py`, `ivf.py` | `seek` then `read` on a shared file object: two threads on one reader get the wrong bytes at the right length. | Fixed |
| C11 | `mp4.py` writer | 32-bit fields overflow at `close()`, leaving a recording with no `moov`: a first timestamp of 2^31, a span over 2^32 ticks, `timescale=0`, a width of 70000. | Fixed |
| C12 | `mp4.py` writer | `__exit__` raises "no samples" over the exception already propagating. | Fixed |
| C13 | `playback.py` `Player` | A converter or a `flush()` that raises strands the frames not yet accepted; `close()` stops at the first failure. | Fixed |
| C14 | `playback.py`, `annexb.py`, `ivf.py`, `mp4.py` | A reader or writer opened and then failing is left to the garbage collector. | Fixed |

What the fixes decided, for a maintainer to weigh:

- `Player` makes each output texture at the decoded frame's size and retires
  one of another size; `Player.size` is the size of the frame shown, and
  `Movie.size` stays what the container states. A stated size no backend
  reaches is retried with no size, so a small stream in a container stating
  65535x65535 plays.
- `container.py` holds the one bounds-checked `Cursor` and the positioned
  `FileBytes` reads every reader uses. `MP4Error` and `IVFError` derive from
  `ContainerError(VideoError, ValueError)`.
- The Annex-B reader holds no `mmap`: it indexes through a one-megabyte window
  and reads samples with `pread`.
- The MP4 writer refuses, at `write()`, a timestamp `close()` could not pack:
  the first under 2^30, none below -2^30, none more than 2^30 behind the latest
  or 2^32-1 ahead. A recording with no samples is removed; one with samples and
  no parameter sets is left as `mdat` and `close()` raises.
- An IVF frame that runs past the end of the file ends the index before it.
- MP4 sample tables are built for a track when it is first read, so a table
  error in a track nobody reads is never raised.

Remaining limit: an Annex-B stream that changes a parameter set before every
picture holds a tuple of sets for each, up to about 2.4 KB a picture.

`tests/test_fuzz_containers.py` runs 3,500 seeded mutations in the suite.

### H. H.264 and HEVC parsing and picture buffers

The parsers held almost no syntax element to its range. The mechanism under
most findings is the one in the threat model: a value that does not fit a
ctypes field is masked.

| Id | Where | Finding | Status |
|---|---|---|---|
| H1 | `h264_parse.py`, `h264_dpb.py` | `num_ref_idx_l0/l1_active_minus1` unbounded: 200 reaches the driver beside a 32-entry list; 2^28 allocates 2 GB from a few dozen bytes; 40 with weighted prediction raises `IndexError`. | Fixed |
| H2 | `h264_parse.py` slice header | `first_mb_in_slice`, `cabac_init_idc`, `slice_qp_delta`, `disable_deblocking_filter_idc`, the deblocking offsets, weight denominators, weights and offsets reach the driver unchecked or masked. | Fixed |
| H3 | `hevc_parse.py` slice header | `collocated_ref_idx`, `slice_segment_address`, `five_minus_max_num_merge_cand`, QP and deblocking offsets, weights reach the driver unchecked or masked. | Fixed |
| H4 | `hevc_parse.py` SPS and PPS | Picture size (0x0 accepted; 65600 asked of the surface and 64 told to the driver), block-size logarithms, PCM sizes, `num_long_term_ref_pics_sps`, `init_qp_minus26`, `diff_cu_qp_delta_depth` unchecked. | Fixed |
| H5 | `hevc_parse.py` `_tile_sizes` | A uniform tile grid with more tiles than CTBs gives zero-width tiles, written as `column_width_minus1 = 65535`. | Fixed |
| H6 | `h264_dpb.py` `_fill_gaps` | One iteration for each skipped `frame_num`, and `log2_max_frame_num_minus4` is unchecked: a 45-byte stream does not return. | Fixed |
| H7 | `h264_decode.py` | `slice_data_bit_offset` is 16 bits: a header of 68024 bits is written as 2488. | Fixed |
| H8 | `h264_parse.py` | Weight denominators are used as shift counts: a 41-byte unit allocates 2 GB. | Fixed |
| H9 | both | DPB size and reorder depth come from the stream: 100,001 surfaces are asked for. | Fixed |
| H10 | `hevc_dpb.py` | For a BLA picture with P slices a stand-in reference is chosen and then dropped from `held()`: the picture is decoded into the surface it predicts from. | Fixed |
| H11 | `h264_dpb.py`, `h264_decode.py` | A P or B slice with no reference is submitted with `VA_INVALID_SURFACE` in an active list entry. | Fixed |
| H12 | `h264_parse.py` SPS | `pic_width_in_mbs_minus1`, `log2_max_frame_num_minus4` and others are masked on their way to the driver. | Fixed |
| H13 | `h264_parse.py`, `h264_dpb.py` | `long_term_frame_idx` unbounded; `_max_long_term_frame_idx` written and never read. | Fixed |
| H14 | both | Parameter set ids are unbounded dictionary keys: 20,000 sets are held. | Fixed |
| H15 | `h264.py` `ue` | 32 leading zeroes are accepted; the specification's largest code has 31. | Fixed |
| H16 | `hevc_parse.py` | An inter-predicted reference picture set can exceed 16 pictures; only the explicit form is checked. | Fixed |
| H17 | `h264_decode.py:371`, `vaapi/decoder.py:69` | `AssertionError` from `flush()` before any picture; `as_decoder_error` names three exception types and lets `IndexError`, `MemoryError` and `AssertionError` through. | Fixed |
| H18 | `h264.py`, `hevc.py` writers | `qp`, `max_num_ref_frames`, `idr_pic_id`, a zero frame-rate term and `nal_unit` type and temporal id are written unchecked. | Fixed |

The ranges every element is now held to, with the clause for each, are in
*Appendix A*. No bare `ue()` or `se()` remains in either parser:
`ue_in` and `se_in` state the range with the read.

What the fixes decided, for a maintainer to weigh:

- H11: a P or B slice with no decoded reference at all is refused. A list that
  is only partly empty has a decoded frame standing in for the missing entry,
  on the reading that 8.2.4.2 allows "no reference picture" entries in a
  conforming stream, as just after an IDR whose PPS default count exceeds the
  frames held. That reading is from memory of the clause and wants checking
  against the text; refusing instead is a small change in `_decoded`
  (`h264_dpb.py`).
- H13: `_max_long_term_frame_idx` is removed and `long_term_frame_idx` is
  held below `max_num_ref_frames` at the parser. Enforcing the stored maximum
  would refuse valid streams after a seek past the MMCO 4 that set it.
- `max_dec_frame_buffering` below `max_num_ref_frames` is refused (E.2.1). An
  encoder that writes it wrongly no longer plays.
- Every value the H.264 and HEVC control layers give libva goes through
  `store` or `store_each`, which read the field back and raise on a
  difference; a syntax-tree test fails for a builder that assigns a field
  directly.

Found while fixing, and fixed: an H.264 frame pushed out by a `frame_num` gap
was shown after the next picture's surface was chosen, so that picture could be
decoded into a surface still waiting to be shown.

`tests/test_fuzz_h264.py` and `tests/test_fuzz_hevc.py` run 8,000 and 6,000
seeded mutations in the suite. 100,000 more of each on other seeds found
nothing.

### V. VA-API device and session layer

| Id | Where | Finding | Status |
|---|---|---|---|
| V1 | `vaapi/decoder.py` `_Session.activate` | The old pool is retired before the new one is made; when making it fails, `self.pool` is the destroyed pool, and the next picture is submitted to a destroyed context and surfaces. Stream-triggerable: a size the device refuses, then the first size again. | Fixed |
| V2 | `linux/dmabuf.py`, `inputs.py`, `convert.py` | OpenGL objects are deleted by name in whichever context is current. Reproduced on EGL: releasing an imported texture with another context current deleted that context's texture of the same name. | Fixed |
| V3 | `vaapi/encoder.py` | A failed submission leaves the slot counters advanced: the next picture reuses the coded buffer and source surface of the one in flight. | Fixed |
| V4 | `vaapi/encoder.py`, `vaapi/decoder.py`, `linux/dmabuf.py` | `close()` sets `_closed` and stops at the first exception; the display and the render-node descriptor are never released. | Fixed |
| V5 | `vaapi/encoder.py` `_collect` | The coded buffer is read on the driver's word: no bound on a segment's size, no limit on the chain, no status bit tested, so an overflowed picture is a truncated packet. | Fixed |
| V6 | `vaapi/device.py` | A failed `vaInitialize` never calls `vaTerminate`: 927 bytes a failed open, which is every open on a machine with libva and no driver. | Fixed |
| V7 | `linux/dmabuf.py`, `inputs.py` | `import_texture` and `create_rgba_texture` leave `GL_TEXTURE_2D` bound to 0; the EGL image leaks if texture creation raises. | Fixed |
| V8 | `vaapi/device.py` `picture` | `vaRenderPicture` failing leaves a begun picture with no `vaEndPicture`. | Fixed |
| V9 | `vaapi/encoder.py`, `vaapi/decoder.py`, `control.py` | Dimensions, bit rate, group length and frame rate are not validated before ctypes: `VARectangle(…, 70000, 70000)` stores 4464; a frame rate of `(90000, 3003)` is packed as 24464/3003; a bit rate of -1 is 4294967295. `ValueError` and `ZeroDivisionError` escape `H264Encoding` and `AV1Encoding`. | Fixed |
| V10 | `tests/test_va_abi.py` | The coverage test skips `_`-prefixed names and every `Union`; bit-field positions are recorded for three unions only. | Fixed |
| V11 | `inputs.py` | `glTexImage2D(…, None)` reads from a bound pixel-unpack buffer. | Fixed |
| V12 | `vaapi/decoder.py` | Decode surfaces are imported as textures with no clear; a picture whose slices do not cover it shows whatever the driver left. Unconfirmed without a driver. | Open |

What the fixes decided, for a maintainer to weigh:

- An object that owns OpenGL names records the context it was made in
  (`glcontext.Owner`) and refuses to delete them in any other, a context that
  shares objects with it included: EGL has no query for which contexts share.
  Everything that is not OpenGL is released first, and closing again in the
  right context finishes.
- A decoder asks for at most 18 surfaces for its own pictures (AV1 with film
  grain: nine frames of two surfaces), and `output_slots` is 1 to 64.
- An encoder whose submission fails after the driver has the picture stops:
  `encode` is refused, `flush` and `close` work.
- After a format a device refuses, the session holds no pool and refuses every
  picture until a format is activated.
- `tests/va_abi.json` records every union, nested member and bit-field image,
  from the libva 2.20 headers.

Sound: all 31 libva entry points carry `argtypes` and `restype`; every
structure handed to the driver is built fresh for the call; every exported
file descriptor is closed exactly once on every path exercised.

### L. VA-API codec control layers

The findings that are a parser's range are listed under H and A. These are the
control layers' own.

| Id | Where | Finding | Status |
|---|---|---|---|
| L1 | `av1_decode.py` | `self._current` is published with a surface before its picture parameters are built. When building raises, the next lone tile group is submitted with no picture parameter buffer. | Fixed |
| L2 | `av1_parse.py`, `av1_decode.py` | Film grain point counts are read as 4 bits and written into arrays of 14 and 10; the tile grid has no 64x64 limit on the non-uniform path; `context_update_tile_id` is not below the tile count. | Fixed |
| L3 | `av1_parse.py` `_tile_info` | `ZeroDivisionError` and `ValueError` for a reference slot never written. | Fixed |
| L4 | every `*_decode.py` | A parsed value is assigned to a ctypes field with nothing reading it back. | Fixed |

### W. Windows and oneVPL

Nothing here could be run. Verified by simulation on Linux (now
`tests/test_vpl_session.py` over the stand-ins in `tests/vpl_runtime.py`) and
by reading against the published interface definitions. Every COM vtable index
and both interface identifiers match the published ones.

| Id | Where | Finding | Status |
|---|---|---|---|
| W1 | `vpl/encoder.py` `close`, `__del__` | `close()` stopped at the first failure with the session open; the destructor ran `close()`, OpenGL calls included, on whichever thread and context the collector was in. | Fixed |
| W2 | `vpl/encoder.py` `_submit` | A `SyncOperation` time-out (`MFX_WRN_IN_EXECUTION`) was read as success, and the buffer the runtime was still writing was returned as a packet. | Fixed |
| W3 | `vpl/encoder.py` `encode` | One `mfxFrameSurface1` was rewritten for every submission while the runtime still held its address; `Data.Locked` was never read. | Fixed |
| W4 | `vpl/encoder.py` `_submit` | The `MFX_WRN_DEVICE_BUSY` retry had no deadline. | Fixed |
| W5 | `vpl/encoder.py` `_packet` | `string_at` trusted the runtime's `DataOffset` and `DataLength`. | Fixed |
| W6 | `vpl/encoder.py` allocator | An exception in a callback returned whatever was in the result slot as a status. | Fixed |
| W7 | `vpl/encoder.py` allocator | `GetHDL` answered success with a null texture for one that had been closed; a partly failed allocation stayed in `owned`. | Fixed |
| W8 | `vpl/encoder.py`, `windows/d3d11.py` | Width, height, group length, B-frame count, bit rate and timestamp were written to 16- and 32-bit fields unchecked; `fps=(n, 0)` raised `ZeroDivisionError`. | Fixed |
| W9 | `vpl/api.py` | A 32-bit process was not refused, and the layouts are those of a 64-bit build. | Fixed |
| W10 | `windows/d3d11.py` | `CreateDXGIFactory1` and `D3D11CreateDevice` were called with no declared types. | Fixed |
| W11 | `windows/interop.py` | `ctypes.GetLastError()` is read after a call PyOpenGL dispatched. | Open |
| W12 | `windows/interop.py` | A texture could be closed while locked for drawing, locked after it was closed, and encoded while held. | Fixed |
| W13 | `windows/d3d11.py`, `windows/interop.py` | Adapter interfaces leaked when `GetDesc1` failed; a registration leaked when the framebuffer could not be made. | Fixed |
| W14 | `vpl/encoder.py` | `rate_control='constqp'` wrote bit rates into the quantiser fields. | Fixed |
| W15 | `vpl/encoder.py` | The session was not tied to an Intel adapter. | Fixed |
| W16 | `tests/vpl_abi.json` | Nine anonymous unions and members have no header-derived offset; no Direct3D structure or vtable index has a test. | Open |

Sound: no library is loaded from the working directory or `PATH` (Python 3.10
and later load a bare name with `LOAD_LIBRARY_SEARCH_DEFAULT_DIRS`); every
`subprocess` call in the tools is a list with no shell; importing every module
on Linux makes no native call.

### A. AV1 and VP9 parsing

200,000 mutated AV1 streams (4.2 million submitted pictures) and 160,000 VP9
streams (3.7 million) through `AV1Decoding` and `VP9Decoding` over a stand-in
session. Every tile offset and size submitted lay inside its buffer, and no
invalid reference surface was submitted. VP9 raised nothing but `StreamError`
and `DecoderError`.

| Id | Where | Finding | Status |
|---|---|---|---|
| A1 | `av1_parse.py` `_tile_info` | The stated tile grid has no limit of 64 columns and rows: a 16384-wide frame gives 256 columns, written to an 8-bit field as 0. An inter frame at the start of a stream is parsed before any size limit applies: 1024x1024 tiles in a 2 MB sample had not returned after 150 s. | Fixed |
| A2 | `av1_parse.py`, `av1_decode.py` | A frame size is never held to the sequence header's maximum (D6). | Fixed |
| A3 | `av1_parse.py` `_color_config` | Profile 0 with the sRGB and identity triple is parsed as 4:4:4 and submitted under the Main profile. | Fixed |
| A4 | `av1_decode.py` | A sequence header may be replaced with no key frame, and between a frame header and its tile group: a monochrome sequence over a 4:2:0 reference is submitted. | Fixed |
| A5 | `bits.py`, `av1_parse.py` `parse_tile_group` | `BitReader` holds the whole OBU as one integer and each read shifts it, so reading tile sizes is quadratic in the sample: 40 s for a 16 MB frame with the legal 64x64 tiles. | Fixed |
| A6 | `vp9_parse.py`, `vp9_decode.py` | An inter frame's profile is not compared with its key frame's (D8). | Fixed |
| A7 | `vp9_decode.py` | A dimension of 65536 is written to a 16-bit field as 0 (D8). | Fixed |
| A8 | `vp9_parse.py` `_tile_info` | `tile_cols_log2` can pass 6. | Fixed |
| A9 | `vp9_parse.py` | A frame with no tile bytes, or a `tile_size` past the end of the frame, is submitted. | Fixed |
| A10 | `av1_parse.py` | `context_update_tile_id` is not below the tile count (L2). | Fixed |
| A11 | `av1_parse.py` `_film_grain_params` | Point counts above 14 and 10 raise `IndexError` at the ctypes array; point values that do not increase, a `film_grain_params_ref_idx` matching no reference, and unequal chroma point counts in 4:2:0 reach the driver (L2). | Fixed |
| A12 | `av1_parse.py` | A frame sized from a slot never written raises `ValueError` or `ZeroDivisionError` (L3). | Fixed |
| A13 | `bits.py` | A negative count or position is not refused; `leb128()` reads from an unaligned position; `read_leb128` accepts more than 32 bits. No call site in the package reaches these with a stream's value. | Fixed |
| A14 | `av1.py` writer | `base_q_idx = 0` writes the syntax of a lossy frame; a zero frame-rate term divides by zero; `BitWriter.su` wraps. | Fixed |

What the fixes decided, for a maintainer to weigh:

- A changed AV1 sequence header must be followed by a shown key frame while any
  reference slot holds a frame, and is refused between a frame header and its
  tile groups. Headers are compared as parsed, leaving out the operating
  parameters, which the specification lets differ within a sequence.
- An AV1 inter frame's references must match it in bit depth, sampling and
  monochrome and lie within the 2x and 16x scaling limits; an intra-only frame
  that refreshes all eight slots, a `show_existing_frame` of a frame not marked
  showable, a `seq_profile` above 2 and a frame id over 16 bits are refused.
- `BitReader` reads each field from the bytes it spans, so a read costs the
  same wherever it falls in the sample.
- libva's AV1 tile-size arrays hold 63 entries. With the legal 64 tiles the
  last size is not stored, on the understanding that a driver takes the last
  tile as what is left of the frame; that wants checking against a driver.

`tests/test_fuzz_av1.py` and `tests/test_fuzz_vp9.py` run 2,500 and 4,000
seeded mutations in the suite. 40,000 more of each on other seeds, about
870,000 pictures a codec, found nothing.

### N. NVENC and NVDEC

Run on the RTX 3060 Ti here, driver 580.178.04, NVENC API 13.0. About 3,000
mutated H.264, HEVC and VP9 streams through NVDEC, on both transfers, raised
the package's errors and nothing else. AV1 did not survive: every batch of
mutated AV1 ended in a segmentation fault inside `libnvcuvid` within five
cases. The saved inputs and the backtraces are kept with the review's scratch
files and are named in the tests that hold each fix.

| Id | Where | Finding | Status |
|---|---|---|---|
| N1 | `nvdec/decoder.py` `_on_decode` | **Segmentation fault.** An AV1 stream whose first decoded sample is not a key frame (after `reset()`, from a truncated file, or from a container that misstates `keyframe`) makes NVIDIA's parser build a picture whose reference indexes are 0xFF or name a picture the hardware never decoded; the callback passes it to `cuvidDecodePicture` unchanged and the driver dereferences it. 14 of 15 fixture and start-point combinations crash, with no mutation at all. | Fixed |
| N2 | `nvdec/decoder.py` `_on_decode` | **Segmentation fault.** A truncated AV1 temporal unit gives `nBitstreamDataLen = 0xFFFFFFF4`, and `cuvidDecodePicture` copies that many bytes. | Fixed |
| N3 | `cuvidParseVideoData` | **Segmentation fault inside NVIDIA's parser**, with no callback of this package on the stack: one flipped bit in an AV1 sequence header (`order_hint_bits` 7 to 5) makes it read at the payload pointer plus a 32-bit offset from the stream. `describe()` and `obu.split` accept the packet. | Fixed |
| N4 | `nvenc/encoder.py` `close`, `__del__` | **The process hangs at exit.** `close()` sets `_closed` and then releases with no `try`: a failure (OpenGL already torn down at interpreter shutdown, or no context current) skips `nvEncDestroyEncoder` for good, and the driver's exit handler then waits on a worker thread that never finishes. | Fixed |
| N5 | `nvenc/encoder.py` `encode` | **Segmentation fault.** A handle another encoder registered, or one from a closed encoder, is passed to `nvEncMapInputResource`. | Fixed |
| N6 | `nvdec/decoder.py` `_show`, `copy_out` | `picture_index` is not checked against the surfaces the decoder was made with. When a sequence keeps its size and changes its buffer count, the parser's queued pictures are mapped from the new decoder: the caller is lent a frame of driver fill, 43 times in 550 cases. | Fixed |
| N7 | `nvdec/decoder.py` `_open` | A refused sequence leaves the old session in place, and the new sequence's pictures are handed to the old decoder. | Fixed |
| N8 | `nvenc/encoder.py` `encode` | A submission is queued before the status of `nvEncEncodePicture` is read; after a failure the next call returns an empty packet with an earlier timestamp. | Fixed |
| N9 | `nvenc/api.py` | A function-table entry is wrapped in its prototype with no test for null; calling a null one is a segmentation fault in ctypes. A driver older than the header leaves the newer entries null. | Fixed |
| N10 | `nvdec/decoder.py` | Plane textures are made and deleted with no record of their context (the hazard V2 fixed elsewhere): with no context current `close()` leaks them silently, and the host-memory transfer returns a frame whose textures were never made. | Fixed |
| N11 | `nvenc/encoder.py` | Output buffers of pending submissions are never destroyed at close; a `_read` or `_acquire_output` that raises leaves a mapping behind. | Fixed |
| N12 | `nvenc/encoder.py` | `fps=(1, 0)` raises `ZeroDivisionError`; `bframes=-10` gives `input_slots = -8`; a bit rate or group length of 2^40 is truncated into a 32-bit field. | Fixed |
| N13 | `tests/nvenc_abi.json`, `tests/nvdec_abi.json` | Bit-field positions are not recorded: 107 bit-fields in nine structures. | Fixed |
| N14 | `nvdec/decoder.py` `Upload.copy` | `GL_UNPACK_SWAP_BYTES` is not reset, so a renderer that left it set gets 10-bit uploads byte-swapped. | Fixed |

How the crashes were closed, and what was measured:

- `nvdec/guard.py` reads each picture NVIDIA's parser builds, in the decode
  callback, before the hardware has it: the coded length is held to the bytes
  fed, the picture index to the surfaces made, and every reference to the
  pictures this session's decoder has decoded. It covers all four codecs. In
  3,000 mutated H.264, HEVC and VP9 streams it refused 496 pictures for a
  reference and 7 for a length.
- `nvdec/av1.py` `Gate` reads every AV1 sample with this package's own parser
  before NVIDIA's sees it, requires a shown key frame after open, reset or a
  refusal, and hands on only the OBUs it read, rewritten in canonical form:
  metadata, padding and reserved OBUs are dropped. It holds no OpenGL or CUDA
  and is tested without a GPU.
- AV1 crashes: every batch within five cases before; 0 in 5,600 cases (66,085
  frames) and 0 in a confirming 2,400 after. H.264, HEVC and VP9: 0 in 3,000
  after.
- `CUVIDAV1PICPARAMS` and the H.264, HEVC and VP9 picture parameters are
  declared whole and recorded from NVIDIA's headers (SDK 13.1.15): 16
  structures, 351 nested members, 194 bit-fields.

Found while fixing, and fixed:

- **Segmentation fault** in `nvEncDestroyEncoder` when an encoder was closed
  with pictures held for reordering and their output buffers already destroyed.
  `close()` sends the end of stream first.
- **Segmentation fault** in NVIDIA's parser, about once in thousands of feeds,
  for an empty metadata OBU ahead of a frame header. The gate drops metadata.
- Headers NVIDIA's parser reads differently from the specification: a reduced
  still-picture header in a sequence that is not one still picture, and a stray
  bit after a sequence header's trailing bits, are read as 4:4:4 at 16 bits.
  Both are refused, the first in `av1_parse.py` as well.

For a maintainer to weigh:

- **One crash is mitigated and not explained.** A segmentation fault inside
  `cuvidDestroyDecoder`, with a second destroy and a `cuvidCreateDecoder`
  running beside it, was seen once in about 12,000 cases and could not be
  reproduced in 9,000 concurrent create and destroy rounds or 8,400 cases under
  gdb, so there is no native frame for it. A lock now serialises decoder
  creation and destruction (`_Session.lifecycle`), and it has not recurred in
  about 13,600 cases since, which at that rate is weak evidence. The lock makes
  a decoder opened as the last one closes wait about 260 ms.
- **What the gate cannot see.** AV1 tile data goes to the hardware unread, and
  a header NVIDIA's parser reads differently in a way not yet found would pass.
  NVDEC AV1 survived the runs above; it has not been proven safe for hostile
  input, and nothing short of driving the hardware from this package's own
  parser would prove it.
- A picture NVIDIA's parser shows that the current decoder never decoded is
  dropped with a warning, where it was lent as a frame of driver fill.
- The trailing-bit and alignment checks for AV1 sequence headers, frame headers
  and tile groups live in the gate. `av1_parse.py` is the better home, with a
  `header_bits` on `SequenceHeader`, so that the VA-API path has them too.

`tests/test_fuzz_nvdec.py` runs nine batches of seeded mutations, each in a
child process, so that a crash is a failed assertion.

Sound: all 43 CUDA, cuvid and NVENC entry points carry their types; every NVENC
structure passed has its `version` set; the three parser callbacks are held for
the parser's life, trap every exception and re-raise it after the parser
returns, and keep nothing the parser lent them; every copy out of a mapped
picture is sized from the same sequence as its destination; the CUDA context
stack is balanced after decode, a raising flush, reset and close.

## Still open

Four findings are open, each for something this container lacks:

| Id | Needs |
|---|---|
| W11 | PyOpenGL's WGL binding loaded with `use_last_error=True` |
| W16 | The oneVPL headers and a compiler where the recorder runs; a Windows machine for the Direct3D structures and vtable indexes |
| V12 | A VA-API driver, to see what an uncovered surface holds |
| N3, in part | See *What the gate cannot see* under N |

And the Windows and VA-API hardware paths have still only been read and
simulated: the fixes to `vpl/`, `windows/` and the VA-API device layer are
tested over stand-ins, and want one run on an Intel Windows machine and one on
a machine with a VA-API driver before a release.

In detail:

- W11 needs PyOpenGL's WGL binding loaded with `use_last_error=True`, which is
  a change to PyOpenGL. Until then `InteropError.result` for a WGL call
  describes the failure and is not to be branched on;
  `windows.interop.last_error` says so.
- W16 needs the oneVPL headers and a compiler where the recorder runs. (V10 and
  N13 were closed with recordings from the libva 2.20 and NVIDIA SDK 13.1.15
  headers.)
- V12 needs a VA-API driver to observe what an uncovered surface holds.

## Coding standards

These are the rules the findings reduce to. Each is written so that a reviewer
or a static check can decide whether a line breaks it.

### 1. A value read from a stream is checked where it is read

1. Every syntax element is read through a call that states its range:
   `reader.ue_in(low, high, name)`, `reader.se_in(…)`, or `reader.u(n)` where
   the width is the whole range. A bare `ue()` or `se()` in a `*_parse.py`
   module is a defect.
2. The range is the specification's, and the clause is cited beside it. Where
   the specification bounds a value by a derived quantity (`PicSizeInCtbsY`,
   `num_ref_idx_active_minus1`, the tile count), the check uses that quantity
   and not the field's bit width.
3. Both arms of a two-armed derivation are checked: the explicit and the
   inter-predicted reference picture set, the uniform and the stated tile
   grid.
4. A stream-supplied id is held to its range before it is a dictionary key.
5. A parser raises the package's own error for anything a stream can cause. No
   `assert` guards a state a stream can reach. A boundary that translates
   exceptions translates every exception type, not a list of three.

### 2. Nothing sized by the stream is allocated or looped over unchecked

6. `[x] * n`, `range(n)`, `1 << n`, `reader.u(n)`, `file.read(n)` and
   `file.seek(n)` take an `n` from a file only after it has been compared with
   what the enclosing box, the file or the specification allows.
7. A loop's iteration count is a count of bits or bytes consumed, or a value
   already bounded. A loop over a derived quantity such as `MaxFrameNum` is a
   defect.
8. `data[a:a + n]` with a file-supplied `n` is followed by a check that the
   slice is `n` long. A slice never raises, so a short one is silent.
9. Every container has one cursor type with `u8`, `u16`, `u32`, `u64` and
   `take(n)` that raise the container's error. `struct.unpack_from` and
   `buffer[i]` on file bytes appear nowhere else.
10. Division by a file field is preceded by a zero check where the field is
    read.

### 3. A value written to a fixed-width field is proven to fit

11. A parsed or caller-supplied integer reaches a ctypes field, bit-field or
    narrower-than-64-bit argument only through a check: `bounds.whole`,
    `bounds.frame_size`, `bounds.frame_rate`, or a store that reads the value
    back and raises on a difference.
12. `value - 1` written to an unsigned field is preceded by proof that
    `value >= 1`.
13. A count field and the array it counts are filled by the same loop under
    the same bound, and unused reference entries hold the interface's invalid
    marker, never zero.
14. Two values packed into one integer are each masked and range-checked.

### 4. A length from native code is checked against our allocation

15. `ctypes.string_at`, `memmove`, `from_address`, and any `addressof(x) + n`
    take an offset and a length only after `bounds.span` has compared them
    with the size of the buffer this process allocated. A buffer this process
    owns is read through `memoryview`, which checks, in preference to
    `string_at`, which does not.
16. A linked list or a retry loop driven by native code has a bound: a segment
    count, a deadline.
17. A status word the interface defines is read. A positive status that means
    "not done" is not success.

### 5. ctypes declarations

18. Every foreign function has `argtypes` and `restype` assigned before its
    first call. A test introspects each binding for one that has neither.
19. Every `Structure` and `Union`, public or `_`-prefixed, appears in the
    recorded-ABI file for its binding with its size, every field offset, and
    for a bit-field the byte image of each member set to all ones. The
    coverage test fails for a class the recording lacks.
20. A module that declares `_pack_` or a 64-bit-only layout refuses a process
    of another pointer size.

### 6. Lifetimes

21. Memory native code may read after the call returns is held by a Python
    reference on the object whose lifetime matches it, with a comment naming
    what holds the pointer. ctypes keeps nothing alive for `cast`, `addressof`
    or a pointer stored in a structure field.
22. A structure handed to an asynchronous interface is not rewritten until the
    interface says it has finished with it.
23. Every callback native code makes is wrapped whole in a handler that
    returns the interface's failure status. An exception that reaches ctypes is
    printed and the caller receives whatever the result slot held.
24. State a later call submits from (`self._current`, `self.pool`) is assigned
    after every step that can raise has completed, and the old value is cleared
    before it is released.
25. A counter that selects a slot advances after the native call it selected
    for has succeeded.
26. A collection that keeps a surface alive (`held()`) names every surface the
    picture under construction still references.

### 7. Release

27. `close()` runs every release whatever the others do, through
    `teardown.release_all`, and each release clears its handle first so that a
    second call does nothing.
28. A resource acquired in a constructor or a factory is released by an
    `except BaseException` around everything after the acquisition.
29. No `__del__` reaches OpenGL, EGL or WGL. A destructor releases native
    objects that hold pointers into the dying object, warns with
    `ResourceWarning`, and leaves OpenGL names to their context.
30. An OpenGL name is deleted in the context that made it. An object that owns
    one records the context at creation and refuses to release in another.
31. A binding changed for a call is put back to what it was, not to zero.
32. `__exit__` does not raise over an exception that is already propagating.

### 8. Files

33. A file read from more than one call site is read with `os.pread` or under
    a lock; `seek` then `read` on a shared object is a defect.
34. An `mmap` is not held past indexing.
35. A writer validates in its constructor everything it will pack at `close()`.

## Holding the standards

Proposed, in the order they pay for themselves:

1. **OGC rules** in `openglcontext-checks`, each a syntax-tree check with no
   dependency: a bare `ue()`/`se()` in a `*_parse.py`; `ctypes.string_at` or
   `from_address` outside a function that calls `bounds.span`; a `__del__`
   whose module imports `OpenGL` (the existing GL-in-`__del__` rule, widened to
   a method the destructor calls); `assert` in `src/`; `ctypes.GetLastError`.
2. **A binding introspection test** shared by the four bindings: every
   function object reachable from an `api` module has `argtypes` and
   `restype`; every `Structure` and `Union` is in the recorded ABI.
3. **A fuzz gate** (done for every parser, container and NVDEC:
   `tests/test_fuzz_*.py`, about 30,000 seeded cases and 70 s in all): the
   scripts the review used, kept as
   `tests/test_fuzz_*.py` with a fixed seed and a few thousand cases each, run
   in the ordinary suite. Each asserts three things of every mutated input:
   only the package's errors escape, no case takes longer than a second, and
   every buffer handed to the recording device passes the range and
   round-trip assertions. A longer run with a random seed belongs in CI on a
   schedule.
4. **A round-trip store** used by every control layer, so that rule 11 is a
   property of one function and not of four hundred assignments. Done:
   `vaapi/control.py` has `store`, `store_each` and `Checked`, a structure
   whose every integer store is read back. The H.264 and HEVC layers use the
   first two, the AV1 and VP9 layers build inside a `Checked`, and a
   syntax-tree test for each fails a builder that fills a structure any other
   way.

## Appendix A: syntax elements and their ranges

The H.264 clauses are those of Rec. ITU-T H.264; the HEVC clauses those of
Rec. ITU-T H.265.

### H.264

| Element | Range | Clause |
|---|---|---|
| `ue(v)` code number | 0 to 2^32-2 | 9.1 |
| `seq_parameter_set_id` | 0 to 31 | 7.4.2.1.1 |
| `chroma_format_idc` | 0 to 3 | 7.4.2.1.1 |
| `bit_depth_luma_minus8`, `bit_depth_chroma_minus8` | 0 to 6 | 7.4.2.1.1 |
| `log2_max_frame_num_minus4` | 0 to 12 | 7.4.2.1.1 |
| `log2_max_pic_order_cnt_lsb_minus4` | 0 to 12 | 7.4.2.1.1 |
| `num_ref_frames_in_pic_order_cnt_cycle` | 0 to 255 | 7.4.2.1.1 |
| `offset_for_non_ref_pic`, `offset_for_top_to_bottom_field`, `offset_for_ref_frame` | -2^31+1 to 2^31-1 | 7.4.2.1.1 |
| `max_num_ref_frames` | 0 to 16 | 7.4.2.1.1, A.3.1 |
| `pic_width_in_mbs_minus1`, `pic_height_in_map_units_minus1` | `PicSizeInMbs` at most 139264 | A.3.1 |
| frame cropping offsets | the cropped picture is not empty | 7.4.2.1.1 |
| `max_dec_frame_buffering` | `max_num_ref_frames` to 16 | E.2.1 |
| `max_num_reorder_frames` | 0 to `max_dec_frame_buffering` | E.2.1 |
| `cpb_cnt_minus1` | 0 to 31 | E.2.2 |
| `pic_parameter_set_id` | 0 to 255 | 7.4.2.2 |
| `num_slice_groups_minus1` | 0 to 7 | 7.4.2.2 |
| `num_ref_idx_l0/l1_default_active_minus1` | 0 to 31 | 7.4.2.2 |
| `weighted_bipred_idc` | 0 to 2 | 7.4.2.2 |
| `pic_init_qp_minus26` | -(26 + QpBdOffsetY) to 25 | 7.4.2.2 |
| `pic_init_qs_minus26` | -26 to 25 | 7.4.2.2 |
| `chroma_qp_index_offset`, `second_chroma_qp_index_offset` | -12 to 12 | 7.4.2.2 |
| `delta_scale` | -128 to 127 | 7.4.2.1.1.1 |
| `first_mb_in_slice` | 0 to `PicSizeInMbs` - 1 | 7.4.3 |
| `slice_type` in an IDR unit | I or SI | 7.4.3 |
| `idr_pic_id` | 0 to 65535 | 7.4.3 |
| `redundant_pic_cnt` | 0 to 127 | 7.4.3 |
| `num_ref_idx_l0/l1_active_minus1` | 0 to 15 for a frame, 0 to 31 for a field | 7.4.3 |
| `cabac_init_idc` | 0 to 2 | 7.4.3 |
| `slice_qp_delta` | SliceQPY in -QpBdOffsetY to 51 | 7.4.3 |
| `disable_deblocking_filter_idc` | 0 to 2 | 7.4.3 |
| `slice_alpha_c0_offset_div2`, `slice_beta_offset_div2` | -6 to 6 | 7.4.3 |
| `luma_log2_weight_denom`, `chroma_log2_weight_denom` | 0 to 7 | 7.4.3.2 |
| weights and offsets | -128 to 127 | 7.4.3.2 |
| `long_term_frame_idx` | below `max_num_ref_frames` | 7.4.3.3 |
| `max_long_term_frame_idx_plus1` | 0 to `max_num_ref_frames` | 7.4.3.3 |
| slice header length | at most 65535 bits, the field libva carries it in | - |

### HEVC

| Element | Range | Clause |
|---|---|---|
| `sps_max_sub_layers_minus1`, `vps_max_sub_layers_minus1` | 0 to 6 | 7.4.3.1, 7.4.3.2.1 |
| `sps_seq_parameter_set_id` | 0 to 15 | 7.4.3.2.1 |
| `pic_width_in_luma_samples`, `pic_height_in_luma_samples` | above 0, a multiple of MinCbSizeY | 7.4.3.2.1 |
| conformance window | the cropped picture is not empty | 7.4.3.2.1 |
| `bit_depth_luma_minus8`, `bit_depth_chroma_minus8` | 0 to 8 | 7.4.3.2.1 |
| `sps_max_dec_pic_buffering_minus1` | 0 to 15 | 7.4.3.2.1, A.4.2 |
| `sps_max_num_reorder_pics` | 0 to `sps_max_dec_pic_buffering_minus1` | 7.4.3.2.1 |
| `log2_min_luma_coding_block_size_minus3`, `log2_diff_max_min_luma_coding_block_size` | CtbLog2SizeY in 4 to 6 | 7.4.3.2.1 |
| `log2_min_luma_transform_block_size_minus2`, `log2_diff_max_min_luma_transform_block_size` | MinTbLog2SizeY below MinCbLog2SizeY; MaxTbLog2SizeY at most min(CtbLog2SizeY, 5) | 7.4.3.2.1 |
| `max_transform_hierarchy_depth_inter`, `_intra` | 0 to CtbLog2SizeY - MinTbLog2SizeY | 7.4.3.2.1 |
| `scaling_list_dc_coef_minus8` | -7 to 247 | 7.4.5 |
| `pcm_sample_bit_depth_luma_minus1`, `_chroma_minus1` | at most BitDepth - 1 | 7.4.3.2.1 |
| PCM block-size logarithms | 3 to min(CtbLog2SizeY, 5) | 7.4.3.2.1 |
| pictures in a reference picture set, either form | at most `sps_max_dec_pic_buffering_minus1` | 7.4.8 |
| `abs_delta_rps_minus1`, `delta_poc_s0_minus1`, `delta_poc_s1_minus1` | 0 to 2^15-1 | 7.4.8 |
| `num_long_term_ref_pics_sps` | 0 to 32 | 7.4.3.2.1 |
| `pps_pic_parameter_set_id` | 0 to 63 | 7.4.3.3.1 |
| `init_qp_minus26` | -(26 + QpBdOffsetY) to 25 | 7.4.3.3.1 |
| `diff_cu_qp_delta_depth` | 0 to `log2_diff_max_min_luma_coding_block_size` | 7.4.3.3.1 |
| `pps_cb_qp_offset`, `pps_cr_qp_offset` | -12 to 12 | 7.4.3.3.1 |
| `num_tile_columns_minus1`, `num_tile_rows_minus1` | below PicWidthInCtbsY, PicHeightInCtbsY | 7.4.3.3.1 |
| `pps_beta_offset_div2`, `pps_tc_offset_div2` | -6 to 6 | 7.4.3.3.1 |
| `log2_parallel_merge_level_minus2` | 0 to CtbLog2SizeY - 2 | 7.4.3.3.1 |
| `slice_segment_address` | 0 to PicSizeInCtbsY - 1 | 7.4.7.1 |
| `slice_type` in an IRAP picture | I | 7.4.7.1 |
| `slice_pic_parameter_set_id` | the same in every segment of a picture | 7.4.7.1 |
| `collocated_ref_idx` | 0 to `num_ref_idx_lX_active_minus1` | 7.4.7.1 |
| `five_minus_max_num_merge_cand` | 0 to 4 | 7.4.7.1 |
| `slice_qp_delta` | SliceQpY in -QpBdOffsetY to 51 | 7.4.7.1 |
| `slice_cb_qp_offset`, `slice_cr_qp_offset` | -12 to 12, alone and summed with the PPS's | 7.4.7.1 |
| `slice_beta_offset_div2`, `slice_tc_offset_div2` | -6 to 6 | 7.4.7.1 |
| `delta_luma_weight`, `luma_offset`, `delta_chroma_weight` | -128 to 127 | 7.4.7.3 |
| `num_entry_point_offsets` | the tile and CTB-row counts allow | 7.4.7.1 |
| reserved VCL unit types 10 to 15 and 22 to 31 | ignored | 7.4.2.2 |
