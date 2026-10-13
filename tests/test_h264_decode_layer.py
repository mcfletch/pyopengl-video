"""The H.264 control layer over written streams, against a session with no driver.

Each stream is written here, a parameter set and a slice at a time, so that
it holds the one thing under test: a picture decoded before any other, a
parameter set that changes within a picture, a gap in frame_num.
"""
import pytest

from pyopengl_video.decoder import Sample
from pyopengl_video.vaapi.h264 import BitWriter, StreamError, annexb, nal_unit
from pyopengl_video.vaapi.h264_decode import H264Decoding
from tests.h264_syntax import restriction_vui, slice_unit, sps_bytes, write_pps_head


class RecordingSession:
    """A session that hands out surface numbers and records what a layer does."""

    def __init__(self):
        self.held = set
        self.formats = []
        self.submitted = []
        self.shown = []
        self.events = []
        self.count = 0

    def activate(self, format):
        self.formats.append(format)
        self.count = format.surfaces

    def acquire(self):
        surface = min(set(range(self.count)) - self.held())
        self.events.append(('acquired', surface))
        return surface

    def submit(self, surface, batches, what):
        self.submitted.append((surface, batches, what))

    def output(self, surface, *, timestamp, duration, visible, colour):  # noqa: ARG002 - the session's signature
        self.shown.append((surface, timestamp, visible))
        self.events.append(('shown', surface))


def h264_layer():
    session = RecordingSession()
    layer = H264Decoding(session, output_slots=4)
    session.held = layer.held
    return layer, session


def written_sets(**sps):
    """An SPS and a PPS as Annex-B: a 64x64 picture, picture order count type 0."""
    writer = BitWriter()
    write_pps_head(writer)
    return annexb(nal_unit(sps_bytes(**sps), 7), nal_unit(writer.rbsp(), 8))


def written_slice(frame_num=0, poc_lsb=0, *, idr=False, kind=None, first_mb=0):
    """One slice of a reference picture as Annex-B: an I slice of an IDR
    picture, or a P slice predicting from one frame."""
    def write(writer):
        writer.ue(first_mb)
        writer.ue(kind if kind is not None else 7 if idr else 5)
        writer.ue(0)                            # pic_parameter_set_id
        writer.u(frame_num, 4)
        if idr:
            writer.ue(0)                        # idr_pic_id
        writer.u(poc_lsb, 6)
        if idr:
            writer.flag(0)                      # no_output_of_prior_pics_flag
            writer.flag(0)                      # long_term_reference_flag
        else:
            writer.flag(0)                      # num_ref_idx_active_override_flag
            writer.flag(0)                      # ref_pic_list_modification_flag_l0
            writer.flag(0)                      # adaptive_ref_pic_marking_mode_flag
            writer.ue(0)                        # cabac_init_idc
        writer.se(0)                            # slice_qp_delta
        writer.ue(1)                            # disable_deblocking_filter_idc
    return annexb(slice_unit(write, nal_unit_type=5 if idr else 1).data)


def sample(data, index=0):
    return Sample(data, index, 1, False, index)


def test_a_flush_before_any_picture_shows_nothing():
    """H17: nothing has been decoded, so there is nothing to show; with the
    parameter sets alone the same."""
    layer, session = h264_layer()
    layer.flush()
    layer.decode(sample(written_sets()))
    layer.flush()
    assert (session.submitted, session.shown) == ([], [])


def test_a_stream_opening_on_a_predicted_slice_is_refused():
    """H11: no frame has been decoded, so the slice's list has no surface to
    name and the picture is not submitted."""
    layer, session = h264_layer()
    with pytest.raises(StreamError, match='no decoded frame to predict from'):
        layer.decode(sample(written_sets() + written_slice(3, 6)))
    assert session.submitted == []


def test_a_slice_coded_against_another_sequence_than_its_pictures_is_refused():
    """A sequence parameter set of another size arrives between two slices of
    one picture: the second is not added to a picture of the first's size."""
    layer, session = h264_layer()
    data = (written_sets() + written_slice(idr=True)
            + annexb(nal_unit(sps_bytes(width_mbs=8, height_mbs=8), 7))
            + written_slice(idr=True, first_mb=20))
    with pytest.raises(StreamError, match='other parameter sets than the picture'):
        layer.decode(sample(data))
    assert session.submitted == []


def test_parameter_sets_sent_again_unchanged_do_not_end_a_picture():
    layer, session = h264_layer()
    layer.decode(sample(written_sets() + written_slice(idr=True) + written_sets()
                        + written_slice(idr=True, first_mb=8)))
    (_, batches, _), = session.submitted
    assert len(batches) == 1 + 2


def test_a_sequence_asks_for_seventeen_surfaces_at_most():
    """H9: sixteen frames in the largest buffer there is, and the picture
    being decoded."""
    layer, session = h264_layer()
    layer.decode(sample(written_sets(max_num_ref_frames=16, level_idc=51)
                        + written_slice(idr=True)))
    assert session.formats[-1].surfaces == 17
    with pytest.raises(StreamError, match='max_num_ref_frames is 100000'):
        layer.decode(sample(written_sets(max_num_ref_frames=100000)))


def test_a_frame_a_gap_pushes_out_is_shown_before_its_surface_is_decoded_into():
    """Frames 0 and 1 wait to be shown in a buffer of two. frame_num then
    skips 2, the frame inferred for it takes a place, and frame 0 is output
    when the picture begins. Its surface is free from then on, so it is shown
    before a surface is chosen for the picture."""
    layer, session = h264_layer()
    layer.decode(sample(written_sets(max_num_ref_frames=2, gaps=1,
                                     vui=restriction_vui(2, 2))
                        + written_slice(idr=True)))
    layer.decode(sample(written_slice(1, 2), 1))
    assert session.events == [('acquired', 0), ('acquired', 1)]
    layer.decode(sample(written_slice(3, 6), 2))
    assert session.events[2:4] == [('shown', 0), ('acquired', 0)]
