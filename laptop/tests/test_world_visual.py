"""v2.2: alignment from the glasses' camera seeing the phone (contracts §12)."""

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from pointview import world as w
from pointview.interfaces import Alignment, PhoneVisualFix

angle = st.floats(-180, 180)


def fix_from(head_from_phone):
    return PhoneVisualFix(0.0, head_from_phone, (0.0, 0.3, -0.3), 1.0, True, 7)


@settings(max_examples=300, deadline=None)
@given(
    head_yaw=angle,
    head_pitch=st.floats(-40, 40),
    phone_yaw_r=angle,
    offset=angle,
    pitch=st.floats(-60, 60),
)
def test_visual_offset_recovers_the_true_phone_to_head_offset(
    head_yaw, head_pitch, phone_yaw_r, offset, pitch
):
    # Truth: the phone's top edge points at tracking yaw (phone_yaw_r + offset), pitch `pitch`.
    q_head = w.quat_from_yaw_pitch_roll(head_yaw, head_pitch)
    q_phone_r = w.quat_from_yaw_pitch_roll(phone_yaw_r, pitch)  # phone attitude in its frame R
    q_phone_t = w.quat_from_yaw_pitch_roll(phone_yaw_r + offset, pitch)  # same, in tracking T
    head_from_phone = w.quat_mul(w.quat_conj(q_head), q_phone_t)  # what the camera measures
    got = w.visual_phone_to_head_deg(fix_from(head_from_phone), q_head, q_phone_r)
    assert abs(w.wrap_deg_signed(got - offset)) < 1e-6


def test_matches_one_tap_align_when_looking_where_the_phone_points():
    q_head = w.quat_from_yaw_pitch_roll(30.0, 5.0)
    q_phone_r = w.quat_from_yaw_pitch_roll(-12.0, 5.0)
    a = w.align(q_head, q_phone_r, None, None, 0.0)
    # The phone points exactly where the head looks: in the head frame its top edge is +y.
    q_phone_t = w.quat_from_yaw_pitch_roll(30.0, 5.0)
    got = w.visual_phone_to_head_deg(
        fix_from(w.quat_mul(w.quat_conj(q_head), q_phone_t)), q_head, q_phone_r
    )
    assert abs(w.wrap_deg_signed(got - a.phone_to_head_deg)) < 1e-9


def test_straight_up_is_undefined():
    q = w.quat_from_yaw_pitch_roll(0.0, 89.0)
    assert math.isnan(w.visual_phone_to_head_deg(fix_from(q), (1.0, 0.0, 0.0, 0.0), q))


def test_blend_is_circular_and_keeps_north():
    a = Alignment(phone_to_head_deg=179.0, north_offset_deg=12.0, aligned_at=1.0)
    b = w.blend_alignment(a, -179.0, 2.0, gain=0.5)
    assert abs(w.wrap_deg_signed(b.phone_to_head_deg - 180.0)) < 1e-9
    assert b.north_offset_deg == 12.0 and b.aligned_at == 2.0
    first = w.blend_alignment(Alignment(), 42.0, 3.0)
    assert first.phone_to_head_deg == 42.0 and first.aligned_at == 3.0
    assert w.blend_alignment(a, math.nan, 4.0) is a
