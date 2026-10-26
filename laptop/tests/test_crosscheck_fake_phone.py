"""The fake phone's independent quaternion code must agree with world.py's pointer math."""

from hypothesis import given, settings
from hypothesis import strategies as st

from pointview import world
from pointview.mock import load_fake_phone

fp = load_fake_phone()


@settings(max_examples=200, deadline=None)
@given(yaw=st.floats(-180, 180), pitch=st.floats(-80, 80))
def test_fake_phone_pointer_matches_world(yaw, pitch):
    q = fp.pointer_quat(yaw, pitch)
    y, p = world.yaw_pitch_from_vector(world.phone_pointer_vector(q))
    assert abs(world.wrap_deg_signed(y - yaw)) < 1e-6
    assert abs(p - pitch) < 1e-6
    # ... and with a fresh Align at that pose and no heading, the ray is the head direction.
    head = world.quat_from_yaw_pitch_roll(yaw + 17.0, pitch)
    a = world.align(head, q, None, None, 0.0)
    ry, rp = world.pointer_ray(q, a)
    assert abs(world.wrap_deg_signed(ry - (yaw + 17.0))) < 1e-6 and abs(rp - pitch) < 1e-6
