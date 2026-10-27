"""W2's independent reference frame math must agree with pointview.world (W3)."""

from hypothesis import given, settings
from hypothesis import strategies as st

from pointview import world
from pointview.interfaces import CameraIntrinsics

from . import _world_ref as ref

INTR = CameraIntrinsics(1000.0, 1000.0, 960.0, 540.0, 1920, 1080)
angle = st.floats(-170, 170)


@settings(max_examples=200, deadline=None)
@given(
    u=st.floats(0, 1920),
    v=st.floats(0, 1080),
    yaw=angle,
    pitch=st.floats(-60, 60),
    north=angle,
)
def test_pixel_to_world_matches_reference(u, v, yaw, pitch, north):
    q = world.quat_from_yaw_pitch_roll(yaw, pitch)
    ident = (1.0, 0.0, 0.0, 0.0)
    a = world.pixel_to_world(u, v, INTR, q, north, ident)
    b = ref.pixel_to_world(u, v, INTR, q, north, ident)
    assert abs(world.wrap_deg_signed(a[0] - b[0])) < 1e-6 or abs(a[1]) > 89.9
    assert abs(a[1] - b[1]) < 1e-6
    assert world.camera_angles_deg(u, v, INTR) == ref.camera_angles_deg(u, v, INTR) or all(
        abs(x - y) < 1e-9
        for x, y in zip(
            world.camera_angles_deg(u, v, INTR), ref.camera_angles_deg(u, v, INTR), strict=True
        )
    )
