from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from decanting_workspace.robots import load_robot_bundle


URDF_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "robots"
    / "sr12ia"
    / "sr12ia_approx.urdf"
)


@pytest.fixture(scope="module")
def sr12ia():
    return load_robot_bundle(
        "sr12ia", URDF_PATH, floating_base=False, load_visual=True
    )


def _frame_translation(bundle, frame_name: str, q: np.ndarray) -> np.ndarray:
    import pinocchio as pin

    data = bundle.model.createData()
    pin.framesForwardKinematics(bundle.model, data, q)
    frame_id = bundle.model.getFrameId(frame_name)
    assert frame_id < bundle.model.nframes
    return np.asarray(data.oMf[frame_id].translation).reshape(3).copy()


def test_zero_pose_matches_published_two_link_scara_skeleton(sr12ia):
    model = sr12ia.model
    assert model.nq == 4
    assert list(model.names[1:]) == ["joint1", "joint2", "joint3", "joint4"]

    q_zero = np.zeros(model.nq)
    tool0 = _frame_translation(sr12ia, "tool0", q_zero)
    joint2 = _frame_translation(sr12ia, "arm_2", q_zero)
    joint3 = _frame_translation(sr12ia, "vertical_slide", q_zero)

    assert joint2 == pytest.approx((0.450, 0.0, 0.336), abs=1e-12)
    assert joint3 == pytest.approx((0.900, 0.0, 0.336), abs=1e-12)
    assert tool0 == pytest.approx((0.900, 0.0, 0.336), abs=1e-12)
    assert np.hypot(tool0[0], tool0[1]) == pytest.approx(0.900, abs=1e-12)


@pytest.mark.parametrize("j3_extension", (0.0, 0.125, 0.300, 0.450))
def test_j3_extension_moves_tool0_down_without_xy_or_fixed_flange_offset(
    sr12ia, j3_extension
):
    q = np.array((0.0, 0.0, j3_extension, 0.0))
    tool0 = _frame_translation(sr12ia, "tool0", q)
    joint4 = _frame_translation(sr12ia, "tool_head", q)

    assert tool0 == pytest.approx((0.900, 0.0, 0.336 - j3_extension), abs=1e-12)
    assert tool0 == pytest.approx(joint4, abs=1e-12)


def test_no_process_tool_is_attached_to_sr12ia_fallback(sr12ia):
    assert not sr12ia.model.existFrame("cutter_proxy")
    assert not sr12ia.model.existFrame("cutter_tcp")
    assert all("cutter" not in geometry.name for geometry in sr12ia.collision_model.geometryObjects)
    assert all("cutter" not in geometry.name for geometry in sr12ia.visual_model.geometryObjects)


def test_official_joint_ranges_are_encoded(sr12ia):
    model = sr12ia.model

    assert model.lowerPositionLimit == pytest.approx(
        (-np.deg2rad(145.0), -np.deg2rad(145.0), 0.0, -np.deg2rad(720.0)),
        abs=1e-9,
    )
    assert model.upperPositionLimit == pytest.approx(
        (np.deg2rad(145.0), np.deg2rad(145.0), 0.450, np.deg2rad(720.0)),
        abs=1e-9,
    )
