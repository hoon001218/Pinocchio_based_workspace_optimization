from __future__ import annotations

import numpy as np
import pytest

from decanting_workspace import load_scene_spec
from decanting_workspace.robots import floating_configuration, load_robot_bundle


@pytest.fixture(scope="module")
def assets():
    spec = load_scene_spec()
    ur = load_robot_bundle("ur20", spec.robots["ur20"].urdf_path, load_visual=True)
    sr = load_robot_bundle(
        "sr12ia",
        spec.robots["sr12ia"].urdf_path,
        load_visual=True,
        cutter_proxy=spec.robots["sr12ia"].cutter_proxy,
    )
    return spec, ur, sr


def test_ur20_visual_urdf_is_six_axis_and_nonempty(assets):
    spec, ur, _ = assets
    assert ur.model.nq == 13
    assert ur.model.nv == 12
    assert list(ur.model.names[2:]) == [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    assert ur.model.getFrameId("tool0") < ur.model.nframes
    assert ur.visual_model.ngeoms >= 9
    assert ur.collision_model.ngeoms >= 9
    q = floating_configuration(ur, spec.robots["ur20"].nominal_base, spec.robots["ur20"].nominal_q)
    assert q.shape == (13,)
    assert np.linalg.norm(q[3:7]) == pytest.approx(1.0)
    assert ur.arm_velocity_slice == slice(6, 12)


def test_sr12ia_approximation_is_four_axis_and_nonempty(assets):
    spec, _, sr = assets
    assert sr.model.nq == 11
    assert sr.model.nv == 10
    assert list(sr.model.names[2:]) == ["joint1", "joint2", "joint3", "joint4"]
    assert sr.visual_model.ngeoms >= 10
    assert sr.collision_model.ngeoms >= 10
    q = floating_configuration(sr, spec.robots["sr12ia"].nominal_base, spec.robots["sr12ia"].nominal_q)
    assert q.shape == (11,)
    assert sr.arm_velocity_slice == slice(6, 10)


def test_sr12ia_nominal_working_pose_places_cutter_tcp_at_uncasing_frame(assets):
    import pinocchio as pin

    spec, _, sr = assets
    sr_spec = spec.robots["sr12ia"]
    q = floating_configuration(sr, sr_spec.nominal_base, sr_spec.nominal_q)
    data = sr.model.createData()
    pin.framesForwardKinematics(sr.model, data, q)
    cutter_tcp = np.asarray(
        data.oMf[sr.model.getFrameId("cutter_tcp")].translation
    ).reshape(3)

    assert cutter_tcp == pytest.approx(
        spec.frames["UncasingLoadFrame"].translation_m,
        abs=1e-9,
    )
