from __future__ import annotations

import numpy as np
import pytest

from decanting_workspace import load_scene_spec
from decanting_workspace.robots import floating_configuration, load_robot_bundle
from decanting_workspace.scene import sr_cutting_workspace_footprint_box
from decanting_workspace.transforms import quaternion_matrix


@pytest.fixture(scope="module")
def assets():
    spec = load_scene_spec()
    ur = load_robot_bundle("ur20", spec.robots["ur20"].urdf_path, load_visual=True)
    sr = load_robot_bundle(
        "sr12ia",
        spec.robots["sr12ia"].urdf_path,
        load_visual=True,
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


def test_sr12ia_nominal_tool0_is_near_conveyor_edge_of_task_workspace(assets):
    import pinocchio as pin

    spec, _, sr = assets
    sr_spec = spec.robots["sr12ia"]
    q = floating_configuration(sr, sr_spec.nominal_base, sr_spec.nominal_q)
    data = sr.model.createData()
    pin.framesForwardKinematics(sr.model, data, q)
    tool_pose = data.oMf[sr.model.getFrameId("tool0")]
    tool0 = np.asarray(tool_pose.translation).reshape(3)
    load_frame = spec.frames["UncasingLoadFrame"]
    group_frame = spec.frames["UncaseGroup"]
    group_pose = quaternion_matrix(
        group_frame.translation_m,
        group_frame.rotation_xyzw,
    )
    group_positive_y = group_pose[:3, 1]
    elbow = np.asarray(
        data.oMf[sr.model.getFrameId("arm_2")].translation
    ).reshape(3)
    footprint = sr_cutting_workspace_footprint_box(spec, sr_spec.nominal_base)
    yaw = np.deg2rad(footprint.yaw_deg)
    c = np.cos(yaw)
    s = np.sin(yaw)
    half_x, half_y = np.asarray(footprint.size_m[:2]) / 2.0
    footprint_corner_x = [
        footprint.center_m[0] + c * sign_x * half_x - s * sign_y * half_y
        for sign_x in (-1.0, 1.0)
        for sign_y in (-1.0, 1.0)
    ]
    conveyor_edge_x = max(footprint_corner_x)
    expected_xy = np.array(
        (
            conveyor_edge_x - 0.02,
            load_frame.translation_m[1] + 0.10 * group_positive_y[1],
        )
    )
    world_offset = tool0[:2] - np.asarray(footprint.center_m[:2])
    footprint_local = np.array(
        (
            c * world_offset[0] + s * world_offset[1],
            -s * world_offset[0] + c * world_offset[1],
        )
    )

    assert tool0[:2] == pytest.approx(expected_xy, abs=1e-9)
    assert tool0[2] > load_frame.translation_m[2]
    assert np.asarray(tool_pose.rotation)[:, 0] == pytest.approx(
        group_positive_y,
        abs=1e-9,
    )
    assert sr_spec.nominal_q[1] < 0.0
    assert elbow[0] > sr_spec.nominal_base.x_m
    assert conveyor_edge_x - tool0[0] == pytest.approx(0.02, abs=1e-9)
    assert np.all(np.abs(footprint_local) <= np.asarray(footprint.size_m[:2]) / 2.0)


def test_ur20_nominal_pose_points_tool0_toward_pallet(assets):
    import pinocchio as pin

    spec, ur, _ = assets
    ur_spec = spec.robots["ur20"]
    q = floating_configuration(ur, ur_spec.nominal_base, ur_spec.nominal_q)
    data = ur.model.createData()
    pin.framesForwardKinematics(ur.model, data, q)
    tool0 = np.asarray(data.oMf[ur.model.getFrameId("tool0")].translation).reshape(3)
    tool_direction = tool0[:2] - np.asarray(ur_spec.nominal_base.xyz_m[:2])
    pallet_direction = np.asarray(spec.pallet.center_xy_m) - np.asarray(
        ur_spec.nominal_base.xyz_m[:2]
    )
    cosine = float(
        np.dot(tool_direction, pallet_direction)
        / (np.linalg.norm(tool_direction) * np.linalg.norm(pallet_direction))
    )

    assert cosine > 0.999999
