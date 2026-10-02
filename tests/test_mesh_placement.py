from __future__ import annotations

from dataclasses import replace

from decanting_workspace import BoxPrimitive, load_scene_spec, materialize_scene
from decanting_workspace.models import MeshObstacle
from decanting_workspace.placement import (
    _inflate_box,
    _mesh_overlap_3d,
    validate_base_placement,
)


def _cube_mesh(center, size, *, name="mesh_obstacle", collision_enabled=True):
    offsets = (
        (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
    )
    vertices = tuple(
        tuple(center[axis] + offset[axis] * size[axis] / 2.0 for axis in range(3))
        for offset in offsets
    )
    triangles = (
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    )
    return MeshObstacle(
        name=name,
        role="obstacle",
        vertices_m=vertices,
        triangles=triangles,
        collision_enabled=collision_enabled,
    )


def _pedestal(*, yaw=0.0, size=(0.3, 0.3, 0.5)):
    return BoxPrimitive(
        name="ur20_pedestal", role="pedestal", center_m=(0.0, 0.0, 0.25),
        size_m=size, yaw_deg=yaw,
    )


def test_mesh_overlap_reports_placement_issue_and_honors_collision_toggle():
    spec = load_scene_spec()
    snapshot = materialize_scene(spec)
    pedestal = next(box for box in snapshot.boxes if box.name == "ur20_pedestal")
    mesh = _cube_mesh(pedestal.center_m, (0.1, 0.1, 0.1))
    mesh_spec = replace(spec, static_meshes=(mesh,))

    report = validate_base_placement(mesh_spec, snapshot)

    assert any(
        issue.code == "static_mesh_overlap"
        and issue.objects == ("ur20_pedestal", "mesh_obstacle")
        for issue in report.issues
    )
    disabled_spec = replace(mesh_spec, static_meshes=(replace(mesh, collision_enabled=False),))
    assert validate_base_placement(disabled_spec, snapshot).valid


def test_mesh_components_leave_the_gap_in_their_combined_bounds_available():
    left = _cube_mesh((-2.0, 0.0, 0.25), (1.0, 1.0, 1.0))
    right = _cube_mesh((2.0, 0.0, 0.25), (1.0, 1.0, 1.0))
    joined = replace(
        left,
        vertices_m=left.vertices_m + right.vertices_m,
        triangles=left.triangles + tuple(tuple(index + 8 for index in face) for face in right.triangles),
    )

    assert not _mesh_overlap_3d(_pedestal(), joined)


def test_mesh_checks_exact_pedestal_yaw():
    pedestal = _pedestal(yaw=45.0, size=(2.0, 0.2, 0.5))
    along_long_axis = _cube_mesh((0.65, 0.65, 0.25), (0.05, 0.05, 0.05))
    within_world_bounds_but_off_axis = _cube_mesh((0.65, -0.65, 0.25), (0.05, 0.05, 0.05))

    assert _mesh_overlap_3d(pedestal, along_long_axis)
    assert not _mesh_overlap_3d(pedestal, within_world_bounds_but_off_axis)


def test_touching_mesh_is_allowed_and_clearance_uses_existing_box_envelope():
    pedestal = _pedestal()
    touching = _cube_mesh((0.2, 0.0, 0.25), (0.1, 0.1, 0.5))

    assert not _mesh_overlap_3d(pedestal, touching)
    assert _mesh_overlap_3d(_inflate_box(pedestal, 0.01), touching)


def test_closed_mesh_enclosing_entire_pedestal_is_rejected():
    enclosing = _cube_mesh((0.0, 0.0, 0.25), (2.0, 2.0, 2.0))

    assert _mesh_overlap_3d(_pedestal(), enclosing)


def test_closed_component_still_encloses_pedestal_with_an_unrelated_open_sheet():
    enclosing = _cube_mesh((0.0, 0.0, 0.25), (2.0, 2.0, 2.0))
    mixed_mesh = replace(
        enclosing,
        vertices_m=enclosing.vertices_m + ((5.0, 0.0, 0.0), (6.0, 0.0, 0.0), (5.0, 1.0, 0.0)),
        triangles=enclosing.triangles + ((8, 9, 10),),
    )

    assert _mesh_overlap_3d(_pedestal(), mixed_mesh)


def test_open_surface_has_no_assumed_solid_interior():
    shell = _cube_mesh((0.0, 0.0, 0.25), (2.0, 2.0, 2.0))
    open_shell = replace(shell, triangles=shell.triangles[:-2])

    assert not _mesh_overlap_3d(_pedestal(), open_shell)


def test_closed_nested_shell_preserves_its_empty_cavity():
    outer = _cube_mesh((0.0, 0.0, 0.25), (2.0, 2.0, 2.0))
    inner = _cube_mesh((0.0, 0.0, 0.25), (1.0, 1.0, 1.0))
    hollow = replace(
        outer,
        vertices_m=outer.vertices_m + inner.vertices_m,
        triangles=outer.triangles + tuple(tuple(index + 8 for index in face) for face in inner.triangles),
    )

    assert not _mesh_overlap_3d(_pedestal(), hollow)
