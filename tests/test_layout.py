from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import pytest

from decanting_workspace.layout import apply_layout_offsets
from decanting_workspace.layout_models import (
    LayoutBox,
    LayoutFrame,
    LayoutNode,
    LayoutReferenceDimensions,
    LayoutRobot,
    LayoutScene,
)


_IDENTITY = (0.0, 0.0, 0.0, 1.0)
_YAW_90 = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))


@pytest.fixture
def layout_scene() -> LayoutScene:
    # Child comes first to exercise parent lookup independent of tuple order.
    return LayoutScene(
        source_path=Path("example.usd"),
        nodes=(
            LayoutNode("/World/Group/Child", "/World/Group", (4.0, 6.0, 2.0), _IDENTITY),
            LayoutNode("/World/Group", "", (4.0, 5.0, 1.0), _YAW_90, kind="group"),
            LayoutNode("/World/Other", "", (10.0, 5.0, 0.0), _IDENTITY, kind="group"),
        ),
        boxes=(
            LayoutBox("child_box", "/World/Group/Child", (4.2, 6.3, 2.4), (1.0, 2.0, 3.0),
                      _YAW_90, source_prim_path="/World/Group/Child/Mesh"),
            LayoutBox("other_box", "/World/Other", (10.0, 5.0, 0.5), (1.0, 1.0, 1.0), _IDENTITY),
        ),
        robots=(LayoutRobot("ur20", "/World/Group/Child", (4.0, 6.0, 2.0), 45.0,
                            Path("ur20.urdf"), (0.1, 0.2), source_prim_path="/World/Group/Child/RB"),),
        frames=(LayoutFrame("task", "/World/Group/Child", (4.0, 6.0, 3.0), _YAW_90,
                            source_prim_path="/World/Group/Child/Task"),),
    )


def test_parent_and_child_offsets_are_applied_once_in_parent_axes(layout_scene):
    original_child = layout_scene.nodes[0]
    moved = apply_layout_offsets(layout_scene, {
        "/World/Group": (1.0, 2.0, 3.0),
        "/World/Group/Child": (2.0, 0.0, 0.5),
    })

    assert moved.nodes[1].translation_m == pytest.approx((5.0, 7.0, 4.0))
    assert moved.nodes[0].translation_m == pytest.approx((5.0, 10.0, 5.5))
    assert moved.boxes[0].center_m == pytest.approx((5.2, 10.3, 5.9))
    assert moved.robots[0].translation_m == pytest.approx((5.0, 10.0, 5.5))
    assert moved.frames[0].translation_m == pytest.approx((5.0, 10.0, 6.5))
    assert moved.nodes[2] == layout_scene.nodes[2]
    assert moved.boxes[1] == layout_scene.boxes[1]
    assert layout_scene.nodes[0] == original_child
    assert moved.boxes[0].rotation_xyzw == layout_scene.boxes[0].rotation_xyzw
    assert moved.boxes[0].size_m == layout_scene.boxes[0].size_m
    assert moved.boxes[0].source_prim_path == layout_scene.boxes[0].source_prim_path
    assert moved.robots[0].yaw_deg == layout_scene.robots[0].yaw_deg
    assert moved.robots[0].nominal_q == layout_scene.robots[0].nominal_q
    assert moved.robots[0].urdf_path == layout_scene.robots[0].urdf_path
    assert moved.frames[0].rotation_xyzw == layout_scene.frames[0].rotation_xyzw
    assert [node.rotation_xyzw for node in moved.nodes] == [node.rotation_xyzw for node in layout_scene.nodes]


def test_grandchild_inherits_offsets_without_using_its_own_rotation(layout_scene):
    grandchild = LayoutNode("/World/Group/Child/Leaf", "/World/Group/Child", (4.0, 7.0, 2.0), _YAW_90)
    leaf_box = LayoutBox("leaf", grandchild.prim_path, (4.0, 7.0, 2.0), (1.0, 1.0, 1.0), _IDENTITY)
    baseline = replace(layout_scene, nodes=(*layout_scene.nodes, grandchild), boxes=(*layout_scene.boxes, leaf_box))
    moved = apply_layout_offsets(baseline, {
        "/World/Group": (1.0, 0.0, 0.0),
        "/World/Group/Child": (1.0, 0.0, 0.0),
        grandchild.prim_path: (0.0, 2.0, 0.0),
    })
    # Group's world offset + child's rotated offset + leaf's identity-parent offset.
    assert moved.nodes[-1].translation_m == pytest.approx((5.0, 10.0, 2.0))
    assert moved.boxes[-1].center_m == pytest.approx((5.0, 10.0, 2.0))


def test_parent_offsets_inherit_to_children_and_do_not_accumulate_between_candidates(layout_scene):
    offsets = {"/World/Group": [0.5, -0.5, 0.1]}
    first = apply_layout_offsets(layout_scene, offsets)
    second = apply_layout_offsets(layout_scene, offsets)
    assert first == second
    assert first.nodes[0].translation_m == pytest.approx((4.5, 5.5, 2.1))
    assert layout_scene.nodes[0].translation_m == (4.0, 6.0, 2.0)
    assert offsets == {"/World/Group": [0.5, -0.5, 0.1]}
    assert apply_layout_offsets(layout_scene, {}) == layout_scene


def test_full_parent_rotation_controls_translation_even_when_parent_is_tilted(layout_scene):
    # An X half-turn maps positive child-local Z into negative World Z.
    group = replace(layout_scene.nodes[1], rotation_xyzw=(1.0, 0.0, 0.0, 0.0))
    baseline = replace(layout_scene, nodes=(layout_scene.nodes[0], group, layout_scene.nodes[2]))
    moved = apply_layout_offsets(baseline, {"/World/Group/Child": (0.0, 0.0, 0.5)})
    assert moved.nodes[0].translation_m == pytest.approx((4.0, 6.0, 1.5))
    assert moved.nodes[1].rotation_xyzw == group.rotation_xyzw


@pytest.mark.parametrize("offset", [(1.0, 2.0), (1.0, 2.0, 3.0, 4.0), (math.nan, 0.0, 0.0),
                                   (0.0, math.inf, 0.0), [[1.0], [2.0], [3.0]], "bad"])
def test_invalid_offsets_fail(layout_scene, offset):
    with pytest.raises(ValueError, match="three finite values"):
        apply_layout_offsets(layout_scene, {"/World/Group": offset})


def test_unknown_offset_node_fails(layout_scene):
    with pytest.raises(ValueError, match="unknown nodes"):
        apply_layout_offsets(layout_scene, {"/World/Missing": (0.0, 0.0, 0.0)})


def test_dangling_parent_fails_even_without_offsets(layout_scene):
    child = replace(layout_scene.nodes[0], parent_path="/World/Missing")
    malformed = replace(layout_scene, nodes=(child, *layout_scene.nodes[1:]))
    with pytest.raises(ValueError, match="unknown parent"):
        apply_layout_offsets(malformed, {})


def test_hierarchy_cycle_fails_even_without_offsets(layout_scene):
    group = replace(layout_scene.nodes[1], parent_path=layout_scene.nodes[0].prim_path)
    malformed = replace(layout_scene, nodes=(layout_scene.nodes[0], group, layout_scene.nodes[2]))
    with pytest.raises(ValueError, match="cycle"):
        apply_layout_offsets(malformed, {})


def test_unknown_attachment_owner_fails(layout_scene):
    malformed = replace(layout_scene, boxes=(replace(layout_scene.boxes[0], node_path="/World/Missing"),))
    with pytest.raises(ValueError, match="unknown node"):
        apply_layout_offsets(malformed, {})


def test_reference_dimensions_do_not_move_or_become_layout_attachments(layout_scene):
    reference = LayoutReferenceDimensions('sample_tote', (.6,.4,.3), '/World/Group/small_KLT')
    baseline = replace(layout_scene, reference_dimensions=(reference,))
    moved = apply_layout_offsets(baseline, {'/World/Group': (2.,3.,4.)})
    assert moved.reference_dimensions == (reference,)
    assert moved.reference_dimensions[0] is reference
    assert moved.boxes[0].center_m != baseline.boxes[0].center_m
    with pytest.raises(ValueError, match='unknown nodes'):
        apply_layout_offsets(baseline, {reference.source_prim_path: (1.,0.,0.)})


def test_reference_dimensions_are_exported_without_clipping_or_render_geometry(layout_scene):
    from decanting_workspace.layout_backend import LayoutPreviewBackend
    from decanting_workspace.layout_geometry import build_layout_geometry

    reference = LayoutReferenceDimensions('sample_tote', (60.,40.,30.), '/World/Group/small_KLT')
    scene = replace(layout_scene, reference_dimensions=(reference,))
    class Viewer:
        def render(self, rendered_scene, fragments):
            self.scene = rendered_scene
            self.fragments = fragments
    viewer = Viewer()
    backend = LayoutPreviewBackend(scene, viewer)
    assert viewer.fragments == build_layout_geometry(layout_scene)
    assert backend.catalog()['stats']['source_boxes'] == 2
    assert backend.catalog()['stats']['source_volume_m3'] == pytest.approx(7.)
    expected = [{'name': 'sample_tote', 'size_m': [60.,40.,30.],
                 'source_prim_path': '/World/Group/small_KLT', 'role': 'tote'}]
    assert backend.catalog()['reference_dimensions'] == expected
    assert backend.export_payload()['scene']['reference_dimensions'] == expected
    assert not any(node['prim_path'] == reference.source_prim_path
                   for node in backend.catalog()['nodes'])


@pytest.mark.parametrize('size', [(0.,.4,.3), (-.6,.4,.3), (math.nan,.4,.3), (.6,.4)])
def test_reference_dimensions_require_real_positive_sizes(size):
    with pytest.raises(ValueError):
        LayoutReferenceDimensions('tote', size, '/World/small_KLT')
