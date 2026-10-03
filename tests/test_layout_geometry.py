from dataclasses import replace
import math
from pathlib import Path

import pytest

from decanting_workspace.layout import apply_layout_offsets
from decanting_workspace.layout_geometry import build_layout_geometry, fragment_volume_m3
from decanting_workspace.layout_models import LayoutBox, LayoutNode, LayoutScene


def _scene(second_x=1., second_rotation=(0., 0., 0., 1.)):
    nodes = tuple(LayoutNode(path, None, (0., 0., 0.), (0., 0., 0., 1.)) for path in ("/A", "/B"))
    return LayoutScene(Path("scene.usd"), nodes, (
        LayoutBox("A", "/A", (0., 0., 0.), (2., 2., 2.), (0., 0., 0., 1.)),
        LayoutBox("B", "/B", (second_x, 0., 0.), (2., 2., 2.), second_rotation),
    ), ())


def _volume(scene):
    return sum(fragment_volume_m3(fragment) for fragment in build_layout_geometry(scene))


def test_overlap_volume_exists_once_with_first_owner():
    scene = _scene()
    fragments = build_layout_geometry(scene)
    assert _volume(scene) == pytest.approx(12.)
    assert sum(fragment_volume_m3(f) for f in fragments if f.owner_path == "/A") == pytest.approx(8.)
    assert sum(fragment_volume_m3(f) for f in fragments if f.owner_path == "/B") == pytest.approx(4.)
    assert scene.boxes[1].size_m == (2., 2., 2.)


def test_contained_duplicate_is_removed_and_restored_after_moving():
    original = _scene(0.)
    assert {f.owner_path for f in build_layout_geometry(original)} == {"/A"}
    moved = apply_layout_offsets(original, {"/B": (3., 0., 0.)})
    assert _volume(moved) == pytest.approx(16.)
    assert _volume(original) == pytest.approx(8.)


def test_contact_without_volume_overlap_keeps_both_solids():
    assert _volume(_scene(2.)) == pytest.approx(16.)
    assert len(build_layout_geometry(_scene(2.))) == 2


def test_rotated_overlap_is_clipped_without_replacing_it_with_an_aabb():
    angle = math.pi / 8.
    scene = _scene(0., (0., 0., math.sin(angle), math.cos(angle)))
    # Intersection of two centred side-2 squares rotated 45deg, extruded by 2m.
    expected = 16. - 16. * (math.sqrt(2.) - 1.)
    assert _volume(scene) == pytest.approx(expected)


def test_inclined_proxy_keeps_its_solid_volume():
    angle = math.pi / 12.
    scene = _scene(4., (0., math.sin(angle), 0., math.cos(angle)))
    assert _volume(scene) == pytest.approx(16.)


def test_multiple_overlaps_form_a_union_instead_of_losing_or_duplicating_volume():
    scene = _scene(1.)
    scene = replace(scene, nodes=scene.nodes + (LayoutNode("/C", None, (0.,0.,0.), (0.,0.,0.,1.)),),
                    boxes=scene.boxes + (LayoutBox("C", "/C", (.5,0.,0.), (2.,2.,2.), (0.,0.,0.,1.)),))
    assert _volume(scene) == pytest.approx(12.)
