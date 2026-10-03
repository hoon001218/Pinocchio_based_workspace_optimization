"""Match the A25 high endpoint to the fixed top conveyor; preserve other units.

Run from any directory with the repository's Pinocchio/USD Python environment.
The collected USD is local-only, so keep this script with a freshly downloaded
asset tree. Only the A25 assembly's existing translation is authored and saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from decanting_workspace.layout_models import LayoutBox, LayoutScene
from decanting_workspace.models import load_scene_spec
from decanting_workspace.transforms import quaternion_matrix
from decanting_workspace.usd_layout import load_usd_layout
from decanting_workspace.usd_scene import _usd_open_path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "USD/Collected_scene_layout/scene_layout.usd"
DEFAULT_BACKUP = ROOT / ".cache/layout_source_backup/scene_layout.before_conveyor_connection.usd"
DEFAULT_REPORT = ROOT / "outputs/layout_conveyor_alignment.json"
MOVING = "/World/ConveyorGroup/ConveyorBelt_A25"
HIGH_SURFACE = MOVING + "/Belt/SM_ConveyorBelt_A25_Belt01_02"
ANCHOR = "/World/ConveyorGroup/conv_top/ConveyorBelt_A05_PR_NVD_04"
ANCHOR_SURFACE = ANCHOR + "/roller_surface"
TOLERANCE_M = 1e-6


def box_bounds(box: LayoutBox) -> np.ndarray:
    rotation = quaternion_matrix((0., 0., 0.), box.rotation_xyzw)[:3, :3]
    half = np.asarray(box.size_m) / 2.
    points = np.array([(x, y, z) for x in (-half[0], half[0])
                       for y in (-half[1], half[1]) for z in (-half[2], half[2])])
    points = points @ rotation.T + box.center_m
    return np.array((points.min(axis=0), points.max(axis=0)))


def plan_alignment(scene: LayoutScene) -> dict:
    boxes = {box.name: box for box in scene.boxes}
    high, anchor = boxes[HIGH_SURFACE], boxes[ANCHOR_SURFACE]
    high_bounds, anchor_bounds = box_bounds(high), box_bounds(anchor)
    overlap = np.minimum(high_bounds[1, :2], anchor_bounds[1, :2]) - np.maximum(
        high_bounds[0, :2], anchor_bounds[0, :2])
    if np.any(overlap <= TOLERANCE_M):
        raise ValueError("A25 high endpoint must already overlap the fixed top conveyor in XY")
    delta_z = float(anchor_bounds[1, 2] - high_bounds[1, 2])
    if abs(delta_z) <= TOLERANCE_M:
        delta_z = 0.
    return {"world_delta_m": [0., 0., delta_z],
            "anchor_top_z_m": float(anchor_bounds[1, 2]),
            "a25_high_top_z_m": float(high_bounds[1, 2]),
            "xy_overlap_m": overlap.tolist()}


def _read_layout(source: Path) -> LayoutScene:
    return load_usd_layout(source, load_scene_spec().robots)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root + "/")


def _verify_preserved(before: LayoutScene, after: LayoutScene) -> dict:
    preserved = {}
    for collection, key in (("nodes", "prim_path"), ("boxes", "node_path")):
        old = tuple(value for value in getattr(before, collection)
                    if not _under(getattr(value, key), MOVING))
        new = tuple(value for value in getattr(after, collection)
                    if not _under(getattr(value, key), MOVING))
        if old != new:
            raise ValueError(f"alignment unexpectedly changed unrelated {collection}")
        preserved[collection] = len(old)
    for collection in ("robots", "frames", "reference_dimensions"):
        if getattr(before, collection) != getattr(after, collection):
            raise ValueError(f"alignment unexpectedly changed {collection}")
        preserved[collection] = len(getattr(before, collection))
    return preserved


def align_source(source: Path = DEFAULT_SOURCE, *, backup: Path = DEFAULT_BACKUP,
                 report: Path = DEFAULT_REPORT) -> dict:
    from pxr import Gf, Sdf, Usd, UsdGeom

    source, backup, report = (Path(path).expanduser().resolve() for path in (source, backup, report))
    if backup == source or report in (source, backup):
        raise ValueError("source, backup and report must be distinct files")
    before = _read_layout(source)
    plan = plan_alignment(before)
    baseline_hash = _sha256(source)
    backup.parent.mkdir(parents=True, exist_ok=True)
    # An exclusive create preserves the first source bytes on repeated runs.
    try:
        with backup.open("xb") as target, source.open("rb") as original:
            shutil.copyfileobj(original, target)
    except FileExistsError:
        pass

    stage = Usd.Stage.Open(_usd_open_path(source), load=Usd.Stage.LoadNone)
    stage.Load(MOVING)
    prim = stage.GetPrimAtPath(MOVING)
    attribute = prim.GetAttribute("xformOp:translate")
    if not attribute or attribute.Get() is None:
        raise ValueError("A25 assembly needs an existing xformOp:translate")
    original = np.asarray(attribute.Get(), dtype=float)
    parent = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent()), dtype=float)
    metres = float(UsdGeom.GetStageMetersPerUnit(stage))
    up = UsdGeom.GetStageUpAxis(stage)
    basis = np.eye(3) if up == "Z" else np.array(((1., 0., 0.), (0., 0., -1.), (0., 1., 0.)))
    local_delta = np.linalg.solve(basis @ parent[:3, :3].T * metres, plan["world_delta_m"])
    updated = original + local_delta
    changed = bool(np.any(local_delta != 0.))
    if changed:
        # Never edit referenced asset layers, orientation, scale or xform order.
        # Saving an NTFS 8.3 file alias can rename the original long filename.
        # Read dependencies through the alias; save only the canonical root.
        root_layer = Sdf.Layer.FindOrOpen(str(source))
        root_attribute = root_layer.GetAttributeAtPath(MOVING + ".xformOp:translate")
        if root_attribute is None:
            raise ValueError("A25 translation must already be authored in the root layer")
        root_attribute.default = Gf.Vec3d(*updated)
        root_layer.Save()
        if Path(stage.GetRootLayer().identifier).is_file():
            stage.GetRootLayer().Reload(force=True)
    after = _read_layout(source)
    preserved = _verify_preserved(before, after)
    checked = plan_alignment(after)
    if checked["world_delta_m"][2] != 0.:
        raise ValueError("saved A25 endpoint does not match the fixed conveyor height")
    result = {"schema_version": 1, "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
              "backup": str(backup.relative_to(ROOT)) if backup.is_relative_to(ROOT) else str(backup),
              "moving_prim": MOVING, "anchor_prim": ANCHOR,
              "moving_surface": HIGH_SURFACE, "anchor_surface": ANCHOR_SURFACE,
              "tolerance_m": TOLERANCE_M, "changed": changed,
              "original_translate": original.tolist(), "updated_translate": updated.tolist(),
              "local_delta": local_delta.tolist(), "world_delta_m": plan["world_delta_m"],
              "before": plan, "after": checked, "preserved": preserved,
              "baseline_source_sha256": baseline_hash,
              "updated_source_sha256": _sha256(source), "backup_sha256": _sha256(backup)}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--backup", type=Path, default=DEFAULT_BACKUP)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = align_source(args.source, backup=args.backup, report=args.report)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
