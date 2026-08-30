from __future__ import annotations

import json

from decanting_workspace.candidate_cli import build_parser, main
from decanting_workspace.models import load_scene_spec
from decanting_workspace.workflow import WORKFLOW_SCHEMA_VERSION


def test_candidate_cli_defaults_to_official_ur20_and_both_modes():
    args = build_parser().parse_args([])

    assert args.ur20_source == "official"
    assert args.coordination == "both"
    assert args.corner == "all"
    assert args.approach_mm == 150.0


def test_invalid_candidate_writes_compact_json_without_running_ik(tmp_path):
    spec = load_scene_spec()
    ur = spec.robots["ur20"].nominal_base
    sr = spec.robots["sr12ia"].nominal_base
    output = tmp_path / "invalid-candidate.json"

    exit_code = main(
        [
            "--ur20-source",
            "primitive",
            "--corner",
            "southwest",
            "--coordination",
            "sequential",
            "--ur-base",
            str(spec.installation_region.raw_xy_min_m[0]),
            str(ur.y_m),
            str(ur.z_m),
            str(ur.yaw_deg),
            "--sr-base",
            str(sr.x_m),
            str(sr.y_m),
            str(sr.z_m),
            str(sr.yaw_deg),
            "--output-json",
            str(output),
        ]
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["workflow_schema_version"] == WORKFLOW_SCHEMA_VERSION
    assert not report["placement"]["valid"]
    assert report["sr_cutting_workspace"] is None
    assert report["sr_concurrent_exclusion"] is None
    assert report["coordination"][0]["mode"] == "sequential"
    assert not report["coordination"][0]["cell_feasible"]
    assert report["coordination"][0]["scenarios"] == []
    assert report["evaluation_settings"]["ur20_urdf"].endswith(
        "ur20_primitive.urdf"
    )
