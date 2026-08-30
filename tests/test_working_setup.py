from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path

import pytest

from decanting_workspace.precompute_cli import load_case_grid_yaml


_ROOT = Path(__file__).resolve().parents[1]


def test_verified_working_grid_and_cache_cover_the_declared_setup():
    grid = load_case_grid_yaml(_ROOT / "config" / "case_grid_working.yaml")

    assert grid.case_count == 40
    assert grid.skus == ("123591", "430060", "180043", "049995", "000509")
    assert grid.lift_heights_m == pytest.approx((0.2,))
    assert grid.tote_offsets_m == pytest.approx((0.7,))
    assert [mode.value for mode in grid.coordination_modes] == [
        "sequential",
        "simultaneous",
    ]

    with gzip.open(
        _ROOT / "outputs" / "precomputed_working_cases.json.gz",
        "rt",
        encoding="utf-8",
    ) as stream:
        cache = json.load(stream)

    assert cache["schema_version"] == 2
    assert cache["settings"]["cartesian_translation_step_m"] == pytest.approx(0.05)
    assert cache["settings"]["cartesian_rotation_step_rad"] == pytest.approx(
        0.08726646259971647
    )
    assert cache["settings"]["joint_interpolation_samples"] == 3
    assert len(cache["cases"]) == 40

    sequential = [
        case
        for case in cache["cases"]
        if case["key"]["coordination_mode"] == "sequential"
    ]
    assert len(sequential) == 20
    assert all(
        case["coordination_summaries"][0]["hard_feasible"]
        for case in sequential
    )
    feasible_skus = Counter(
        case["key"]["sku"]
        for case in sequential
        if case["coordination_summaries"][0]["cell_feasible"]
    )
    assert feasible_skus == Counter({"123591": 4, "430060": 4, "000509": 4})

    hard_samples = [
        sample
        for case in sequential
        for step in case["scenarios"][0]["steps"]
        if step["criticality"] == "hard"
        for check in step["checks"]
        for sample in check["samples"]
    ]
    assert len(hard_samples) == 1434
    assert all(sample["ik_status"] == "success" for sample in hard_samples)
    assert all(sample["manipulability"] is not None for sample in hard_samples)

    final_checks = [
        case["scenarios"][0]["steps"][-1]["checks"][-1]
        for case in sequential
    ]
    assert {check["name"] for check in final_checks} == {
        "filled_tote_final_pose_at_supply"
    }
    assert all(check["status"] == "success" for check in final_checks)
    assert all(len(check["samples"]) == 1 for check in final_checks)
    assert all(not check["include_in_metric_summary"] for check in final_checks)
