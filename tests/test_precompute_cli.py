from __future__ import annotations

import json
from pathlib import Path

import pytest

from decanting_workspace.models import load_scene_spec
from decanting_workspace.precompute_cli import (
    PrecomputeDependencies,
    build_parser,
    load_case_grid_yaml,
    main,
)


def _write_grid(path: Path, spec, *, extra_ur_x=(), modes=("sequential",)) -> Path:
    ur = spec.robots["ur20"].nominal_base
    sr = spec.robots["sr12ia"].nominal_base
    ur_x = ", ".join(str(value) for value in (ur.x_m, *extra_ur_x))
    mode_text = ", ".join(modes)
    path.write_text(
        f"""schema_version: 1
units: m
ur_base:
  x_m: [{ur_x}]
  y_m: [{ur.y_m}]
  z_m: [{ur.z_m}]
  yaw_deg: [{ur.yaw_deg}]
sr_base:
  x_m: [{sr.x_m}]
  y_m: [{sr.y_m}]
  z_m: [{sr.z_m}]
  yaw_deg: [{sr.yaw_deg}]
skus: [\"123591\"]
lift_heights_m: [0.0]
tote_offsets_m: [0.0]
corners: [southwest]
coordination_modes: [{mode_text}]
sr_j3_strokes_m: [0.30]
clearances_m: [null]
tote_present: true
""",
        encoding="utf-8",
    )
    return path


def _dependencies(spec, *, calls, cache=object()):
    def resolver():
        calls.append(("resolver",))
        return spec.robots["ur20"].urdf_path, (spec.source_path.parent,)

    def robot_loader(name, path, **kwargs):
        bundle = object()
        calls.append(("robot", name, Path(path), kwargs, bundle))
        return bundle

    def precomputer(received_spec, grid, ur_bundle, sr_bundle, **kwargs):
        calls.append(
            (
                "precompute",
                received_spec,
                grid,
                ur_bundle,
                sr_bundle,
                kwargs,
            )
        )
        return cache

    def saver(received_cache, path):
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"saved": True}), encoding="utf-8")
        calls.append(("save", received_cache, target))
        return target

    return PrecomputeDependencies(
        spec_loader=lambda path: spec,
        official_ur20_resolver=resolver,
        robot_loader=robot_loader,
        precomputer=precomputer,
        cache_saver=saver,
    )


def test_grid_argument_is_explicitly_required(tmp_path):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--output-json", str(tmp_path / "out.json")])


def test_yaml_axes_produce_the_exact_cartesian_case_count(tmp_path):
    spec = load_scene_spec()
    ur = spec.robots["ur20"].nominal_base
    path = _write_grid(
        tmp_path / "grid.yaml",
        spec,
        extra_ur_x=(ur.x_m + 0.01,),
        modes=("simultaneous", "sequential"),
    )
    text = path.read_text(encoding="utf-8").replace(
        'skus: ["123591"]',
        'skus: ["123591", "430060"]',
    ).replace(
        "corners: [southwest]",
        "corners: [southwest, northeast]",
    ).replace(
        "sr_j3_strokes_m: [0.30]",
        "sr_j3_strokes_m: [0.30, 0.45]",
    ).replace(
        "clearances_m: [null]",
        "clearances_m: [null, 0.01]",
    )
    path.write_text(text, encoding="utf-8")

    grid = load_case_grid_yaml(path)

    # 2 UR poses * 2 SKUs * 2 corners * 2 modes * 2 J3 * 2 clearances.
    assert grid.case_count == 64


def test_max_cases_guard_prints_exact_count_before_robot_loading(
    tmp_path,
    capsys,
):
    spec = load_scene_spec()
    ur = spec.robots["ur20"].nominal_base
    grid_path = _write_grid(
        tmp_path / "grid.yaml",
        spec,
        extra_ur_x=(ur.x_m + 0.01,),
    )
    calls = []
    deps = _dependencies(spec, calls=calls)
    output = tmp_path / "out.json"

    exit_code = main(
        [
            "--grid",
            str(grid_path),
            "--output-json",
            str(output),
            "--max-cases",
            "1",
            "--ur20-source",
            "primitive",
        ],
        dependencies=deps,
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Exact case count: 2" in captured.out
    assert "exceeding --max-cases 1" in captured.err
    assert not any(call[0] in {"resolver", "robot", "precompute", "save"} for call in calls)
    assert not output.exists()


def test_primitive_run_injects_sampling_options_and_saves_json(tmp_path, capsys):
    spec = load_scene_spec()
    grid_path = _write_grid(tmp_path / "grid.yaml", spec)
    output = tmp_path / "nested" / "cache.json"
    calls = []
    cache = object()
    deps = _dependencies(spec, calls=calls, cache=cache)

    exit_code = main(
        [
            "--grid",
            str(grid_path),
            "--output-json",
            str(output),
            "--ur20-source",
            "primitive",
            "--approach-mm",
            "200",
            "--cartesian-step-mm",
            "25",
            "--cartesian-rotation-step-deg",
            "2",
            "--joint-interpolation-samples",
            "5",
            "--characteristic-length-mm",
            "400",
            "--keepout-margin-mm",
            "30",
            "--position-tolerance-mm",
            "0.01",
            "--orientation-tolerance-deg",
            "0.02",
            "--ik-max-iterations",
            "123",
            "--ik-damping",
            "0.0002",
            "--ik-max-backtracking-steps",
            "9",
            "--extra-arm-seed-deg",
            "0",
            "-90",
            "90",
            "-90",
            "-90",
            "0",
        ],
        dependencies=deps,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Exact case count: 1" in captured.out
    assert "Saved 1 precomputed cases" in captured.out
    assert output.is_file()
    assert not any(call[0] == "resolver" for call in calls)
    robots = [call for call in calls if call[0] == "robot"]
    assert [call[1] for call in robots] == ["ur20", "sr12ia"]
    assert robots[0][2] == spec.robots["ur20"].urdf_path
    assert not robots[0][3]["load_visual"]
    precompute_call = next(call for call in calls if call[0] == "precompute")
    settings = precompute_call[5]
    options = settings["options"]
    assert settings["approach_distance_m"] == pytest.approx(0.2)
    assert settings["keepout_margin_m"] == pytest.approx(0.03)
    assert options.cartesian_translation_step_m == pytest.approx(0.025)
    assert options.cartesian_rotation_step_rad == pytest.approx(
        0.03490658503988659
    )
    assert options.joint_interpolation_samples == 5
    assert options.characteristic_length_m == pytest.approx(0.4)
    assert options.ik.position_tolerance_m == pytest.approx(1e-5)
    assert options.ik.orientation_tolerance_rad == pytest.approx(
        0.00034906585039886593
    )
    assert options.ik.max_iterations == 123
    assert options.ik.damping == pytest.approx(0.0002)
    assert options.ik.max_backtracking_steps == 9
    assert options.extra_arm_seeds[0] == pytest.approx(
        (0.0, -1.5707963267948966, 1.5707963267948966, -1.5707963267948966, -1.5707963267948966, 0.0)
    )
    save_call = next(call for call in calls if call[0] == "save")
    assert save_call[1] is cache


def test_official_source_uses_resolver_package_dirs_and_fingerprinted_path(tmp_path):
    spec = load_scene_spec()
    grid_path = _write_grid(tmp_path / "grid.yaml", spec)
    output = tmp_path / "cache.json"
    calls = []
    deps = _dependencies(spec, calls=calls)

    exit_code = main(
        ["--grid", str(grid_path), "--output-json", str(output)],
        dependencies=deps,
    )

    assert exit_code == 0
    assert sum(call[0] == "resolver" for call in calls) == 1
    ur_call = next(call for call in calls if call[:2] == ("robot", "ur20"))
    assert ur_call[3]["package_dirs"] == (spec.source_path.parent,)
    precompute_call = next(call for call in calls if call[0] == "precompute")
    evaluation_spec = precompute_call[1]
    assert evaluation_spec.robots["ur20"].urdf_path == spec.robots["ur20"].urdf_path


def test_cli_filters_exact_sku_and_lift_shards_before_counting(tmp_path, capsys):
    spec = load_scene_spec()
    path = _write_grid(tmp_path / "grid.yaml", spec)
    text = path.read_text(encoding="utf-8").replace(
        'skus: ["123591"]',
        'skus: ["123591", "430060"]',
    ).replace(
        "lift_heights_m: [0.0]",
        "lift_heights_m: [0.0, 0.38]",
    )
    path.write_text(text, encoding="utf-8")
    calls = []

    code = main(
        [
            "--grid",
            str(path),
            "--output-json",
            str(tmp_path / "cache.json"),
            "--ur20-source",
            "primitive",
            "--only-sku",
            "430060",
            "--only-lift-height-mm",
            "380",
        ],
        dependencies=_dependencies(spec, calls=calls),
    )

    assert code == 0
    assert "Exact case count: 1" in capsys.readouterr().out
    precompute_call = next(call for call in calls if call[0] == "precompute")
    filtered = precompute_call[2]
    assert filtered.skus == ("430060",)
    assert filtered.lift_heights_m == (0.38,)


@pytest.mark.parametrize(
    "replacement, message",
    [
        (("schema_version: 1", "schema_version: 2"), "schema_version"),
        (("units: m", "units: mm"), "units"),
        (("  x_m:", "  missing_x_m:"), "x_m"),
    ],
)
def test_invalid_yaml_grid_reports_cli_error_without_loading_robots(
    tmp_path,
    capsys,
    replacement,
    message,
):
    spec = load_scene_spec()
    path = _write_grid(tmp_path / "bad.yaml", spec)
    path.write_text(
        path.read_text(encoding="utf-8").replace(*replacement, 1),
        encoding="utf-8",
    )
    calls = []

    exit_code = main(
        ["--grid", str(path), "--output-json", str(tmp_path / "out.json")],
        dependencies=_dependencies(spec, calls=calls),
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert message in captured.err
    assert not any(call[0] in {"resolver", "robot", "precompute", "save"} for call in calls)
