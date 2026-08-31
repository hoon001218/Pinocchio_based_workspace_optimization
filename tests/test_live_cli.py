from __future__ import annotations

import math
from pathlib import Path

import pytest

from decanting_workspace.live_cli import (
    LiveDependencies,
    _evaluation_profiles,
    build_parser,
    default_initial_grid_path,
    main,
)
from decanting_workspace.live_ui import build_live_control_page
from decanting_workspace.models import SceneState, load_scene_spec
from decanting_workspace.precompute import iter_case_keys
from decanting_workspace.precompute_cli import load_case_grid_yaml


class _Viewer:
    def __init__(self, calls):
        self.calls = calls

    def render(self, *args, **kwargs):
        self.calls.append(("render", args, kwargs))

    def url(self):
        return "http://127.0.0.1:7000/static/"


class _Server:
    def __init__(self, calls):
        self.calls = calls
        self.control_url = "http://127.0.0.1:8766/"

    def serve_forever(self):
        self.calls.append(("serve",))

    def server_close(self):
        self.calls.append(("close",))


def test_parser_defaults_to_working_grid_quick_profile_and_distinct_port():
    args = build_parser().parse_args([])

    assert args.initial_grid == default_initial_grid_path()
    assert args.initial_grid.is_file()
    assert args.default_profile == "quick"
    assert args.host == "127.0.0.1"
    assert args.port == 8766
    assert args.cartesian_step_mm == pytest.approx(50.0)
    assert args.cartesian_rotation_step_deg == pytest.approx(5.0)
    assert args.joint_interpolation_samples == 3
    assert not hasattr(args, "cache")


def test_profiles_share_ik_and_metric_settings_but_keep_sampling_contracts():
    args = build_parser().parse_args(
        [
            "--default-profile",
            "full",
            "--cartesian-step-mm",
            "25",
            "--cartesian-rotation-step-deg",
            "2.5",
            "--joint-interpolation-samples",
            "7",
            "--characteristic-length-mm",
            "420",
            "--position-tolerance-mm",
            "0.02",
            "--orientation-tolerance-deg",
            "0.03",
            "--ik-max-iterations",
            "321",
            "--ik-damping",
            "0.0002",
            "--ik-max-backtracking-steps",
            "9",
            "--extra-arm-seed-deg",
            "10",
            "20",
            "30",
            "40",
            "50",
            "60",
        ]
    )

    quick, full = _evaluation_profiles(args)

    assert args.default_profile == "full"
    assert (quick.profile_id, full.profile_id) == ("quick", "full")
    assert quick.options.cartesian_translation_step_m == pytest.approx(10.0)
    assert quick.options.cartesian_rotation_step_rad == pytest.approx(math.pi)
    assert quick.options.joint_interpolation_samples == 0
    assert full.options.cartesian_translation_step_m == pytest.approx(0.025)
    assert full.options.cartesian_rotation_step_rad == pytest.approx(
        math.radians(2.5)
    )
    assert full.options.joint_interpolation_samples == 7
    assert quick.options.characteristic_length_m == pytest.approx(0.420)
    assert full.options.characteristic_length_m == pytest.approx(0.420)
    assert quick.options.ik is full.options.ik
    assert quick.options.ik.position_tolerance_m == pytest.approx(0.00002)
    assert quick.options.ik.orientation_tolerance_rad == pytest.approx(
        math.radians(0.03)
    )
    assert quick.options.ik.max_iterations == 321
    assert quick.options.ik.damping == pytest.approx(0.0002)
    assert quick.options.ik.max_backtracking_steps == 9
    assert quick.options.extra_arm_seeds == full.options.extra_arm_seeds
    assert quick.options.extra_arm_seeds[0] == pytest.approx(
        tuple(math.radians(value) for value in (10, 20, 30, 40, 50, 60))
    )


def test_cli_loads_evaluation_models_once_and_keeps_fanuc_display_separate(
    tmp_path,
    capsys,
):
    source_spec = load_scene_spec()
    grid = load_case_grid_yaml(default_initial_grid_path())
    calls = []
    viewer = _Viewer(calls)
    server = _Server(calls)
    official_ur = tmp_path / "official_ur20.urdf"
    evaluation_sr = tmp_path / "sr_evaluation_proxy.urdf"
    prepared_sr = tmp_path / "fanuc_display.urdf"
    for path in (official_ur, evaluation_sr, prepared_sr):
        path.write_text("<robot name='test'/>", encoding="utf-8")
    ur_bundle = object()
    sr_bundle = object()

    def grid_loader(path):
        calls.append(("grid", Path(path)))
        return grid

    def robot_loader(name, urdf, **kwargs):
        calls.append(("robot", name, Path(urdf), kwargs))
        return ur_bundle if name == "ur20" else sr_bundle

    def backend_factory(*args, **kwargs):
        calls.append(("backend", args, kwargs))
        return object()

    def server_factory(*args, **kwargs):
        calls.append(("server", args, kwargs))
        return server

    deps = LiveDependencies(
        spec_loader=lambda path: source_spec,
        grid_loader=grid_loader,
        official_ur20_resolver=lambda: (official_ur, (tmp_path / "packages",)),
        robot_loader=robot_loader,
        prepared_sr12ia_resolver=lambda: prepared_sr,
        viewer_factory=lambda: viewer,
        backend_factory=backend_factory,
        server_factory=server_factory,
        browser_open=lambda url: calls.append(("open", url)) or False,
    )

    code = main(
        [
            "--sr12ia-urdf",
            str(evaluation_sr),
            "--default-profile",
            "full",
            "--approach-mm",
            "125",
            "--keepout-margin-mm",
            "8",
            "--show-collisions",
            "--hide-frames",
            "--open",
            "--port",
            "0",
        ],
        dependencies=deps,
    )

    assert code == 0
    assert next(call for call in calls if call[0] == "grid")[1] == (
        default_initial_grid_path()
    )
    robot_calls = [call for call in calls if call[0] == "robot"]
    assert len(robot_calls) == 2
    assert [call[1] for call in robot_calls] == ["ur20", "sr12ia"]
    assert robot_calls[0][2] == official_ur
    assert robot_calls[0][3]["package_dirs"] == (tmp_path / "packages",)
    assert robot_calls[0][3]["load_visual"] is False
    assert robot_calls[0][3]["floating_base"] is True
    assert robot_calls[0][3]["suction_proxy"] is not None
    assert robot_calls[1][2] == evaluation_sr
    assert robot_calls[1][3]["load_visual"] is False
    assert robot_calls[1][3]["floating_base"] is True

    render = next(call for call in calls if call[0] == "render")
    render_spec, snapshot = render[1]
    expected_initial = next(iter_case_keys(render_spec, grid))
    assert render_spec.robots["ur20"].urdf_path == official_ur.resolve()
    assert render_spec.robots["sr12ia"].urdf_path == evaluation_sr.resolve()
    assert snapshot.state == SceneState(
        lift_height_m=expected_initial.lift_height_m,
        sku=expected_initial.sku,
        corner_samples=(expected_initial.corner,),
        tote_present=True,
        tote_long_axis_offset_m=expected_initial.tote_offset_m,
        sr_j3_stroke_m=expected_initial.sr_j3_stroke_m,
        clearance_m=expected_initial.clearance_m,
    )
    assert render[2]["ur20_urdf"] == official_ur
    assert render[2]["sr12ia_urdf"] == prepared_sr
    assert render[2]["sr12ia_visual_only"] is True
    assert render[2]["show_collisions"] is True
    assert render[2]["show_frames"] is False

    backend = next(call for call in calls if call[0] == "backend")
    assert backend[1][0] is render_spec
    assert backend[1][1] == expected_initial
    assert backend[1][2:5] == (ur_bundle, sr_bundle, viewer)
    profiles = backend[2]["profiles"]
    assert tuple(profile.profile_id for profile in profiles) == ("quick", "full")
    assert backend[2]["default_profile"] == "full"
    assert backend[2]["approach_distance_m"] == pytest.approx(0.125)
    assert backend[2]["keepout_margin_m"] == pytest.approx(0.008)

    server_call = next(call for call in calls if call[0] == "server")
    assert server_call[1][0] == ("127.0.0.1", 0)
    assert server_call[1][2] == viewer.url()
    assert server_call[2]["page_builder"] is build_live_control_page
    assert ("open", server.control_url) in calls
    assert ("serve",) in calls
    assert calls[-1] == ("close",)
    captured = capsys.readouterr()
    assert "LIVE CONTROL UI (parameters + calculated results):" in captured.out
    assert server.control_url in captured.out
    assert "Embedded MeshCat viewer only (no controls):" in captured.out
    assert "OPEN OR REFRESH THE LIVE CONTROL UI" in captured.out
    assert "browser did not open automatically" in captured.err


def test_explicit_urdf_skips_official_resolution(tmp_path):
    spec = load_scene_spec()
    grid = load_case_grid_yaml(default_initial_grid_path())
    calls = []
    viewer = _Viewer(calls)
    server = _Server(calls)
    explicit_ur = tmp_path / "explicit_ur.urdf"
    visual_sr = tmp_path / "visual_sr.urdf"
    explicit_ur.write_text("<robot name='ur'/>", encoding="utf-8")
    visual_sr.write_text("<robot name='sr'/>", encoding="utf-8")

    deps = LiveDependencies(
        spec_loader=lambda path: spec,
        grid_loader=lambda path: grid,
        official_ur20_resolver=lambda: (_ for _ in ()).throw(
            AssertionError("explicit URDF must win")
        ),
        robot_loader=lambda name, urdf, **kwargs: calls.append(
            ("robot", name, Path(urdf), kwargs)
        )
        or object(),
        prepared_sr12ia_resolver=lambda: (_ for _ in ()).throw(
            AssertionError("explicit visual must win")
        ),
        viewer_factory=lambda: viewer,
        backend_factory=lambda *args, **kwargs: object(),
        server_factory=lambda *args, **kwargs: server,
        browser_open=lambda url: None,
    )

    code = main(
        [
            "--ur20-urdf",
            str(explicit_ur),
            "--sr12ia-visual-urdf",
            str(visual_sr),
            "--port",
            "0",
        ],
        dependencies=deps,
    )

    assert code == 0
    ur_call = next(call for call in calls if call[:2] == ("robot", "ur20"))
    assert ur_call[2] == explicit_ur
    assert ur_call[3]["package_dirs"] == ()
    render = next(call for call in calls if call[0] == "render")
    assert render[2]["ur20_urdf"] == explicit_ur
    assert render[2]["sr12ia_urdf"] == visual_sr
    assert render[2]["sr12ia_visual_only"] is True


def test_missing_prepared_sr_mesh_fails_without_explicit_fallback(
    capsys,
):
    spec = load_scene_spec()
    grid = load_case_grid_yaml(default_initial_grid_path())
    created = []
    deps = LiveDependencies(
        spec_loader=lambda path: spec,
        grid_loader=lambda path: grid,
        official_ur20_resolver=lambda: (
            spec.robots["ur20"].urdf_path,
            (),
        ),
        robot_loader=lambda *args, **kwargs: object(),
        prepared_sr12ia_resolver=lambda: (_ for _ in ()).throw(
            FileNotFoundError("prepared mesh missing")
        ),
        viewer_factory=lambda: created.append(True),
        backend_factory=lambda *args, **kwargs: object(),
        server_factory=lambda *args, **kwargs: object(),
        browser_open=lambda url: None,
    )

    code = main(["--port", "0"], dependencies=deps)

    assert code == 2
    assert created == []
    assert "prepared FANUC SR-12iA visual asset is required" in (
        capsys.readouterr().err
    )


def test_missing_prepared_sr_mesh_falls_back_only_when_requested(capsys):
    spec = load_scene_spec()
    grid = load_case_grid_yaml(default_initial_grid_path())
    calls = []
    viewer = _Viewer(calls)
    server = _Server(calls)
    deps = LiveDependencies(
        spec_loader=lambda path: spec,
        grid_loader=lambda path: grid,
        official_ur20_resolver=lambda: (
            spec.robots["ur20"].urdf_path,
            (),
        ),
        robot_loader=lambda *args, **kwargs: object(),
        prepared_sr12ia_resolver=lambda: (_ for _ in ()).throw(
            FileNotFoundError("prepared mesh missing")
        ),
        viewer_factory=lambda: viewer,
        backend_factory=lambda *args, **kwargs: object(),
        server_factory=lambda *args, **kwargs: server,
        browser_open=lambda url: None,
    )

    code = main(
        ["--allow-sr-visual-fallback", "--port", "0"],
        dependencies=deps,
    )

    assert code == 0
    render = next(call for call in calls if call[0] == "render")
    assert render[2]["sr12ia_urdf"] == spec.robots["sr12ia"].urdf_path
    assert render[2]["sr12ia_visual_only"] is False
    assert "--allow-sr-visual-fallback requested" in capsys.readouterr().err
    assert calls[-1] == ("close",)


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (["--cartesian-step-mm", "0"], "finite and positive"),
        (["--joint-interpolation-samples", "-1"], "non-negative"),
        (["--port", "65536"], "port must be in"),
    ),
)
def test_parser_rejects_invalid_live_runtime_settings(arguments, message, capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(arguments)
    assert message in capsys.readouterr().err
