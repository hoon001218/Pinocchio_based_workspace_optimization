from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from decanting_workspace.models import SceneState, load_scene_spec
from decanting_workspace.playback_cli import (
    PlaybackDependencies,
    build_parser,
    default_working_cache_path,
    main,
)
from decanting_workspace.precompute import BasePoseGrid, CaseGrid, CaseKey
from decanting_workspace.workflow import CoordinationMode


@dataclass
class _Case:
    key: CaseKey
    status: str = "evaluated"
    installation_valid: bool = True
    scenarios: tuple = ()
    coordination_summaries: tuple = ()


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
        self.control_url = "http://127.0.0.1:8765/"

    def serve_forever(self):
        self.calls.append(("serve",))

    def server_close(self):
        self.calls.append(("close",))


def _fake_cache(spec):
    ur = spec.robots["ur20"].nominal_base
    sr = spec.robots["sr12ia"].nominal_base
    grid = CaseGrid(
        ur_bases=BasePoseGrid.single(ur),
        sr_bases=BasePoseGrid.single(sr),
        lift_heights_m=(0.0,),
        tote_offsets_m=(0.0,),
        corners=("southwest",),
        coordination_modes=(CoordinationMode.SEQUENTIAL,),
    )
    key = CaseKey(
        ur,
        sr,
        0.0,
        0.0,
        "southwest",
        CoordinationMode.SEQUENTIAL,
        "123591",
        0.3,
        None,
    )
    return type("Cache", (), {"cases": (_Case(key),), "grid": grid})()


def test_cache_defaults_to_verified_working_setup():
    args = build_parser().parse_args([])

    assert args.cache == default_working_cache_path()
    assert args.cache.is_file()


def test_cli_loads_matching_models_renders_and_serves(tmp_path):
    spec = load_scene_spec()
    cache = _fake_cache(spec)
    calls = []
    viewer = _Viewer(calls)
    server = _Server(calls)
    prepared_sr = tmp_path / "sr12ia_mesh.urdf"
    prepared_sr.write_text("<robot name='display-only'/>", encoding="utf-8")

    def cache_loader(path, *, spec):
        calls.append(("cache", Path(path), spec))
        return cache

    deps = PlaybackDependencies(
        spec_loader=lambda path: spec,
        cache_loader=cache_loader,
        official_ur20_resolver=lambda: (_ for _ in ()).throw(AssertionError()),
        prepared_sr12ia_resolver=lambda: prepared_sr,
        viewer_factory=lambda: viewer,
        backend_factory=lambda *args, **kwargs: calls.append(("backend", args, kwargs)) or object(),
        server_factory=lambda *args: calls.append(("server", args)) or server,
        browser_open=lambda url: calls.append(("open", url)),
    )

    code = main(
        [
            "--cache",
            str(tmp_path / "cache.json"),
            "--ur20-source",
            "primitive",
            "--port",
            "0",
            "--open",
        ],
        dependencies=deps,
    )

    assert code == 0
    assert any(call[0] == "cache" for call in calls)
    render = next(call for call in calls if call[0] == "render")
    snapshot = render[1][1]
    assert snapshot.state == SceneState(corner_samples=("southwest",))
    cache_call = next(call for call in calls if call[0] == "cache")
    assert (
        cache_call[2].robots["sr12ia"].urdf_path
        == spec.robots["sr12ia"].urdf_path
    )
    assert render[2]["sr12ia_urdf"] == prepared_sr
    assert render[2]["sr12ia_visual_only"] is True
    backend = next(call for call in calls if call[0] == "backend")
    assert backend[1][0].robots["sr12ia"].urdf_path == spec.robots["sr12ia"].urdf_path
    assert ("serve",) in calls
    assert ("open", server.control_url) in calls
    assert calls[-1] == ("close",)


def test_cli_rejects_empty_cache_before_viewer_creation(tmp_path, capsys):
    spec = load_scene_spec()
    created = []
    empty = type("Cache", (), {"cases": ()})()
    deps = PlaybackDependencies(
        spec_loader=lambda path: spec,
        cache_loader=lambda *args, **kwargs: empty,
        official_ur20_resolver=lambda: (spec.robots["ur20"].urdf_path, ()),
        prepared_sr12ia_resolver=lambda: spec.robots["sr12ia"].urdf_path,
        viewer_factory=lambda: created.append(True),
        backend_factory=lambda *args, **kwargs: object(),
        server_factory=lambda *args: object(),
        browser_open=lambda url: None,
    )

    code = main(
        ["--cache", str(tmp_path / "empty.json")],
        dependencies=deps,
    )

    assert code == 2
    assert "contains no cases" in capsys.readouterr().err
    assert created == []


def test_explicit_visual_sr_does_not_change_evaluation_fingerprint_spec(
    tmp_path,
):
    spec = load_scene_spec()
    cache = _fake_cache(spec)
    calls = []
    viewer = _Viewer(calls)
    server = _Server(calls)
    visual = tmp_path / "manufacturer_visual.urdf"
    visual.write_text("<robot name='manufacturer-visual'/>", encoding="utf-8")

    def cache_loader(path, *, spec):
        calls.append(("cache", spec))
        return cache

    deps = PlaybackDependencies(
        spec_loader=lambda path: spec,
        cache_loader=cache_loader,
        official_ur20_resolver=lambda: (
            spec.robots["ur20"].urdf_path,
            (),
        ),
        prepared_sr12ia_resolver=lambda: (_ for _ in ()).throw(
            AssertionError("explicit visual path must win")
        ),
        viewer_factory=lambda: viewer,
        backend_factory=lambda *args, **kwargs: object(),
        server_factory=lambda *args: server,
        browser_open=lambda url: None,
    )

    code = main(
        [
            "--cache",
            str(tmp_path / "cache.json"),
            "--ur20-source",
            "primitive",
            "--sr12ia-visual-urdf",
            str(visual),
            "--port",
            "0",
        ],
        dependencies=deps,
    )

    assert code == 0
    evaluation_spec = next(call[1] for call in calls if call[0] == "cache")
    assert (
        evaluation_spec.robots["sr12ia"].urdf_path
        == spec.robots["sr12ia"].urdf_path
    )
    render = next(call for call in calls if call[0] == "render")
    assert render[2]["sr12ia_urdf"] == visual
    assert render[2]["sr12ia_visual_only"] is True


def test_missing_prepared_sr_mesh_falls_back_to_evaluation_visual(
    tmp_path,
    capsys,
):
    spec = load_scene_spec()
    cache = _fake_cache(spec)
    calls = []
    viewer = _Viewer(calls)
    server = _Server(calls)
    deps = PlaybackDependencies(
        spec_loader=lambda path: spec,
        cache_loader=lambda *args, **kwargs: cache,
        official_ur20_resolver=lambda: (
            spec.robots["ur20"].urdf_path,
            (),
        ),
        prepared_sr12ia_resolver=lambda: (_ for _ in ()).throw(
            FileNotFoundError("prepared mesh missing")
        ),
        viewer_factory=lambda: viewer,
        backend_factory=lambda *args, **kwargs: object(),
        server_factory=lambda *args: server,
        browser_open=lambda url: None,
    )

    code = main(
        [
            "--cache",
            str(tmp_path / "cache.json"),
            "--ur20-source",
            "primitive",
            "--port",
            "0",
        ],
        dependencies=deps,
    )

    assert code == 0
    render = next(call for call in calls if call[0] == "render")
    assert render[2]["sr12ia_urdf"] == spec.robots["sr12ia"].urdf_path
    assert render[2]["sr12ia_visual_only"] is False
    assert "displaying the evaluation model" in capsys.readouterr().err
