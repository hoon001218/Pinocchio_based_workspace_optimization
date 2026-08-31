from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from decanting_workspace.evaluation import EvaluationOptions
from decanting_workspace.kinematics import IKOptions
from decanting_workspace.live_backend import (
    EvaluationProfile,
    LiveEvaluationBackend,
)
from decanting_workspace.models import BasePose, load_scene_spec
from decanting_workspace.precompute import CaseKey
from decanting_workspace.workflow import CoordinationMode


class _RecordingPrecomputer:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.caches: list[object] = []

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, dict(kwargs)))
        cache = SimpleNamespace(cases=(object(),))
        self.caches.append(cache)
        return cache


class _PlaybackBackend:
    def __init__(self, case_id: str) -> None:
        self.case_id = case_id
        self.select_calls: list[dict[str, object]] = []

    def catalog(self) -> dict[str, object]:
        return {
            "cases": [
                {
                    "id": self.case_id,
                    "status": "evaluated",
                    "installation_valid": True,
                    "steps": [],
                }
            ]
        }

    def select(self, request: object) -> dict[str, object]:
        received = dict(request)
        self.select_calls.append(received)
        return {
            "case_id": received.get("case_id"),
            "step_status": "success",
            "request": received,
        }


class _BackendFactory:
    def __init__(self, case_ids: tuple[str, ...] = ("case-1",)) -> None:
        self.case_ids = iter(case_ids)
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.backends: list[_PlaybackBackend] = []

    def __call__(self, *args: object, **kwargs: object) -> _PlaybackBackend:
        self.calls.append((args, dict(kwargs)))
        backend = _PlaybackBackend(next(self.case_ids))
        self.backends.append(backend)
        return backend


class _Clock:
    def __init__(self, *values: float) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


@pytest.fixture(scope="module")
def spec():
    return load_scene_spec()


def _initial_key(spec) -> CaseKey:
    return CaseKey(
        ur_base=spec.robots["ur20"].nominal_base,
        sr_base=spec.robots["sr12ia"].nominal_base,
        lift_height_m=0.0,
        tote_offset_m=0.0,
        corner="southwest",
        coordination_mode=CoordinationMode.SEQUENTIAL,
        sku="123591",
        sr_j3_stroke_m=0.30,
        clearance_m=None,
    )


def _profiles() -> tuple[EvaluationProfile, ...]:
    return (
        EvaluationProfile(
            "fast",
            "Fast",
            "Coarse interactive evaluation",
            EvaluationOptions(
                cartesian_translation_step_m=0.10,
                cartesian_rotation_step_rad=math.radians(10.0),
                joint_interpolation_samples=0,
                ik=IKOptions(max_iterations=75),
            ),
        ),
        EvaluationProfile(
            "precise",
            "Precise",
            "Default production sampling",
            EvaluationOptions(),
        ),
    )


def _harness(
    spec,
    *,
    case_ids: tuple[str, ...] = ("case-1",),
    clock_values: tuple[float, ...] = (10.0, 12.5),
):
    precomputer = _RecordingPrecomputer()
    backend_factory = _BackendFactory(case_ids)
    ur_bundle = object()
    sr_bundle = object()
    viewer = object()
    profiles = _profiles()
    backend = LiveEvaluationBackend(
        spec,
        _initial_key(spec),
        ur_bundle,
        sr_bundle,
        viewer,
        profiles=profiles,
        default_profile="fast",
        approach_distance_m=0.18,
        keepout_margin_m=0.025,
        precomputer=precomputer,
        backend_factory=backend_factory,
        clock=_Clock(*clock_values),
    )
    return SimpleNamespace(
        backend=backend,
        precomputer=precomputer,
        backend_factory=backend_factory,
        ur_bundle=ur_bundle,
        sr_bundle=sr_bundle,
        viewer=viewer,
        profiles=profiles,
    )


def _parameters(backend: LiveEvaluationBackend) -> dict[str, object]:
    return dict(backend.catalog()["initial_parameters"])


def test_catalog_describes_live_defaults_schema_and_profiles(spec):
    harness = _harness(spec)

    catalog = harness.backend.catalog()

    assert catalog["schema_version"] == 1
    assert catalog["mode"] == "live"
    assert catalog["default_profile"] == "fast"
    assert catalog["result_persistence"] == "memory_only"
    assert catalog["evaluation_serialized"] is True
    assert catalog["initial_parameters"] == {
        "ur_x": spec.robots["ur20"].nominal_base.x_m,
        "ur_y": spec.robots["ur20"].nominal_base.y_m,
        "ur_z": spec.robots["ur20"].nominal_base.z_m,
        "ur_yaw": spec.robots["ur20"].nominal_base.yaw_deg,
        "sr_x": spec.robots["sr12ia"].nominal_base.x_m,
        "sr_y": spec.robots["sr12ia"].nominal_base.y_m,
        "sr_z": spec.robots["sr12ia"].nominal_base.z_m,
        "sr_yaw": spec.robots["sr12ia"].nominal_base.yaw_deg,
        "sku": "123591",
        "lift": 0.0,
        "tote": 0.0,
        "stroke": 0.30,
        "clearance": None,
        "corner": "southwest",
        "mode": "sequential",
    }

    schema = catalog["parameter_schema"]
    assert set(schema) == {
        "ur_x",
        "ur_y",
        "ur_z",
        "ur_yaw",
        "sr_x",
        "sr_y",
        "sr_z",
        "sr_yaw",
        "sku",
        "lift",
        "tote",
        "stroke",
        "clearance",
        "corner",
        "mode",
    }
    assert schema["ur_x"]["min"] == spec.installation_region.raw_xy_min_m[0]
    assert schema["ur_x"]["max"] == spec.installation_region.raw_xy_max_m[0]
    assert schema["clearance"]["nullable"] is True
    assert [option["value"] for option in schema["stroke"]["options"]] == list(
        spec.robots["sr12ia"].workspace.j3_stroke_options_m
    )

    profiles = catalog["profiles"]
    assert [profile["id"] for profile in profiles] == ["fast", "precise"]
    assert profiles[0]["label"] == "Fast"
    assert profiles[0]["settings"] == {
        "cartesian_translation_step_m": 0.10,
        "cartesian_rotation_step_deg": pytest.approx(10.0),
        "joint_interpolation_samples": 0,
        "characteristic_length_m": 0.5,
        "ik_max_iterations": 75,
    }
    assert harness.precomputer.calls == []


def test_evaluate_maps_parameters_to_one_case_and_activates_selection(spec):
    harness = _harness(spec, clock_values=(20.0, 23.25))
    parameters = _parameters(harness.backend)
    parameters.update(
        {
            "ur_x": float(parameters["ur_x"]) + 0.03,
            "ur_yaw": 12.0,
            "sr_y": float(parameters["sr_y"]) - 0.02,
            "sr_yaw": -17.0,
            "stroke": 0.45,
            "clearance": 0.012,
            "corner": "northeast",
            "mode": "simultaneous",
        }
    )

    result = harness.backend.evaluate(
        {
            "parameters": parameters,
            "profile": "fast",
            "ellipsoid_scale": 0.23,
            "show_ellipsoid": False,
        }
    )

    assert len(harness.precomputer.calls) == 1
    args, kwargs = harness.precomputer.calls[0]
    assert args[0] is spec
    grid = args[1]
    assert args[2] is harness.ur_bundle
    assert args[3] is harness.sr_bundle
    assert grid.case_count == 1
    assert tuple(grid.ur_bases.poses()) == (
        BasePose(
            parameters["ur_x"],
            parameters["ur_y"],
            parameters["ur_z"],
            parameters["ur_yaw"],
        ),
    )
    assert tuple(grid.sr_bases.poses()) == (
        BasePose(
            parameters["sr_x"],
            parameters["sr_y"],
            parameters["sr_z"],
            parameters["sr_yaw"],
        ),
    )
    assert grid.lift_heights_m == (parameters["lift"],)
    assert grid.tote_offsets_m == (parameters["tote"],)
    assert grid.corners == ("northeast",)
    assert grid.coordination_modes == (CoordinationMode.SIMULTANEOUS,)
    assert grid.skus == ("123591",)
    assert grid.sr_j3_strokes_m == (0.45,)
    assert grid.clearances_m == (0.012,)
    assert grid.tote_present is True
    assert kwargs == {
        "options": harness.profiles[0].options,
        "approach_distance_m": 0.18,
        "keepout_margin_m": 0.025,
    }

    assert len(harness.backend_factory.calls) == 1
    factory_args, factory_kwargs = harness.backend_factory.calls[0]
    assert factory_args[0] is spec
    assert factory_args[1] is harness.precomputer.caches[0]
    assert factory_args[2] is harness.viewer
    assert factory_kwargs == {"ellipsoid_scale": 0.23}
    playback = harness.backend_factory.backends[0]
    assert playback.select_calls == [
        {
            "case_id": "case-1",
            "ellipsoid_scale": 0.23,
            "show_ellipsoid": False,
        }
    ]
    assert result["evaluation_id"] == 1
    assert result["profile"] == "fast"
    assert result["elapsed_seconds"] == pytest.approx(3.25)
    assert result["case"]["id"] == "case-1"
    assert result["selection"]["case_id"] == "case-1"
    assert result["persisted"] is False


def test_select_reuses_latest_evaluation_without_precomputing_again(spec):
    harness = _harness(spec)
    evaluated = harness.backend.evaluate(
        {"parameters": _parameters(harness.backend)}
    )
    request = {
        "case_id": evaluated["case"]["id"],
        "step_index": 8,
        "check_index": 1,
        "sample_index": 2,
        "ellipsoid_scale": 0.19,
        "show_ellipsoid": True,
    }

    selected = harness.backend.select(request)

    assert len(harness.precomputer.calls) == 1
    playback = harness.backend_factory.backends[0]
    assert playback.select_calls[-1] == request
    assert len(playback.select_calls) == 2
    assert selected["request"] == request


def test_select_rejects_requests_before_evaluation_and_for_stale_case(spec):
    harness = _harness(
        spec,
        case_ids=("case-old", "case-latest"),
        clock_values=(0.0, 1.0, 2.0, 4.0),
    )

    with pytest.raises(ValueError, match="before selecting"):
        harness.backend.select({"case_id": "case-old"})

    first = harness.backend.evaluate(
        {"parameters": _parameters(harness.backend)}
    )
    second_parameters = _parameters(harness.backend)
    second_parameters["ur_x"] = float(second_parameters["ur_x"]) + 0.01
    second = harness.backend.evaluate({"parameters": second_parameters})

    assert first["case"]["id"] == "case-old"
    assert first["evaluation_id"] == 1
    assert second["case"]["id"] == "case-latest"
    assert second["evaluation_id"] == 2
    with pytest.raises(ValueError, match="latest live evaluation"):
        harness.backend.select({"case_id": "case-old"})
    assert len(harness.precomputer.calls) == 2
    assert len(harness.backend_factory.backends[-1].select_calls) == 1


@pytest.mark.parametrize(
    ("remove", "updates", "match"),
    (
        ("ur_x", {}, "missing live parameters"),
        (None, {"unexpected": 1.0}, "unknown live parameters"),
        (None, {"lift": float("nan")}, "lift must be finite"),
        (None, {"ur_x": True}, "ur_x must be numeric"),
        (None, {"ur_z": 0.0}, "robot base z must be positive"),
        (None, {"ur_z": -0.01}, "ur_base z must be at or above the floor"),
        (None, {"sr_z": -0.01}, "sr_base z must be at or above the floor"),
        (None, {"sku": "not-a-sku"}, "unknown box SKU"),
        (None, {"corner": "center"}, "unknown pallet corner"),
        (None, {"mode": "parallel"}, "unknown coordination mode"),
        (None, {"clearance": -0.001}, "clearance must be non-negative"),
    ),
)
def test_evaluate_rejects_invalid_parameters(spec, remove, updates, match):
    harness = _harness(spec)
    parameters = _parameters(harness.backend)
    if remove is not None:
        parameters.pop(remove)
    parameters.update(updates)

    with pytest.raises(ValueError, match=match):
        harness.backend.evaluate({"parameters": parameters})

    assert harness.precomputer.calls == []
    assert harness.backend_factory.calls == []


def test_evaluate_requires_parameters_object_and_known_profile(spec):
    harness = _harness(spec)

    with pytest.raises(ValueError, match="parameters must be an object"):
        harness.backend.evaluate({})
    with pytest.raises(ValueError, match="unknown evaluation profile"):
        harness.backend.evaluate(
            {
                "parameters": _parameters(harness.backend),
                "profile": "missing",
            }
        )

    assert harness.precomputer.calls == []
