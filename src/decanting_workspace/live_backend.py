"""On-demand candidate evaluation backend for the live MeshCat UI.

The cached playback mode remains immutable.  This module evaluates exactly one
parameter set at a time, converts the in-memory result to the existing replay
schema, and then reuses the tested step/check/sample presenter.  Robot bundles
are supplied once at startup and all calls are serialized by the HTTP server.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Callable, Mapping

from .evaluation import EvaluationOptions
from .models import BasePose, SceneSpec
from .playback_backend import CachedPlaybackBackend, make_meshcat_backend
from .precompute import BasePoseGrid, CaseGrid, CaseKey, PrecomputedCache, precompute_cases
from .robots import RobotBundle
from .scene import CORNER_SIGNS, tote_long_axis_offset_range_m
from .viewer import CellViewer
from .workflow import CoordinationMode


@dataclass(frozen=True)
class EvaluationProfile:
    """Named sampling policy exposed by the live UI."""

    profile_id: str
    label: str
    description: str
    options: EvaluationOptions

    def __post_init__(self) -> None:
        if not self.profile_id or not self.profile_id.replace("_", "").isalnum():
            raise ValueError("profile_id must contain only letters, digits, or underscores")
        if not self.label:
            raise ValueError("profile label must not be empty")


class LiveEvaluationBackend:
    """Evaluate arbitrary UI parameters and retain only the latest result.

    A live evaluation is intentionally not persisted as a precomputed cache.
    The single result is converted to an in-memory one-case cache solely to
    reuse the exact replay hierarchy and MeshCat presentation code.
    """

    def __init__(
        self,
        spec: SceneSpec,
        initial_key: CaseKey,
        ur_bundle: RobotBundle,
        sr_bundle: RobotBundle,
        viewer: CellViewer,
        *,
        profiles: tuple[EvaluationProfile, ...],
        default_profile: str,
        approach_distance_m: float = 0.15,
        keepout_margin_m: float = 0.0,
        precomputer: Callable[..., PrecomputedCache] = precompute_cases,
        backend_factory: Callable[..., CachedPlaybackBackend] = make_meshcat_backend,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.spec = spec
        self.initial_key = initial_key
        self.ur_bundle = ur_bundle
        self.sr_bundle = sr_bundle
        self.viewer = viewer
        self.profiles = _profile_mapping(profiles)
        if default_profile not in self.profiles:
            raise ValueError(f"unknown default live profile: {default_profile}")
        self.default_profile = default_profile
        self.approach_distance_m = _positive_finite(
            approach_distance_m,
            "approach_distance_m",
        )
        self.keepout_margin_m = _nonnegative_finite(
            keepout_margin_m,
            "keepout_margin_m",
        )
        self.precomputer = precomputer
        self.backend_factory = backend_factory
        self.clock = clock
        self._active_backend: CachedPlaybackBackend | None = None
        self._active_case_id: str | None = None
        self._evaluation_id = 0

    def catalog(self) -> Mapping[str, object]:
        """Return continuous input bounds and the initial parameter values."""

        return {
            "schema_version": 1,
            "mode": "live",
            "initial_parameters": _parameter_payload(self.initial_key),
            "parameter_schema": _parameter_schema(self.spec, self.initial_key),
            "profiles": [
                {
                    "id": profile.profile_id,
                    "label": profile.label,
                    "description": profile.description,
                    "settings": _profile_settings(profile.options),
                }
                for profile in self.profiles.values()
            ],
            "default_profile": self.default_profile,
            "result_persistence": "memory_only",
            "evaluation_serialized": True,
        }

    def evaluate(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Evaluate one complete parameter set and display its default sample."""

        parameters = _required_mapping(request.get("parameters"), "parameters")
        key = _case_key_from_parameters(self.spec, parameters)
        profile_id = request.get("profile", self.default_profile)
        if not isinstance(profile_id, str) or profile_id not in self.profiles:
            raise ValueError(f"unknown evaluation profile: {profile_id!r}")
        ellipsoid_scale = _optional_positive_float(
            request.get("ellipsoid_scale"),
            "ellipsoid_scale",
        )
        show_ellipsoid = _boolean(
            request.get("show_ellipsoid", True),
            "show_ellipsoid",
        )
        ellipsoid_wireframe = _boolean(
            request.get("ellipsoid_wireframe", False),
            "ellipsoid_wireframe",
        )

        profile = self.profiles[profile_id]
        started = self.clock()
        cache = self.precomputer(
            self.spec,
            _single_case_grid(key),
            self.ur_bundle,
            self.sr_bundle,
            options=profile.options,
            approach_distance_m=self.approach_distance_m,
            keepout_margin_m=self.keepout_margin_m,
        )
        if len(cache.cases) != 1:
            raise RuntimeError("live evaluation must produce exactly one case")
        playback_backend = self.backend_factory(
            self.spec,
            cache,
            self.viewer,
            ellipsoid_scale=(0.15 if ellipsoid_scale is None else ellipsoid_scale),
        )
        catalog = playback_backend.catalog()
        cases = catalog.get("cases")
        if not isinstance(cases, list) or len(cases) != 1:
            raise RuntimeError("live playback catalog must contain exactly one case")
        case = cases[0]
        if not isinstance(case, Mapping):
            raise RuntimeError("live playback case must be a mapping")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise RuntimeError("live playback case has no stable identifier")
        selection = playback_backend.select(
            {
                "case_id": case_id,
                "ellipsoid_scale": ellipsoid_scale,
                "show_ellipsoid": show_ellipsoid,
                "ellipsoid_wireframe": ellipsoid_wireframe,
            }
        )
        elapsed = self.clock() - started
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise RuntimeError("live evaluation clock returned an invalid duration")

        self._active_backend = playback_backend
        self._active_case_id = case_id
        self._evaluation_id += 1
        return {
            "evaluation_id": self._evaluation_id,
            "profile": profile_id,
            "elapsed_seconds": elapsed,
            "case": dict(case),
            "selection": dict(selection),
            "persisted": False,
        }

    def select(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Replay a nested pose from the latest live result without recomputing."""

        if self._active_backend is None or self._active_case_id is None:
            raise ValueError("run a live evaluation before selecting a task pose")
        case_id = request.get("case_id")
        if case_id != self._active_case_id:
            raise ValueError("case_id is not the latest live evaluation")
        return self._active_backend.select(request)


def _profile_mapping(
    profiles: tuple[EvaluationProfile, ...],
) -> dict[str, EvaluationProfile]:
    if not profiles:
        raise ValueError("at least one live evaluation profile is required")
    result: dict[str, EvaluationProfile] = {}
    for profile in profiles:
        if profile.profile_id in result:
            raise ValueError(f"duplicate live evaluation profile: {profile.profile_id}")
        result[profile.profile_id] = profile
    return result


def _single_case_grid(key: CaseKey) -> CaseGrid:
    return CaseGrid(
        ur_bases=BasePoseGrid.single(key.ur_base),
        sr_bases=BasePoseGrid.single(key.sr_base),
        lift_heights_m=(key.lift_height_m,),
        tote_offsets_m=(key.tote_offset_m,),
        corners=(key.corner,),
        coordination_modes=(key.coordination_mode,),
        skus=(key.sku,),
        sr_j3_strokes_m=(key.sr_j3_stroke_m,),
        clearances_m=(key.clearance_m,),
        tote_present=True,
    )


def _case_key_from_parameters(
    spec: SceneSpec,
    parameters: Mapping[str, object],
) -> CaseKey:
    required = {
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
    missing = sorted(required - set(parameters))
    unknown = sorted(set(parameters) - required)
    if missing:
        raise ValueError(f"missing live parameters: {missing}")
    if unknown:
        raise ValueError(f"unknown live parameters: {unknown}")

    ur_base = BasePose.from_sequence(
        tuple(
            _finite_number(parameters[name], name)
            for name in ("ur_x", "ur_y", "ur_z", "ur_yaw")
        ),
        "ur_base",
    )
    sr_base = BasePose.from_sequence(
        tuple(
            _finite_number(parameters[name], name)
            for name in ("sr_x", "sr_y", "sr_z", "sr_yaw")
        ),
        "sr_base",
    )
    if ur_base.z_m <= 0.0 or sr_base.z_m <= 0.0:
        raise ValueError("robot base z must be positive")
    sku = parameters["sku"]
    corner = parameters["corner"]
    mode = parameters["mode"]
    if not isinstance(sku, str) or sku not in spec.box_skus:
        raise ValueError(f"unknown box SKU: {sku!r}")
    if not isinstance(corner, str) or corner not in CORNER_SIGNS:
        raise ValueError(f"unknown pallet corner: {corner!r}")
    try:
        coordination_mode = CoordinationMode(str(mode))
    except ValueError as exc:
        raise ValueError(f"unknown coordination mode: {mode!r}") from exc
    clearance_raw = parameters["clearance"]
    clearance = (
        None
        if clearance_raw is None
        else _nonnegative_finite(clearance_raw, "clearance")
    )
    return CaseKey(
        ur_base=ur_base,
        sr_base=sr_base,
        lift_height_m=_finite_number(parameters["lift"], "lift"),
        tote_offset_m=_finite_number(parameters["tote"], "tote"),
        corner=corner,
        coordination_mode=coordination_mode,
        sku=sku,
        sr_j3_stroke_m=_positive_finite(parameters["stroke"], "stroke"),
        clearance_m=clearance,
    )


def _parameter_payload(key: CaseKey) -> dict[str, object]:
    return {
        "ur_x": key.ur_base.x_m,
        "ur_y": key.ur_base.y_m,
        "ur_z": key.ur_base.z_m,
        "ur_yaw": key.ur_base.yaw_deg,
        "sr_x": key.sr_base.x_m,
        "sr_y": key.sr_base.y_m,
        "sr_z": key.sr_base.z_m,
        "sr_yaw": key.sr_base.yaw_deg,
        "sku": key.sku,
        "lift": key.lift_height_m,
        "tote": key.tote_offset_m,
        "stroke": key.sr_j3_stroke_m,
        "clearance": key.clearance_m,
        "corner": key.corner,
        "mode": key.coordination_mode.value,
    }


def _parameter_schema(spec: SceneSpec, initial: CaseKey) -> dict[str, object]:
    region = spec.installation_region
    tote_low, tote_high = tote_long_axis_offset_range_m(spec)
    max_base_z = max(1.5, initial.ur_base.z_m + 0.5, initial.sr_base.z_m + 0.5)

    def number(
        label: str,
        unit: str,
        minimum: float,
        maximum: float,
        step: float,
        *,
        nullable: bool = False,
    ) -> dict[str, object]:
        return {
            "kind": "number",
            "label": label,
            "unit": unit,
            "min": minimum,
            "max": maximum,
            "step": step,
            "nullable": nullable,
        }

    def choice(label: str, options: list[dict[str, object]]) -> dict[str, object]:
        return {"kind": "choice", "label": label, "options": options}

    schema = {
        "ur_x": number("UR20 X", "m", region.raw_xy_min_m[0], region.raw_xy_max_m[0], 0.01),
        "ur_y": number("UR20 Y", "m", region.raw_xy_min_m[1], region.raw_xy_max_m[1], 0.01),
        "ur_z": number("UR20 Z", "m", 0.01, max_base_z, 0.01),
        "ur_yaw": number("UR20 yaw", "°", -180.0, 180.0, 1.0),
        "sr_x": number("SR-12iA X", "m", region.raw_xy_min_m[0], region.raw_xy_max_m[0], 0.01),
        "sr_y": number("SR-12iA Y", "m", region.raw_xy_min_m[1], region.raw_xy_max_m[1], 0.01),
        "sr_z": number("SR-12iA Z", "m", 0.01, max_base_z, 0.01),
        "sr_yaw": number("SR-12iA yaw", "°", -180.0, 180.0, 1.0),
        "lift": number(
            "Lift height",
            "m",
            spec.pallet.lift_range_m[0],
            spec.pallet.lift_range_m[1],
            0.01,
        ),
        "tote": number("Tote offset", "m", tote_low, tote_high, 0.01),
        "clearance": number("Clearance", "m", 0.0, 0.20, 0.001, nullable=True),
        "sku": choice(
            "SKU",
            [
                {"value": sku, "label": f"{sku} · {item.label}"}
                for sku, item in spec.box_skus.items()
            ],
        ),
        "corner": choice(
            "Pallet corner",
            [{"value": value, "label": value} for value in CORNER_SIGNS],
        ),
        "mode": choice(
            "Coordination",
            [
                {"value": mode.value, "label": mode.value}
                for mode in (CoordinationMode.SEQUENTIAL, CoordinationMode.SIMULTANEOUS)
            ],
        ),
    }
    workspace = spec.robots["sr12ia"].workspace
    if workspace is None:
        raise ValueError("SR-12iA workspace specification is missing")
    schema["stroke"] = choice(
        "SR J3 stroke",
        [
            {"value": value, "label": f"{value * 1000.0:g} mm"}
            for value in workspace.j3_stroke_options_m
        ],
    )
    return schema


def _profile_settings(options: EvaluationOptions) -> dict[str, object]:
    return {
        "cartesian_translation_step_m": options.cartesian_translation_step_m,
        "cartesian_rotation_step_deg": math.degrees(options.cartesian_rotation_step_rad),
        "joint_interpolation_samples": options.joint_interpolation_samples,
        "characteristic_length_m": options.characteristic_length_m,
        "ik_max_iterations": options.ik.max_iterations,
    }


def _required_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_finite(value: object, name: str) -> float:
    result = _finite_number(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_finite(value: object, name: str) -> float:
    result = _finite_number(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _optional_positive_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _positive_finite(value, name)


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value
