# Pinocchio-based decanting workspace

Python-only scene and robot-model foundation for the decanting-cell base-placement study. The scene is independent of Isaac Sim and ROS/MoveIt at runtime, so the same validated `SceneSnapshot` can later feed Pinocchio/Coal collision-aware IK, UR20 Jacobian metrics, and the candidate-iteration layer.

Implemented in this stage:

- fixed conveyor, worktable, and camera-support proxies extracted from `cu_usd_simplified.usd`;
- a synthesized `World Z = 0` floor;
- the `1.350 x 1.000 x 0.150 m` pallet and `0...0.760 m` lift range;
- all five real box sizes and the four alternative lowest-layer corner samples;
- one representative `0.660 x 0.440 x 0.300 m` tote;
- independently movable UR20 and SR-12iA bases, including mounting height and pedestal;
- the official mesh-based UR20 description as the visualization default;
- a local FANUC/Isaac SR-12iA mesh preparation path and a corrected conservative primitive fallback;
- a provisional UR20 suction proxy and SR cutting-tool proxy;
- an SR-12iA cutter-TCP task footprint tied to the support-cube top.

## Environment

Pixi is not used. Create or update the single Conda prefix environment and install the package in editable mode:

```powershell
conda env create --prefix .\.venv --file environment.yml
conda activate .\.venv
python -m pip install -e . --no-deps --no-build-isolation
```

On this computer, Miniconda is installed at `C:\Users\wogns\miniconda3`, but the PowerShell session used by Codex does not have the Conda shell hook on `PATH`. In addition, Conda reads the Korean workspace path correctly only in UTF-8 mode. The activation-free equivalent is therefore:

```powershell
$env:PYTHONUTF8 = "1"
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run --prefix .\.venv `
  python -m pytest
```

Running `.\.venv\python.exe` directly does not activate `.venv\Library\bin`; dynamically loaded NumPy/BLAS DLLs can then fail. Use `conda activate` or `conda run` for tests and visualization.

## Robot assets

### UR20

`--ur20-source official` is the default. It resolves the Universal Robots ROS 2 description release 4.3.1 at immutable commit `ae333289875f9ba5a9ea6649a54036efb5ccabee`, including DAE visual meshes and STL collision meshes. The checkout and generated URDF are cached by `robot_descriptions`; the repository does not copy the graphical files.

For a deterministic mesh-free fallback, use `--ur20-source primitive`. An explicit model always takes precedence:

```powershell
python -m decanting_workspace --ur20-urdf C:\models\ur20.urdf --open
```

The suction tool dimensions are provisional and live in `config/cell_nominal.yaml`. With the official UR20 they are attached to `tool0` both as a viewer primitive and as two collision geometries. Loading with `suction_proxy=...` also creates a `suction_tcp` operational frame at the configured contact plane; use that frame, not the bare flange `tool0`, for IK and Jacobians.

### FANUC SR-12iA

FANUC provides product outline CAD to registered MyFANUC users. Isaac Sim 6.0 also contains a link-separated FANUC SR-12iA asset at `Isaac/Robots/Fanuc/sr12ia/sr12ia.usd`. The local converter consumes that asset's `payloads/geometries.usd`; it never downloads a file:

```powershell
python -m decanting_workspace.asset_prep `
  C:\path\to\Fanuc\sr12ia\payloads\geometries.usd
```

It writes eight OBJ files, a four-axis URDF, and a provenance manifest to the ignored directory `.cache/robot_assets/sr12ia`. The supplied Isaac asset is the standard 300 mm J3 model, so its generated URDF is limited to `0...0.300 m`. The viewer's default `--sr12ia-source auto` uses it only for the 300 mm scenario. A 450 mm scenario automatically uses the corrected provisional fallback; `--sr12ia-source mesh` rejects that mismatch. Use `--sr12ia-source approximate` to force the fallback, or `--sr12ia-urdf PATH` for a matching manufacturer model.

The fallback follows the real 2R-P-R skeleton: `L1 = L2 = 0.450 m`, the arm plane is `0.336 m` above the base, J3 moves downward through `0...0.450 m`, and J4/tool0 has no artificial translational offset. At zero joint position, `tool0 = [0.900, 0, 0.336] m`. Its shaft uses the longer 450 mm-option height for both choices, conservatively over-approximating the 300 mm fallback until the corresponding manufacturer asset is available. A config-driven cutter collision/visual proxy and `cutter_tcp` are attached when the manufacturer arm mesh has no tool.

The default SR guide is deliberately **not** the arm's full `0.9 m` mechanical reach or a swept-link cylinder. It is the permitted XY target region for `cutter_tcp`: the oriented top footprint of the SR support cube, inset only by the selected `--clearance-mm`. Cutting Z comes from each box-top task pose and is checked later with J3 limits and collision-aware IK. The cutter body and both robots' swept links remain collision geometry, so concurrent UR20/SR motion must be checked along the actual trajectories rather than against this task footprint.

The Universal Robots graphical documentation and FANUC/Isaac assets have separate licenses. In particular, the FANUC 3D Content Sharing Agreement restricts standalone redistribution, so derived SR mesh files stay under `.cache` and are not committed. Review the applicable terms before sharing generated assets.

Sources: [Universal Robots description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description), [FANUC SR-12iA specifications](https://www.fanucamerica.com/products/robot/sr-12ia), [FANUC CAD download information](https://www.fanuc.co.jp/ja/product/outlinedata.html), [Isaac Sim robot asset catalog](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/assets/usd_assets_robots.html), and [FANUC 3D Content Sharing Agreement](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/_downloads/8ea27bd6ad258f4242d675e67e8f7b58/3D%20Content%20Sharing%20Agreement_FANUC.pdf).

## Visualize a scenario

Open the live view:

```powershell
python -m decanting_workspace --open
```

Or save a self-contained review snapshot:

```powershell
python -m decanting_workspace `
  --ur20-source primitive `
  --sr12ia-source approximate `
  --lift-height-mm 380 `
  --sku 049995 `
  --sr-j3-stroke-mm 450 `
  --save-html outputs\cell-lift-380.html
```

Self-contained Meshcat HTML embeds mesh bytes. The CLI therefore rejects
static export when the pinned UR20 or prepared FANUC asset is selected; use the
live `--open` view for those models. Explicit third-party URDF licensing remains
the caller's responsibility.

Candidate/scenario controls are:

```text
--ur-base X_M Y_M Z_M YAW_DEG
--sr-base X_M Y_M Z_M YAW_DEG
--lift-height-mm 0..760
--sku 123591|430060|180043|049995|000509
--corner all|southwest|southeast|northwest|northeast
--sr-j3-stroke-mm 300|450
--clearance-mm VALUE
```

`--corner all` shows four translucent alternative samples; it does not create four simultaneous collision boxes. Selecting one corner creates one physical box scenario.

## Computation/visualization split

Visual meshes must not be loaded inside a base-optimization iteration. Resolve and load each robot once before the candidate loop. For UR20, `load_visual=False` keeps the kinematic model and collision STL geometry while omitting the DAE viewer model:

```python
from decanting_workspace import load_scene_spec
from decanting_workspace.robots import load_robot_bundle, resolve_official_ur20

spec = load_scene_spec()
urdf, package_dirs = resolve_official_ur20()
ur20 = load_robot_bundle(
    "ur20",
    urdf,
    package_dirs=package_dirs,
    load_visual=False,
    suction_proxy=spec.robots["ur20"].suction_proxy,
)

for candidate in base_candidates:
    # Reuse ur20.model and ur20.collision_model here.
    ...
```

If an SR bundle is used directly for collision checks, call
`configure_sr_j3_stroke(bundle, state.sr_j3_stroke_m)` when the scenario changes.
It restores/restricts the limit against the model's native capability, so a
300 mm scenario cannot accidentally evaluate J3 positions out to 450 mm.

The final manipulability definition, collision-aware IK, trajectory checking, simultaneous/sequential timing constraints, and objective aggregation remain the next layer. `config/cell_nominal.yaml` is the authoritative dimension source; USD is retained only as geometry/frame provenance.
