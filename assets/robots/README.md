# Robot assets

- `ur20/ur20_primitive.urdf` is a deterministic mesh-free fallback with the nominal six-axis chain and local visual/collision primitives. The default viewer instead resolves the pinned official Universal Robots mesh description.
- `sr12ia/sr12ia_approx.urdf` is the corrected 2R-P-R fallback. Its kinematics follow the published `0.450 + 0.450 m` reach, `0.336 m` arm-plane height, downward J3 stroke, and zero-offset J4/tool0. Its compound shapes remain conservative proxies, not fabrication CAD.

Licensed manufacturer geometry is deliberately not vendored. `python -m decanting_workspace.asset_prep PATH_TO_GEOMETRIES_USD` converts a locally supplied Isaac/FANUC SR-12iA geometry layer into `.cache/robot_assets/sr12ia`. An explicit URDF can also be selected with `--ur20-urdf` or `--sr12ia-urdf`.
