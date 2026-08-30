# Robot assets

- `ur20/ur20_primitive.urdf` is a deterministic mesh-free fallback with the nominal six-axis chain and local visual/collision primitives.
- `ur20/official/` is the default, offline-capable Universal Robots 4.3.1 bundle. Its URDF uses only bundle-relative mesh paths; `PROVENANCE.json` pins every file hash and the upstream commit. Both the source license and Universal Robots Graphical Documentation terms are included.
- `sr12ia/sr12ia_approx.urdf` is the corrected 2R-P-R fallback. Its kinematics follow the published `0.450 + 0.450 m` reach, `0.336 m` arm-plane height, downward J3 stroke, and zero-offset J4/tool0. Its compound shapes remain conservative proxies, not fabrication CAD.

FANUC manufacturer geometry is deliberately not vendored. Run
`decanting-assets PATH_TO_GEOMETRIES_USD`, or review the linked agreement and
run `decanting-assets --download --accept-fanuc-license`. Both paths generate a
relocatable local bundle under `.cache/robot_assets/sr12ia`; the pinned source
URL and digest are recorded in `sr12ia/download_manifest.json`. An explicit
URDF can also be selected with `--ur20-urdf` or `--sr12ia-urdf`.
