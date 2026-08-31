# 개발 인수인계

마지막 정리일: 2026-08-31

이 문서는 다른 컴퓨터와 다른 개발자가 현재 작업을 이어가기 위한 실행·의사결정 기록이다. 코드 계층과 데이터 흐름은 [ARCHITECTURE.md](ARCHITECTURE.md), 사용자 명령의 전체 옵션은 [README.md](../README.md)를 먼저 함께 확인한다.

## 1. 최종 목표

한쪽 decanting cell에 UR20과 FANUC SR-12iA를 배치한다.

- UR20은 박스와 tote 운반, tote 밀기를 담당한다.
- SR-12iA는 `UncasingLoadFrame`에 놓인 박스 윗면을 절단한다.
- 두 robot base의 평면 위치, 장착 높이, yaw를 최적화한다.
- 팔레트 lift 높이, 실제 SKU 크기, tote 위치, SR J3 option, 동시/순차 작업을 함께 고려한다.
- UR20은 collision-free IK 가능성과 Jacobian manipulability ellipsoid를 평가한다.
- SR-12iA는 절단 범위에 대해 보수적인 task-workspace feasibility를 평가한다.

현재 구현은 목적함수를 정하기 직전 단계다. `evaluate_base_candidate`와 precompute cache가 후보별 원자료를 제공하지만 아직 어떤 metric을 얼마나 가중할지, 어떤 탐색 알고리즘을 사용할지는 결정하지 않았다.

## 2. 사용자와 확정한 공정·환경 결정

### Cell 범위

- 한쪽 cell만 평가한다. 반대편은 나중에 배치를 복제한다.
- 팔레트, 3단 conveyor, worktable은 고정이다.
- installation 평면은 conveyor와 pallet/worktable 사이의 한 cell strip으로 제한한다.
- 두 robot 모두 base `x`, `y`, 장착 높이 `z`, yaw가 설계변수다.
- pedestal은 바닥부터 base 높이까지 채워진 보수적 box로 취급한다.
- BGF/USD 형상은 상대 배치의 근거이고, 계산 치수는 `config/cell_nominal.yaml`이 우선한다.

### 물체 치수

모든 계산 치수는 metre로 config에 저장한다. 원자료의 mm 치수는 다음과 같다.

| SKU | 물체 | 크기 X × Y × Z (mm) |
| --- | --- | --- |
| `123591` | 유앤 스웨디시스타일젤리 | 390 × 237 × 327 |
| `430060` | 썬푸드 오징어와땅콩85g | 520 × 340 × 310 |
| `180043` | 화인 반코팅장갑 | 513 × 275 × 410 |
| `049995` | 리앤웰 비말검정대형7P | 517 × 503 × 545 |
| `000509` | PBICK 일회용접시대10P | 560 × 480 × 240 |

- pallet: `1350 × 1000 × 150 mm`
- pallet lift support height: `0 ... 760 mm`
- tote: `660 × 440 × 300 mm`
- tote는 90° 회전되어 긴 축이 worktable의 짧은 축과 나란하다.
- `ToteLoadFrame`과 tote는 worktable의 긴 축을 따라 함께 이동할 수 있다.
- 팔레트 검사는 가장 낮은 층의 네 corner를 독립 대안으로 평가한다.
- USD의 임시 box 크기는 무시하고 실제 SKU 치수를 사용한다.

### Robot와 tool

- UR20 계산/표시 기본 모델은 Git에 포함된 공식 mesh bundle이다.
- suction gripper는 Isaac Sim 기본 suction을 가정한 임시 cylinder/pad 치수다.
- 모든 UR20 IK와 Jacobian target은 `suction_tcp`다.
- SR-12iA process tool은 현재 모델링하지 않는다.
- SR-12iA의 실제 mesh는 각 사용자가 라이선스 동의 후 준비하며 Git에는 넣지 않는다.
- SR 평가 fallback은 실제 2R-P-R skeleton과 300/450 mm J3 option을 보수적으로 표현한다.
- SR 작업영역은 arm 전체 reach가 아니라 support cube 상면에 설정한 박스 절단 허용영역이다.

### 작업 순서

1. 팔레트의 선택한 box를 top-down pick
2. `UncasingLoadFrame`에 top-down place, World yaw 180°
3. conveyor 2층의 빈 tote를 정면에서 잡고 수직 lift
4. 이동 가능한 `ToteLoadFrame` 위치에 tote place
5. 윗면이 열린 box를 uncasing 위치에서 정면 pick
6. tote 위에서 내용물을 붓는 순서 marker; pose 검사는 아직 없음
7. 빈 box를 conveyor 3층에 버림; dimension/pose 가능성을 기록하지만 SOFT
8. tote의 conveyor 반대쪽 면을 밀어 `ToteSupplyFrame`까지 이동하고 exact final posture를 별도 검사
9. 반복

Step 3과 4는 simultaneous mode에서 지정된 SR 절단 자세와 작업영역을 감싼 conservative exclusion box를 충돌물로 추가한다. sequential mode에는 이 동적 keepout을 추가하지 않지만 SR pedestal과 camera support는 항상 남는다.

## 3. 현재 구현 상태

다음 기능이 구현되어 있다.

- validated dimension/frame config와 immutable scene snapshot
- 두 robot의 독립적인 base pose 및 pedestal materialization
- 팔레트 lift, 다섯 SKU, 네 corner 대안, movable tote frame
- base installation rectangle와 pedestal/static overlap 검사
- official UR20 mesh collision model 및 provisional suction TCP/collision
- SR top-corner footprint/planar 2R/J3 보수 검사
- simultaneous/sequential scenario와 SR concurrent exclusion OBB
- 8단계 workflow의 phase별 object/contact state
- fixed-base deterministic multi-start Pinocchio IK
- explicit Pinocchio/Coal self/environment/payload collision pair
- local Cartesian 및 joint interpolation collision diagnostic
- translational/normalized 6-D Jacobian SVD metric
- candidate report와 finite case-grid precompute
- cached `q`, collision, target, manipulability ellipsoid의 MeshCat UI replay
- 임의의 단일 parameter set을 계산하고 최신 결과만 메모리에 보관하는
  MeshCat live evaluation UI
- schema 기반 15개 live parameter control과 case 전체 `Calculated result`
  요약, step/check/sample 세부 재생
- live control URL과 raw MeshCat viewer URL의 명시적 분리 및 Windows control
  port exclusive bind
- UR20 suction display의 `tool0` 추종
- Step 8 exact supply endpoint의 독립 posture 검사
- checkout 절대경로를 제외한 cache fingerprint
- 다른 PC에서 재사용 가능한 tracked cache와 logical/repo-relative model report 표기
- Git에 포함되는 상대경로 official UR20 bundle
- 라이선스 opt-in SR download/local conversion 및 setup/doctor 경로

### 검증된 working setup

`config/case_grid_working.yaml`의 commissioning point는 다음과 같다.

- UR20 base: `[-1.300, -7.350, 0.850, 90°]`
- SR-12iA base: `[-0.300, -7.073643684, 0.665, -90°]`
- lift: `0.200 m`
- tote offset: `+0.700 m`
- SR J3: `0.450 m`
- 다섯 SKU × 네 corner × 두 coordination mode = 40 cases

현재 checked working result의 의미는 다음과 같다.

- sequential 20/20 cases가 UR20 HARD step 1–5, 8을 통과한다.
- 전체 cell feasible은 12/40이다.
- SR workspace까지 통과한 SKU는 `123591`, `430060`, `000509`이며 각 네 corner다.
- sequential HARD pose sample은 1434개이며 IK와 manipulability가 저장되어 있다.
- simultaneous case는 SR keepout과의 충돌 진단을 위해 의도적으로 cache에 남겨 둔다.
- 최소 관찰 clearance 약 `6.464 mm`인 commissioning point이므로 production robust layout으로 간주하면 안 된다.

`outputs/precomputed_nominal_cases.json.gz`는 720개의 broader diagnostic cases를 포함한다. 이 파일은 Cartesian check의 start/end 위주 task-pose cache이며 `10000 mm`, `180°`, joint interpolation 0 설정으로 생성되었다. 따라서 continuous collision-free path 결과로 해석하면 안 된다.

## 4. 새 컴퓨터에서 시작하기

### 4.1 Clone

일반 clone이면 runtime code와 tracked robot/cache가 준비된다.

```powershell
git clone <REPOSITORY_URL>
cd Pinocchio_based_workspace_optimization
```

`BGF`는 runtime에 필요하지 않다. provenance를 확인해야 할 때만 root submodule 하나를 **비재귀적으로** 초기화한다.

```powershell
git submodule update --init BGF
```

`git clone --recurse-submodules` 또는 `git submodule update --recursive`는 사용하지 않는다. BGF 내부 submodule mapping이 완전하지 않아 불필요한 실패를 만들 수 있다.

### 4.2 Conda 환경

Pixi는 사용하지 않는다. Python 3.12 Pinocchio 환경은 Miniconda의
`pinocchio-workspace` named environment를 사용하며 repository-local Python
environment를 만들지 않는다.

```powershell
conda env create -n pinocchio-workspace --file environment.yml
conda run -n pinocchio-workspace python -m pip install -e . --no-deps --no-build-isolation
```

환경이 이미 있으면 다음과 같이 갱신한다.

```powershell
conda env update -n pinocchio-workspace --file environment.yml --prune
conda run -n pinocchio-workspace python -m pip install -e . --no-deps --no-build-isolation
```

Windows의 비ASCII workspace에서는 먼저 UTF-8 mode를 켠다.

```powershell
$env:PYTHONUTF8 = "1"
```

Windows에서는 Conda environment의 `python.exe`를 경로로 직접 실행하면
`Library\bin` 활성화가 빠져 NumPy/BLAS DLL load가 실패할 수 있다.
`conda activate pinocchio-workspace` 또는 위의
`conda run -n pinocchio-workspace`를 사용한다.

Linux/macOS shell에서도 같은 named environment를 사용한다. PowerShell의
backtick만 제거하면 된다.

```bash
conda env create -n pinocchio-workspace --file environment.yml
conda run -n pinocchio-workspace python -m pip install -e . --no-deps --no-build-isolation
conda run -n pinocchio-workspace python -m pytest
```

기본 경로 탐색은 editable checkout의 모듈 위치를 기준으로 한다. Python을
non-editable 방식으로 설치했다면 `DECANTING_WORKSPACE_ROOT`를 이 저장소의
완전한 clone root로 설정한다. 드라이브 문자, 사용자명, clone 폴더명은
코드나 tracked artifact 계약에 포함되지 않는다.

### 4.3 SR-12iA mesh 준비

SR manufacturer mesh는 license 때문에 repository에 없다. 두 경로 중 하나를 사용한다.

권한 있는 NVIDIA source를 명시적으로 내려받는 경로:

```powershell
conda run -n pinocchio-workspace decanting-setup --accept-fanuc-license
```

이미 보유한 Isaac Sim `payloads/geometries.usd`를 사용하는 경로:

```powershell
conda run -n pinocchio-workspace decanting-setup `
  --sr12ia-usd C:\path\to\Fanuc\sr12ia\payloads\geometries.usd
```

`--accept-fanuc-license`는 자동 download 전에 사용자가 FANUC 3D Content Sharing Agreement를 검토하고 동의했음을 명시한다. downloader는 pinned NVIDIA URL과 SHA-256을 검증한 뒤 변환한다. 출력은 다음 local-only bundle이다.

```text
.cache/robot_assets/sr12ia/
├── sr12ia_mesh.urdf
├── sr12ia_mesh.provenance.json
└── *_visual.obj / *_collision.obj
```

기존 bundle을 download 없이 검사하려면 option 없는 setup을 실행한다.

```powershell
conda run -n pinocchio-workspace decanting-setup
```

이 명령은 SR bundle과 함께 official UR20, config, tracked cache fingerprint 등 실행 계약을 doctor 방식으로 확인한다. SR bundle이 없는 경우 조용히 fallback하지 않고 준비 방법을 포함한 오류를 반환한다.

source 또는 converter가 바뀌어 기존 SR bundle을 강제로 다시 만들 때만 다음 option을 추가한다.

```powershell
conda run -n pinocchio-workspace decanting-setup `
  --accept-fanuc-license `
  --refresh-sr12ia
```

robot asset만 준비하고 아직 tracked cache를 검증하지 않을 특별한 경우에는 `--skip-cache-validation`을 사용할 수 있다. 정상 인수 검증에서는 이 option을 사용하지 않는다.

SR asset만 저수준으로 준비할 수도 있다.

```powershell
conda run -n pinocchio-workspace decanting-assets --download --accept-fanuc-license
conda run -n pinocchio-workspace decanting-assets C:\path\to\geometries.usd
```

`.cache/robot_assets/sr12ia`를 다른 PC에서 복사하는 방식은 권장하지 않는다. 각 PC에서 setup을 실행하면 license opt-in, source hash, converter version, relative mesh reference를 함께 검증할 수 있다.

### 4.4 검증

```powershell
conda run -n pinocchio-workspace python -m pytest
conda run -n pinocchio-workspace decanting-setup
```

2026-08-31 기준 통합 baseline은 `219 passed`, 기존 MeshCat/pyzmq deprecation
warning 1건이다. 테스트 수는 기능 추가에 따라 바뀌므로 고정된 개수보다 전체
suite가 성공하는지를 기준으로 한다.

SR licensed mesh 없이 계산 fallback만 smoke-test하려면 다음 명령을 사용할 수 있다.

```powershell
conda run -n pinocchio-workspace decanting-view `
  --ur20-source official `
  --sr12ia-source approximate `
  --open
```

## 5. 자주 사용하는 명령

### Cached MeshCat UI

verified working cache:

```powershell
conda run -n pinocchio-workspace decanting-playback --open
```

broader nominal diagnostic cache:

```powershell
conda run -n pinocchio-workspace decanting-playback `
  --cache outputs\precomputed_nominal_cases.json.gz `
  --open
```

UI에서 최소한 step 1, 3, 5, 8을 선택해 robot 자세를 확인한다. UR20 suction이 각 `q`를 따라 이동하는지, 성공 sample에서 ellipsoid가 표시되는지, step 8이 기본적으로 `filled_tote_final_pose_at_supply`를 보여 주는지 확인한다.

### On-demand live MeshCat UI

기존 playback은 검증된 불변 cache를 exact lookup으로 재생하며 IK를 다시 풀지
않는다. 이와 별개의 live UI는 파라미터 한 세트를 즉시 계산하고 최신 결과
하나만 메모리에 유지한다.

```powershell
conda run --no-capture-output -n pinocchio-workspace decanting-live --open
```

운영자가 열어야 하는 주소는 터미널의 `LIVE CONTROL UI (parameters +
calculated results)`이며 기본값은 `http://127.0.0.1:8766/`이다. MeshCat
`/static/` 주소는 control page 안의 3-D iframe 전용으로 parameter와 계산
결과가 없다. Conda를 활성화한 shell에서는 `decanting-live --open`을 직접
실행해도 된다.

Live panel에는 UR/SR base pose 8개 값과 SKU, lift, tote offset, clearance,
corner, coordination, J3 stroke를 합친 15개 control이 표시된다. 최초 quick
계산은 page 초기화 직후 자동 실행된다. `Calculated result`에는 evaluation
ID/profile, cell feasibility, installation validity, HARD task 결과, SR workspace,
case status가 표시되고, 별도 Task pose 영역은 최신 case의 step/check/sample을
재계산 없이 선택한다.

기본 `quick` profile은 `10 m`, `180°`, joint interior sample 0으로 끝점 중심의
반응형 진단을 한다. 연속 collision-free path 결과가 아니다. `full` profile은
기본 `50 mm`, `5°`, joint interior sample 3으로 더 촘촘한 로컬 sampling을
수행하지만 station 사이의 전역 motion planning이나 전역 경로 존재를 보장하지
않는다.

```powershell
conda run --no-capture-output -n pinocchio-workspace decanting-live `
  --default-profile full `
  --open
```

UI는 변경을 500 ms debounce하고 계산 중 들어온 여러 변경을 최신 한 건으로
합친다. HTTP server도 evaluation과 pose selection을 하나의 lock으로
직렬화한다. SR J3 limit와 collision checker가 shared concurrent evaluation에
안전하지 않으므로 이 직렬화 계약을 제거하면 안 된다. 결과는
`outputs/`에 저장하지 않으며, step/check/sample 변경은 최신 메모리 case만
재생한다.

FANUC 준비 mesh는 표시 전용 300 mm model이다. live 계산은 기본
`assets/robots/sr12ia/sr12ia_approx.urdf`의 보수적 300/450 mm 평가 proxy를
사용하므로 450 mm feasibility도 proxy 결과다. `--sr12ia-visual-urdf`는 계산
model을 바꾸지 않고, `--sr12ia-urdf`만 평가 proxy를 바꾼다. 준비 mesh가 없을
때 proxy 표시를 허용하려면 명시적으로 `--allow-sr-visual-fallback`을 쓴다.

수동 확인 시 base/lift/tote 값을 바꿔 evaluation elapsed가 갱신되는지,
계산 완료 뒤 step 1, 3, 5, 8을 바꿀 때 추가 evaluation 없이 최신 case의
자세가 표시되는지, Auto calculate를 끄면 Calculate now 전까지 대기하는지
확인한다.

#### 2026-08-31 live UI blank-result 회귀 수정

초기 구현의 숫자 control 생성 loop가 JavaScript 배열이 아닌 괄호식
`(slider, number)`을 순회했다. JavaScript comma operator 때문에 첫
`HTMLInputElement`에서 `not iterable` 예외가 발생했고, 그 결과 parameter
영역이 비어 있으며 최초 `/api/evaluate`도 호출되지 않았다. 현재는
`[slider, number]` 배열 반복으로 수정했고 다음 항목을 회귀 계약으로 둔다.

- catalog의 parameter schema는 15개다.
- control page에는 live marker와 `Calculated result` 영역이 있다.
- 숫자 range/number pair는 배열로 구성한다.
- initial evaluate, result hierarchy, elapsed/status 표시 경로가 존재한다.
- 같은 Windows control port에 두 live server를 동시에 띄울 수 없다.

코드 갱신 전 시작한 Python process는 새 source를 다시 import하지 않는다.
기존 `decanting-live` terminal을 `Ctrl+C`로 종료한 뒤 재실행해야 하며, 브라우저
에서도 `/static/` 탭이 아니라 새 control URL을 열거나 새로고침한다.

### 단일 후보 평가

```powershell
conda run -n pinocchio-workspace decanting-evaluate `
  --sku 123591 `
  --corner all `
  --coordination both `
  --lift-height-mm 200 `
  --sr-j3-stroke-mm 450 `
  --output-json outputs\candidate.json
```

### Working cache 재생성

```powershell
conda run -n pinocchio-workspace decanting-precompute `
  --grid config\case_grid_working.yaml `
  --output-json outputs\precomputed_working_cases.json.gz
```

### Nominal task-pose cache 재생성

```powershell
conda run -n pinocchio-workspace decanting-precompute `
  --grid config\case_grid_nominal.yaml `
  --output-json outputs\precomputed_nominal_cases.json.gz `
  --cartesian-step-mm 10000 `
  --cartesian-rotation-step-deg 180 `
  --joint-interpolation-samples 0
```

full path-sampled nominal cache는 훨씬 오래 걸리고 파일이 커질 수 있다. 실행 전 case count와 목적을 확인하고, 필요하면 `--only-sku` 또는 `--only-lift-height-mm` shard를 사용한다.

## 6. Tracked/generated asset 정책

| Artifact | Git | 새 PC에서의 처리 |
| --- | --- | --- |
| Python source, config, tests, docs | 포함 | clone으로 확보 |
| UR20 official URDF/meshes/licenses/provenance | 포함 | `decanting-setup`이 bundle 무결성 확인; BSD source license와 UR Graphical Documentation 조건을 모두 보존 |
| UR20 primitive fallback | 포함 | 별도 준비 없음 |
| SR approximate evaluation URDF | 포함 | 별도 준비 없음 |
| SR pinned download manifest | 포함 | URL/source SHA/license metadata를 setup이 검증 |
| `.cache/README.md` | 포함 | cache에 source-of-truth가 없다는 정책과 bootstrap 경로 설명 |
| SR manufacturer OBJ/URDF/provenance | 미포함 | license opt-in setup 또는 local USD 변환 |
| Miniconda `pinocchio-workspace`, `.pytest_cache` | 미포함 | 각 PC에서 재생성 |
| live evaluation result | 해당 없음 | 최신 단일 case만 process memory에 유지; 파일로 저장하지 않음 |
| tracked `outputs/*.json.gz` cache | 포함 | path-independent fingerprint로 그대로 검증/replay |
| checked result report | 포함 | repo-relative logical model reference 사용; 계산 변경 시 재생성 |
| BGF | submodule pointer만 포함 | runtime 불필요; 필요할 때 `git submodule update --init BGF` |

절대경로는 process-local log와 사용자가 명시한 CLI input에만 나타날 수 있다. 다음 파일에는 원래 PC의 `C:\Users\...`, home directory, clone directory를 저장하면 안 된다.

- tracked URDF mesh reference
- scene/grid config
- precomputed cache
- checked candidate report의 model identity
- provenance에서 runtime 해석에 사용되는 path

## 7. 변경 후 지켜야 할 절차

### 환경 치수 또는 frame 변경

1. `config/cell_nominal.yaml` 또는 `config/reference_frames.json` 수정
2. config/scene/workflow 관련 test 실행
3. candidate report와 두 tracked cache 재생성
4. 다른 checkout path에서 `decanting-setup`과 playback 검증

### Workflow 변경

1. `workflow.py`의 pose/object/contact 계약 수정
2. `WORKFLOW_SCHEMA_VERSION` 증가
3. workflow/evaluation/cache/UI test 수정
4. 모든 tracked cache와 report 재생성

### Robot model 또는 tool 변경

1. license/provenance와 상대경로 정책 확인
2. `suction_tcp` 또는 `tool0` kinematic contract 확인
3. robot loading, collision, IK, fingerprint test 실행
4. cache/report 재생성
5. MeshCat에서 proxy/mesh가 저장된 `q`와 함께 움직이는지 확인

### Cache serialization 변경

1. `CACHE_SCHEMA_VERSION` 증가
2. writer와 strict parser를 함께 변경
3. stale cache rejection과 round-trip test 추가
4. tracked cache 전부 재생성

## 8. 알려진 한계와 미결정 사항

다음 항목은 버그가 숨겨진 것이 아니라 후속 입력/설계가 필요한 범위다.

- UR/SR base search bounds와 grid resolution이 아직 확정되지 않았다.
- 목적함수, HARD constraint 처리 방식, SKU/corner/lift 간 aggregation과 weight가 없다.
- 최종 pose 허용오차는 확정 전이다. 현재 값은 수치 평가용 기본값이다.
- Step 6 pouring pose와 collision 검사가 없다.
- station 사이의 전역 path, 속도/가속도, cycle time은 평가하지 않는다.
- SR의 exact cutting trajectory, process tool geometry, tool collision이 없다.
- prepared FANUC mesh는 300 mm J3 model뿐이다. 450 mm case는 conservative evaluation model로 판정한다.
- suction 크기와 payload capacity, 진공 접촉 안정성은 임시다.
- tote push force, 마찰, tipping, conveyor transfer dynamics를 검사하지 않는다.
- simultaneous mode는 시간 동기화된 robot-robot path 대신 보수적인 SR exclusion OBB를 쓴다.
- conveyor/worktable/pedestal/camera는 primitive collision proxy다.
- deterministic IK `not_found`와 local interpolation 실패는 전역 infeasibility 증명이 아니다.
- live `quick`은 끝점 중심 반응형 진단이고, `full`도 더 촘촘한 로컬
  sampling일 뿐 전역 motion planning 결과가 아니다.
- live 결과는 최신 단일 case만 메모리에 남으며 재현 artifact가 필요하면
  case-grid와 `decanting-precompute`로 별도 cache를 생성해야 한다.
- working setup의 clearance는 production robustness를 보장하지 않는다.
- `environment.yml`은 호환 version 범위를 고정하지만 platform별 full lockfile은 아니다. tracked cache replay는 동일하지만 cache를 다시 계산할 때 dependency build에 따른 미세한 수치 차이가 생길 수 있다. bitwise 재현성이 필요하면 지원 OS별 Conda explicit lock을 추가해야 한다.
- 현 no-tool-offset SR/support geometry에서는 SKU `180043`, `049995`가 vertical cutting range를 만족하지 않는다. 실제 cutter offset과 mounting geometry가 정해지면 재평가해야 한다.
- 반대편 cell 복제는 아직 구현 범위가 아니다.

## 9. 다음 개발 우선순위

1. 완전히 새로운 ASCII 경로와 비ASCII 경로 clone에서 `decanting-setup`, pytest, cache load, MeshCat playback, live evaluation을 각각 검증한다.
2. 실제 UR suction gripper 치수/TCP와 SR cutting tool offset을 확정한다.
3. base의 `x/y/z/yaw` bounds와 허용 간격을 확정한다.
4. pose tolerance, clearance, collision margin을 공정 요구사항으로 확정한다.
5. HARD feasibility를 constraint로 두고 manipulability, clearance, SOFT step 7을 어떻게 목적함수에 반영할지 설계한다.
6. candidate iterator/optimizer adapter를 구현한다. 병렬화할 경우 worker별 SR `RobotBundle`과 collision checker를 사용한다.
7. 상위 후보를 더 촘촘한 path sampling과 Isaac Sim/실기에서 재검증한다.

## 10. 인수 완료 체크리스트

- [ ] 원래 개발 PC의 절대경로 없이 clean clone이 import된다.
- [ ] `git submodule update --recursive` 없이 작업할 수 있다.
- [ ] Conda 환경 생성과 editable install이 성공한다.
- [ ] vendored UR20 bundle의 URDF, mesh, license, provenance 검사가 성공한다.
- [ ] SR license 동의 download 또는 local USD 변환이 성공하고 bundle path가 상대경로다.
- [ ] `decanting-setup` doctor가 config/model/cache를 모두 통과한다.
- [ ] 전체 pytest suite가 통과한다.
- [ ] tracked working/nominal cache가 현재 fingerprint로 load된다.
- [ ] candidate report에 machine-specific 절대 model path가 없다.
- [ ] MeshCat에서 실제 UR20/SR mesh가 보인다.
- [ ] `decanting-playback`이 cache만 재생하고 `decanting-live`가 최신 단일
  case만 즉시 계산하는지 구분해 확인한다.
- [ ] live control URL에서 15개 parameter와 `Calculated result`가 보이고,
  raw MeshCat `/static/` URL과 혼동하지 않는다.
- [ ] live UI의 quick/full 의미, 500 ms latest-only 직렬 계산, 비영속 결과를
  운영자에게 설명할 수 있다.
- [ ] UR20 suction이 step pose를 따라 움직인다.
- [ ] 성공 sample의 manipulability ellipsoid가 보인다.
- [ ] step 8 exact `ToteSupplyFrame` final posture와 검사 결과가 보인다.
- [ ] 알려진 model limitation과 아직 정하지 않은 optimization 항목을 결과와 구분한다.
