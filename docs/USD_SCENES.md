# 일반 USD 환경 사용

환경을 큐브로 simplified하지 않아도 Mesh 형상을 직접 읽어 MeshCat에
표시하고 Coal 충돌 검사에 사용할 수 있다. 기존 공정 프레임, SKU, 팔레트,
토트, 로봇 URDF와 설치 영역은 `--config` 설정을 계속 사용한다.

Conda 환경에는 `usd-core`가 이미 포함되어 있다. 다른 환경에서는
`python -m pip install -e ".[usd-scene]"`로 선택 의존성을 설치한다.

## 바로 실행

저장소의 `USD/cu_usd_simplified.usd`는 전체 셀이므로 준비된 환경 프로필을
사용한다. 누락된 바닥/카메라 payload와 설정에서 생성하는 이동 물체를
제외하고, 고정 카메라 지지대와 컨베이어·작업대를 유지한다.

```powershell
decanting-live --config config\cell_usd.yaml --open
```

파일만 바꾸려면 `--usd USD\other_cell.usd`를 추가할 수 있다. 이 경우에도
프로필의 Prim 경로·논리 이름 연결은 새 파일의 구조와 일치해야 한다.
`cell_usd.yaml`은 `base_config: cell_nominal.yaml`로 기존 공정 설정·URDF를
상속하므로 해당 값을 복사해서 따로 유지할 필요가 없다. 상속한 파일 경로는
그 경로를 선언한 YAML 기준으로 해석하고, 목록은 자식 프로필 값으로 대체한다.

정적 환경만 들어 있는 USD를 지정한다.

```powershell
python -m decanting_workspace --usd "C:\scenes\environment.usd" --open
decanting-live --usd "C:\scenes\environment.usd" --open
```

`decanting-evaluate`, `decanting-precompute`, `decanting-playback`,
`decanting-setup`에도 같은 `--usd` 옵션이 있다. USD는 시작할 때 읽는다.
파일을 수정한 후에는 프로세스를 다시 시작하고 계산한다. Cache를 사용하는
경우에는 동일한 config/USD로 cache를 다시 생성한다. 실제 World 정점과
삼각형 연결이 cache fingerprint에 들어가므로 형상 변경과 객체 추가가
이전 결과를 무효화한다. USD 경로의 이동 자체는 형상 identity를 바꾸지 않는다.

`--usd` 경로는 현재 작업 폴더를 기준으로 해석한다. YAML의
`usd_scene.file` 상대 경로는 그 YAML 파일이 있는 폴더를 기준으로 한다.
`--usd`가 있으면 YAML의 file만 덮어쓰고 나머지 선택 규칙은 유지한다.

## 환경 경로와 공정 지지면 연결

로봇, 이동하는 토트/팔레트까지 들어 있는 전체 셀이라면 환경 subtree를
선택한다. 기존 scene YAML에 아래와 같은 `usd_scene` 항목을 추가한다.
경로는 사용하는 USD의 실제 Prim 경로로 바꾼다.

```yaml
usd_scene:
  file: ../assets/cell.usd
  include_prim_paths: [/World/Environment]
  exclude_prim_paths: [/World/Environment/Decoration]
  box_prims:
    conveyor_level_1: /World/Environment/Conveyor1
    conveyor_level_2: /World/Environment/Conveyor2
    conveyor_level_3: /World/Environment/Conveyor3
    worktable_current_a: /World/Environment/Worktable
```

`include_prim_paths` 기본값은 `/`이며 그 아래의 지원되는 형상을 자동으로
읽는다. 해당 경로 아래 새 Mesh를 추가해도 개별 목록에 등록할 필요가 없다.
`exclude_prim_paths`는 해당 Prim과 모든 자손을 제외한다. ArticulationRootAPI가
있는 로봇 subtree도 기본적으로 제외한다. API가 없는 로봇이나 설정에서
생성하는 팔레트, 토트, 받침대, 로봇 모듈 카메라가 USD에 들어 있다면 경로로
직접 제외한다. 이 객체들은 기존 공정 설정으로 생성되므로 중복해서 가져오면
환경 장애물로 간주된다.

`box_prims`는 기존 `static_boxes`의 논리 이름을 USD subtree에 연결한다.
예를 들어 컨베이어 프레임과 본체가 여러 Mesh로 구성되어 있어도 하나의
`conveyor_level_2`로 묶을 수 있다. 위치, yaw와 경계 치수는 그 subtree의
변환된 실제 정점에서 갱신한다. 공정 목표와 접촉 예외는 이 논리 이름을
사용하고, 표시와 충돌은 실제 삼각형을 사용한다. 연결한 worktable이 토트의
`motion_support_box`이면 `support_surface_z_m`도 그 상단 높이로 갱신한다.

토트가 접촉하는 작업대와 컨베이어에는 이 연결을 설정해야 기존 지지면
접촉 예외가 적용된다. 연결하지 않은 Mesh의 이름은 `usd:/Prim/Path`가 된다.
`box_prims`를 생략하면 기존 공정 지지면 치수는 config 값을 유지한다.
USD 모드에서 기존 `static_boxes`는 공정 계산용 기준으로 남으며 환경
충돌·표시에는 중복 등록하지 않는다.

작업 프레임은 자동으로 이동하지 않는다. 작업대를 옮겨 공정 위치까지
바뀌면 `reference_frames.json`, 토트 위치/방향과 설치 영역을 함께 조정해야
한다. 일반 장애물은 기울어져 있어도 되지만 공정 지지면으로 연결하는
subtree는 수평이어야 한다. 기존 World +X 토트 접근/밀기와 축 정렬 팔레트
공정 규칙은 유지된다.
토트의 작업대 장축 offset은 World +X 쪽을 양수로 삼고, 장축이 World Y와
평행하면 +Y 쪽을 양수로 삼는다. 같은 작업대를 180도 회전하거나 로컬
X/Y 치수 표현을 바꿔도 이동 방향과 허용 범위는 유지된다.

## 지원 범위

- Mesh의 triangles, quads와 단순한 concave n-gon을 삼각형으로 읽는다.
  `holeIndices`로 제외된 face와 handedness를 반영한다.
- 부모/자식의 translate, rotate, orient, scale, matrix 및 resetXformStack을
  합성한다. 비균일 scale과 일반 장애물의 roll/pitch도 정점에 반영한다.
- `metersPerUnit`을 미터로 변환하고 Y-up 환경을 Z-up으로 회전한다.
  기존 공정 설정은 변환 후의 World 좌표와 일치해야 한다.
- 상대 references/payloads, native instances와 PointInstancer를 읽는다.
  꺼진 PointInstancer instance는 제외한다. 중첩 PointInstancer는 Mesh/native
  instances로 펼쳐야 한다.
- Cube, Sphere, Cylinder, Cone, Capsule, Plane도 삼각형으로 변환한다. 곡면 primitive는
  원주 32분할, 구/반구 위도 16/8분할의 근사 표면이다.
- 기본적으로 보이는 `default`/`render` purpose 형상을 읽는다. 숨겨진 충돌
  형상은 `include_invisible: true`, proxy 형상은 `purposes: [default, proxy]`
  등으로 명시적으로 선택할 수 있다. CollisionAPI가 없는 Mesh도 선택한
  환경 형상이면 장애물로 등록한다.
- 기본 시간의 정적 snapshot을 읽는다. 애니메이션의 특정 시점은
  `time_code: 10`처럼 선택한다. 실시간 USD animation/physics는 실행하지 않는다.

Mesh는 작성된 polygon 표면을 사용한다. subdivision refinement,
재질/텍스처 및 USD 조명은 재현하지 않는다. skinning/blend-shape Mesh는
변형을 정점에 bake한 뒤 가져와야 하며, 변형 정보를 발견하면 오류를 낸다. Points/곡선 등
지원하지 않는 형상은 명확한 오류를 내며 Mesh로 변환해서 사용해야 한다.
Coal 삼각 표면 검사에 닫힌 구성요소의 내부 포함 검사를 추가한다. 닫힌
구성요소의 내부는 점유한 장애물로 해석하고, 중첩된 shell의 cavity는
유지한다. 열린 구성요소는 표면으로 해석한다. 로봇·운반 물체와 받침대
배치에 이 규칙을 적용하며 원본 환경을 큐브 경계로 단순화하지 않는다.

제외하거나 선택하지 않은 payload는 처음부터 로드하지 않는다.
선택한 subtree의 reference/payload가 없거나 topology가 잘못되어 있으면
가져오기를 실패시킨다. 의도적으로 불필요한 누락 asset은 해당 subtree를
제외한다. Reference가 있는 USD는 관련 asset 파일도 함께 옮겨야 한다.
Windows의 비ASCII 경로는 사용 가능한 8.3 경로로 열며, 단일 layer만 임시
복사해서 상대 reference를 잃게 하는 우회는 사용하지 않는다.
