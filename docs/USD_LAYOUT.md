# USD 기반 단순화 환경

현재 레이아웃을 읽고 위치를 조정하는 모드다. 최적화 목적함수나 공정 SKU
평가는 실행하지 않는다. 기존 공정 평가 모드는 `decanting-live`에 남아 있다.

저장소 루트의 `pinocchio-workspace` 환경에서 실행한다.

```powershell
python -m decanting_workspace.layout_cli --config config\cell_layout.yaml --open
```

패키지를 설치한 환경에서는 `decanting-layout`으로도 실행할 수 있다.
다른 파일을 쓰려면 `--usd "USD\other_scene.usd"`를 추가한다.

## USD에서 읽는 정보

- ConveyorGroup, UncaseGroup, LoadGroup, PalletGroup, ToteGroup, GroundPlane의
  계층, 위치 및 형상을 읽는다. 그룹과 주요 하위 요소가 위치 조정 단위다.
- 컨베이어는 벨트 또는 롤러의 운반 영역을 단순화한다. 결합된 상세 본체의
  기둥·다리·프레임 전체를 장애물로 만들지 않는다. 기존 숨겨진 Conv1 등의
  큐브를 실제 운반 면과 중복 생성하지 않는다.
- 팔레트, 지지대, 벽과 바닥은 USD 형상에서 단순화한다.
- USD의 토트와 공급 토트는 환경 형상으로 생성하지 않는다. 실제 visual 형상에서
  읽은 크기만 `reference_dimensions`에 metre 단위로 보관하며 원본 Prim 경로를 남긴다.
  토트는 위치 조정, 겹침 제거 및 로봇 충돌 환경에 포함되지 않는다.
- USD의 제품 예시 및 FlatBox SKU는 제외한다. 기존 YAML의 SKU를 이 환경에
  생성하지도 않는다. 새로운 작업 물체는 이후 공정 평가 입력으로 별도 정의한다.
- 로봇의 **base_link 월드 XYZ**를 읽고, 기존 URDF의 회전·joint 설정으로
  Pinocchio 모델을 배치한다. 그룹 원점이 로봇 장착점이라고 가정하지 않는다.
  USD 로봇 Mesh는 환경 장애물로 가져오지 않는다.
- 재질, 텍스처와 USD physics 대신 단순화된 형상과 URDF를 표시한다.

## 위치 조정과 겹침

3D 화면은 왼쪽 드래그로 회전, 가운데/오른쪽 드래그로 이동, 휠로 확대·축소한다.
`정면`, `측면`, `상단`, `처음 보기` 버튼으로 시점을 선택할 수 있으며,
노드 위치를 수정해도 현재 카메라 시점은 유지된다. 토트의 참고 치수는 별도로 표시한다.
초기 시점과 회전 중심은 바닥·방 벽을 제외한 작업 셀을 기준으로 맞춘다.
카메라 위치, 회전 중심과 up 축은 World Z-up 좌표로 통일하며, 휠을 올리면
회전 중심을 향해 확대하고 내리면 축소한다.

UI에서 노드를 선택하고 XYZ offset을 지정한다. offset은 부모의 좌표축을
기준으로 하며, 최상위 그룹의 부모 좌표축은 World다. 회전은 변경하지 않는다.
그룹을 이동하면 자식 형상·로봇·작업 프레임이 같이 이동한다. 자식의 offset은
부모 이동에 더해지며 중복 적용되지 않는다.

단순화한 형상끼리 겹치는 부피는 먼저 생성된 요소 한 곳에 배정하고, 뒤의
요소에서 해당 부분을 잘라낸다. 기울어진 형상도 convex clipping으로 처리한다.
공유 경계면의 접촉은 허용한다. 원래의 단순화 형상은 유지하므로 위치를
수정할 때마다 겹침을 다시 계산하며, 떨어뜨린 요소는 전체 형상을 되찾는다.

겹침 처리는 정적 환경 형상에 적용한다. URDF 로봇의 링크는 독립된 Pinocchio
모델이며 환경 형상으로 잘라내지 않는다. 초기 배치를 충돌 검사로 거부하지 않는다.

현재 환경을 JSON으로 저장하려면 `--output outputs\layout.json`을 추가한다.
저장물에는 그룹 계층, 단순화 전 proxy, 겹침을 제거한 fragments, 로봇 장착 위치,
작업 프레임과 원본 USD Prim 경로가 포함된다. 원본 USD는 수정하지 않는다.

Windows에서는 원래 디렉터리의 NTFS 8.3 경로로 USD를 열어 수집된 상대 참조가
260자 제한에 걸리는 문제를 줄인다. 실제 필요한 에셋 누락은 오류로 처리한다.

## A25와 최상단 conveyor 연결

수집된 환경의 연결 대상은 A25의 **높은 쪽 운송면**과
`ConveyorGroup/conv_top/ConveyorBelt_A05_PR_NVD_04`다. 두 운송면은 XY에서
이미 겹치므로 A25 assembly의 기존 translation Z만 약 `+28.663 mm` 보정한다.
높이는 약 `2.379307 m`로 맞추며 ToteGroup·UncaseGroup의 conveyor, 로봇과
작업 프레임 위치는 유지한다. 겹침 제거 후에도 두 운송면의 전체 합집합은
유지되고, 겹친 부피만 한 소유자의 fragment로 남는다.

원본 `USD/Collected_scene_layout/scene_layout.usd`에 이 위치를 저장한다.
USD 자산은 Git에서 제외되므로 원본을 다시 내려받았을 때는 아래 명령으로
같은 연결을 재현한다. 이미 높이가 맞으면 USD를 다시 수정하지 않는다.

```powershell
python scripts/align_a25_conveyors.py
```

첫 실행 전 원본은
`.cache/layout_source_backup/scene_layout.before_conveyor_connection.usd`에
보관하고, 변경량과 원본/수정 hash는 `outputs/layout_conveyor_alignment.json`에
기록한다. 새 USD를 저장한 후에는 layout 프로세스를 다시 시작한다.

## USD 작성 규칙

- 선택한 그룹 아래의 Mesh 및 USD 기본 도형은 월드 변환과 실제 크기를 읽어
  방향을 유지한 박스로 단순화한다. 부모 이동, scale, `metersPerUnit` 및 Y/Z
  `upAxis`를 반영한다. 새 형상을 해당 그룹 아래에 추가하면 가져오기 대상이 된다.
- 그룹 이름이나 로봇 계층을 바꾸면 `config/cell_layout.yaml`의 `groups`와
  `robot_base_prims`를 실제 Prim 경로에 맞춘다. 로봇 경로는 장착 기준 `base_link`다.
- 새 작업 물체를 환경에서 제외하려면 해당 Prim에 문자열 속성
  `userProperties:layoutRole = "sku"`를 지정하거나 `usd_layout.exclude_prim_paths`에
  경로를 넣는다. `ignore_name_patterns`로 제외할 이름 패턴도 추가할 수 있다.
- 컨베이어의 벨트/롤러가 본체와 구분되어 있어야 다리를 제외한 운반 영역을
  추출할 수 있다. 현재 수집된 컨베이어의 `ConveyorBelt`, `Roller`, `_Belt` 명명에
  맞춰 선택한다. 다른 에셋 구조에는 선택 규칙을 추가해야 한다.
- 참조와 payload 파일을 함께 보관한다. PointInstancer는 개별 Prim으로 펼친 뒤
  사용한다. 재질이나 Isaac 전용 물리 플러그인은 이 환경의 입력에 필요하지 않다.
