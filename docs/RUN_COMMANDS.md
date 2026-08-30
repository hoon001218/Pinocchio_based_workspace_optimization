# 실행 명령어 모음

이 문서는 저장소를 처음 받은 PC에서 환경을 준비하고, 로봇 에셋을
검증하고, Pinocchio 계산 및 MeshCat UI를 실행하는 명령을 한곳에 모은
운영용 문서다. 별도 설명이 없는 명령은 모두 **저장소 최상위 폴더**에서
실행한다.

프로그램 구조와 계산 의미는 [ARCHITECTURE.md](ARCHITECTURE.md), 현재 결과와
후속 작업은 [HANDOFF.md](HANDOFF.md)를 참고한다.

## 1. 처음 한 번만 수행하는 설치

### 1.1 저장소 받기

```powershell
git clone REPOSITORY_URL
Set-Location Pinocchio_based_workspace_optimization
git submodule update --init BGF
```

`BGF` 내부의 중첩 submodule 선언은 현재 불완전하므로
`git submodule update --init --recursive`는 사용하지 않는다. BGF는 환경과
좌표계를 이해하기 위한 참고 자료이며 현재 Python runtime의 필수 의존성은
아니다.

### 1.2 Conda 명령을 찾을 수 없을 때

먼저 현재 shell에서 Conda가 보이는지 확인한다.

```powershell
Get-Command conda -ErrorAction SilentlyContinue
```

출력이 없더라도 Conda가 설치되어 있을 수 있다. 일반적인 사용자 설치
위치는 다음 명령으로 확인할 수 있다.

```powershell
Get-Item `
  "$env:USERPROFILE\miniconda3\Scripts\conda.exe", `
  "$env:USERPROFILE\anaconda3\Scripts\conda.exe" `
  -ErrorAction SilentlyContinue
```

Miniconda 위치가 확인되었다면 현재 PowerShell 세션에서는 다음처럼 실행할
수 있다. Anaconda를 사용한다면 경로의 `miniconda3`만 `anaconda3`로 바꾼다.

```powershell
$CondaExe = "$env:USERPROFILE\miniconda3\Scripts\conda.exe"
& $CondaExe --version
```

이후 문서의 `conda`를 `& $CondaExe`로 치환하면 PowerShell 초기화 여부와
관계없이 실행할 수 있다. 예: `conda run ...` 대신 `& $CondaExe run ...`.

향후 새 PowerShell에서도 `conda` 명령 자체를 사용하려면 한 번 초기화한 뒤
PowerShell을 다시 연다.

```powershell
& $CondaExe init powershell
```

### 1.3 단일 Python 환경 생성 및 패키지 설치

```powershell
conda env create --prefix .\.venv --file environment.yml
conda run --prefix .\.venv python -m pip install -e . --no-deps --no-build-isolation
```

프로젝트는 Pixi를 사용하지 않는다. Windows에서는 Conda DLL 경로가 빠질
수 있으므로 `.\.venv\python.exe`를 직접 실행하지 말고 `conda activate` 또는
`conda run --prefix`를 사용한다.

저장소 경로에 한글이 포함된 Windows 환경에서는 다음 설정을 현재 shell에
적용하는 것이 안전하다.

```powershell
$env:PYTHONUTF8 = "1"
```

## 2. 로봇 에셋 준비와 설치 검증

UR20 공식 URDF와 mesh는 저장소에 포함되어 있다. FANUC SR-12iA mesh는
라이선스 때문에 Git에 포함하지 않으며 각 PC에서 한 번 준비해야 한다.

NVIDIA에 고정된 원본을 받아 SHA-256을 검증하고 변환하려면 라이선스 확인
후 다음 명령을 실행한다.

```powershell
conda run --prefix .\.venv decanting-setup --accept-fanuc-license
```

이미 허가된 `geometries.usd` 파일이 있다면 다운로드 대신 로컬 파일을
사용한다.

```powershell
conda run --prefix .\.venv decanting-setup `
  --sr12ia-usd "C:\path\to\Fanuc\sr12ia\payloads\geometries.usd"
```

기존 SR-12iA bundle, scene config, UR20 bundle, 결과 cache를 변경 없이
검사하는 doctor 명령은 다음과 같다.

```powershell
conda run --prefix .\.venv decanting-setup
```

SR-12iA bundle을 강제로 다시 생성해야 할 때만 `--refresh-sr12ia`를 붙인다.

```powershell
conda run --prefix .\.venv decanting-setup `
  --accept-fanuc-license `
  --refresh-sr12ia
```

에셋만 준비하고 precomputed cache 검증은 생략하려면
`--skip-cache-validation`을 사용할 수 있다.

## 3. 권장 검증 명령

전체 자동 테스트:

```powershell
conda run --prefix .\.venv python -m pytest
```

설치 상태와 tracked cache를 함께 검사:

```powershell
conda run --prefix .\.venv decanting-setup
```

각 CLI의 현재 옵션을 확인하려면 다음을 사용한다.

```powershell
conda run --prefix .\.venv decanting-setup --help
conda run --prefix .\.venv decanting-view --help
conda run --prefix .\.venv decanting-evaluate --help
conda run --prefix .\.venv decanting-precompute --help
conda run --prefix .\.venv decanting-playback --help
```

## 4. MeshCat 실행

### 4.1 사전 계산된 작업 결과 UI

가장 자주 사용하는 명령이다. 기본 cache는 검증된 40-case commissioning
cache인 `outputs/precomputed_working_cases.json.gz`다.

```powershell
conda run --prefix .\.venv decanting-playback --open
```

UI에서 SKU, pallet corner, 동시/순차 모드, 공정 단계, 세부 check와 sample을
선택하면 저장된 로봇 자세 및 manipulability ellipsoid가 표시된다. UI에서
IK를 다시 계산하지 않는다. 숫자 파라미터는 슬라이더와 직접 입력을 함께
제공하지만, 둘 다 현재 cache에 있는 값만 정확히 선택한다. 입력한 값이
cache에 없으면 오류와 사용 가능한 값 목록을 표시한다. 새 값을 사용하려면
해당 case-grid YAML에 값을 추가하고 `decanting-precompute`를 다시 실행한다.

기본 working cache는 검증된 성공 배치를 빠르게 여는 용도라서 UR/SR base,
lift, tote offset, J3가 각각 한 값뿐이다. 이 축의 slider가 비활성화되는 것은
정상이다. SKU, corner와 coordination mode는 복수 선택할 수 있다.

720-case nominal 진단 cache를 열려면 파일을 명시한다.

```powershell
conda run --prefix .\.venv decanting-playback `
  --cache outputs\precomputed_nominal_cases.json.gz `
  --open
```

충돌 형상도 함께 표시:

```powershell
conda run --prefix .\.venv decanting-playback `
  --show-collisions `
  --open
```

SR mesh가 준비되지 않은 PC에서 임시 근사 형상을 명시적으로 허용하려면
다음을 사용한다. 실제 형상 검토용 명령은 아니다.

```powershell
conda run --prefix .\.venv decanting-playback `
  --allow-sr-visual-fallback `
  --open
```

### 4.2 환경 및 nominal robot pose만 확인

결과 cache UI가 아니라 환경 배치 자체를 확인할 때 사용한다.

```powershell
conda run --prefix .\.venv decanting-view --open
```

파라미터 변경 예:

```powershell
conda run --prefix .\.venv decanting-view `
  --ur-base -1.300 -7.350 0.850 90 `
  --sr-base -0.300 -7.073643684 0.665 -90 `
  --lift-height-mm 200 `
  --sku 123591 `
  --corner southwest `
  --tote-table-offset-mm 700 `
  --sr-j3-stroke-mm 450 `
  --open
```

## 5. Pinocchio 후보 하나 평가

다음 명령은 viewer 없이 UR20 Pinocchio IK, Jacobian manipulability,
Coal collision 및 SR 작업영역 검사를 수행하고 compact JSON을 만든다.

```powershell
conda run --prefix .\.venv decanting-evaluate `
  --ur-base -1.300 -7.350 0.850 90 `
  --sr-base -0.300 -7.073643684 0.665 -90 `
  --sku 123591 `
  --corner all `
  --coordination both `
  --lift-height-mm 200 `
  --tote-table-offset-mm 700 `
  --sr-j3-stroke-mm 450 `
  --output-json outputs\candidate.json
```

`--coordination`은 `both`, `simultaneous`, `sequential` 중 하나다. Step 7의
빈 박스 폐기는 SOFT 조건이며 실패해도 후보 전체를 폐기하지 않는다.

## 6. 모든 명시 case 사전 계산

사전 계산은 YAML에 명시된 유한 Cartesian product만 평가한다. 연속적인
base 위치를 암묵적으로 생성하지 않는다.

### 6.1 검증된 40-case commissioning cache 재생성

```powershell
conda run --prefix .\.venv decanting-precompute `
  --grid config\case_grid_working.yaml `
  --output-json outputs\precomputed_working_cases.json.gz
```

현재 이 grid는 `5 SKU x 4 pallet corner x 2 coordination mode = 40`개다.
이 중 full-cell 통과는 12개이며, 순차 모드 UR20 HARD 조건만 보면 20개 모두
통과한다.

### 6.2 720-case nominal 진단 cache 재생성

현재 저장된 nominal cache와 동일하게 task pose 위주로 빠르게 생성:

```powershell
conda run --prefix .\.venv decanting-precompute `
  --grid config\case_grid_nominal.yaml `
  --output-json outputs\precomputed_nominal_cases.json.gz `
  --cartesian-step-mm 10000 `
  --cartesian-rotation-step-deg 180 `
  --joint-interpolation-samples 0
```

이 720개는 진단 case 총수이며 통과 case 수가 아니다. 위 설정은 중간 경로를
충분히 sampling하지 않으므로 continuous collision-free path 결과로 해석하면
안 된다.

정상적인 path sampling 설정으로 별도 cache를 만들려면 override 없이
실행한다.

```powershell
conda run --prefix .\.venv decanting-precompute `
  --grid config\case_grid_nominal.yaml `
  --output-json outputs\precomputed_nominal_path_sampled.json.gz
```

큰 grid를 실수로 실행하지 않도록 case 상한을 둘 수 있다.

```powershell
conda run --prefix .\.venv decanting-precompute `
  --grid config\case_grid_nominal.yaml `
  --output-json outputs\precomputed_nominal_limited_example.json.gz `
  --max-cases 720
```

SKU나 lift height별 shard 계산에는 `--only-sku`와
`--only-lift-height-mm`을 반복해서 지정할 수 있다. 현재 CLI는 shard 생성은
지원하지만 shard merge용 별도 shell entry point는 제공하지 않는다.

## 7. 환경을 활성화해서 짧게 실행하는 방법

매번 `conda run --prefix .\.venv`를 쓰지 않으려면 환경을 활성화한다.

```powershell
conda activate .\.venv
$env:PYTHONUTF8 = "1"
decanting-setup
python -m pytest
decanting-playback --open
```

PowerShell에서 `conda activate`가 동작하지 않는다면 1.2절의 Conda 초기화를
수행하거나 계속 `conda run --prefix` 형식을 사용한다.

## 8. 다른 checkout 위치에서 실행할 때

Editable install이면 저장소 위치는 자동으로 인식된다. 코드를 이동하거나
새로 clone한 뒤에는 해당 위치에서 editable install을 다시 실행한다.

```powershell
conda run --prefix .\.venv python -m pip install -e . --no-deps --no-build-isolation
```

Non-editable install을 의도적으로 사용하는 경우에만 저장소 root를 명시한다.

```powershell
$env:DECANTING_WORKSPACE_ROOT = (Get-Location).Path
conda run --prefix .\.venv decanting-setup
```

코드나 URDF/mesh가 바뀌면 기존 cache fingerprint가 맞지 않는 것이 정상이다.
이때 cache 검사를 억지로 우회하지 말고 변경된 모델로 precompute cache를 다시
생성한다.

## 9. 자주 발생하는 문제

### `conda`가 명령으로 인식되지 않음

Conda가 없는 것이 아니라 PowerShell PATH/초기화가 빠진 경우가 많다. 1.2절에
따라 `conda.exe`의 실제 위치를 찾고 `& $CondaExe run ...` 형식으로 실행한다.

### NumPy 또는 BLAS DLL load 실패

`.\.venv\python.exe`를 직접 실행하지 않는다. `conda run --prefix .\.venv`나
활성화된 Conda shell을 사용한다.

### playback에서 SR-12iA mesh가 없다고 실패

정상적인 보호 동작이다. 먼저 다음을 수행한다.

```powershell
conda run --prefix .\.venv decanting-setup --accept-fanuc-license
```

라이선스가 허용된 로컬 USD가 있다면 `--sr12ia-usd` 방식을 사용한다.

### cache fingerprint mismatch

현재 config, workflow contract, URDF 또는 참조 mesh 내용이 cache 생성 시점과
다르다는 뜻이다. `decanting-setup`으로 원인을 확인하고 해당 grid의
`decanting-precompute` 명령으로 cache를 다시 생성한다.

### MeshCat 브라우저가 자동으로 열리지 않음

터미널에 출력된 HTTP 주소를 직접 브라우저에 입력한다. 포트를 고정하려면
playback에 `--port PORT`를 추가한다.
