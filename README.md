# 목적
---
세 가지 양자 오류 정정 코드를 세 가지 하드웨어 플랫폼에서 돌리고, 여러 디코더의 logical error rate를 같은 지표로 비교하는 파이프라인이다.
- 공식 지표는 head-LER이다. 디코더의 logical head가 낸 logical Z flip 예측과 실제 flip이 다른 비율이고, 시뮬레이션과 실기기 양쪽에서 계산한다.
- ECR(시뮬레이션 전용)과 parity_LER은 진단용으로만 함께 기록한다.
- 디코더 목표 구성은 ML 디코더 8종(CNN, GCN, GCNII, GAT, APPNP, GNN, GraphTransformer, GraphMamba), MWPM, 하이브리드 2종(MWPM+ML, ML+MWPM), NVIDIA pre-decoder(surface 한정)다.
- 이 저장소는 Heavy-hex_Surface_Code 저장소를 바탕으로 KCS 저장소의 벤치마크 구조를 옮겨 오는 중이다. 아래 "현재 구현 상태"가 지금 실제로 동작하는 범위다.
# 코드와 플랫폼
---
| 코드 | `--code` | `--distance` | 플랫폼 | 기본 백엔드 |
|---|---|---|---|---|
| heavy-hex surface code | `heavyhex` | 33, 35, 53 | IBM Heron | `ibm_pittsburgh` |
| rotated surface code | `surface` | 3, 5 | IBM Nighthawk | `ibm_miami` |
| color code | `color` | 3, 5 | IonQ Forte-1 | `ionq_forte1` |
- heavyhex의 distance는 (dx, dz)를 두 자리 수로 적는다. 33은 (3,3), 35는 (3,5), 53은 (5,3)이고, 3은 33의 줄임으로 받는다.
- 파일 이름의 distance 태그는 `dataset_generation.distance_tag` 하나가 정한다. heavyhex (3,3)은 `d3`, (3,5)는 `d35`, (5,3)은 `d53`이고 surface와 color는 `d3`, `d5`다.
# 현재 구현 상태
---
| 항목 | 상태 |
|---|---|
| heavyhex (3,3) 회로, 데이터셋, 게이트, 하드웨어 경로 | 동작 |
| surface d=3, d=5 회로, 데이터셋, 게이트, 하드웨어 경로 | 동작 |
| heavyhex (3,5), (5,3) | 미구현 |
| color d=3, d=5 | 미구현 |
| MWPM 기준선 (heavyhex (3,3), surface d=3, d=5) | 동작 |
| 모델 8종 (`cnn`, `gcn`, `gcnii`, `gat`, `appnp`, `gnn`, `graph_transformer`, `graph_mamba`) | 동작 |
| 하이브리드 2종 (heavyhex (3,3), surface d=3, d=5) | 동작 |
| NVIDIA pre-decoder (surface d=3, d=5) | 동작, 체크포인트는 따로 받아야 함 |
- 미구현인 (코드, distance)를 지정하면 파일을 만들기 전에 `NotImplementedError`로 멈춘다. 구현된 조합 목록은 `dataset_generation.IMPLEMENTED`다.
- `pipeline.sbatch`는 `config.json`의 `pipeline.distances`에 적힌 조합 중 구현된 것만 돌고 나머지는 건너뛴다고 출력한다.
# 디렉터리 구조
---
- `circuits/heavyhex/` heavy-hex 회로 정의, 37큐빗 하드웨어 회로, coupling map 조회, DD
- `circuits/rotatedSurface/` rotated surface code 회로(홀수 distance 공용)와 하드웨어 임베딩
- `circuits/placement.py` 캘리브레이션을 반영한 자동 배치
- `dataset_generation/` Stim 회로 생성기, 데이터셋 생성(`make_dataset.py`), QPU 캘리브레이션 평균 프로파일(`make_qpu_avg_profile.py`), 설정 읽기와 코드·distance 축 정의(`__init__.py`)
- `model/` 모델 레지스트리(`__init__.py`), 모델 파일 8개, 공용 손실과 그래프 기반 클래스(`common.py`), detector 그래프(`graph.py`), 데이터 로더(`data.py`)
- `decoders/` 하이브리드 디코더(`hybrid.py`), NVIDIA pre-decoder 래퍼(`nvidia_predecoder.py`)
- `docs/` NVIDIA 저장소 API 정리(`nvidia_api_notes.md`)
- `baseline/mwpm.py` PyMatching 기반 MWPM 기준선
- `evaluation/metrics.py` LER, ECR, parity_LER
- `hardware/run_hw.py` QPU 제출, 수거, 분석
- `verification/` 규약 게이트 스크립트
- `tests/` pytest 테스트
- `train.py` 학습 진입점
- `pipeline.sbatch`, `train.sbatch` Slurm 스크립트
- `config.json` 설정 파일
- `environment.yml` 의존성
# 환경
---
의존성은 `environment.yml` 하나로 관리하고 환경 이름은 `KCS2`다.
```bash
mamba env create -f environment.yml -p ~/.conda/envs/KCS2
conda activate KCS2
# GraphMamba 커널 (사전 빌드 휠 두 줄, environment.yml 머리말과 동일)
pip install --no-deps https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install --no-deps https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
python -c "import torch, mamba_ssm, causal_conv1d; print(torch.__version__, torch.cuda.is_available())"
```
- 서버의 mamba는 `-n 이름`으로 만들면 root 소유 경로에 쓰려다 실패하므로 `-p`로 만든다.
- 휠 두 줄은 `--no-deps`가 필요해 `environment.yml`에 넣지 않았다. 빠뜨리면 `mamba_ssm`이 없다.
- 휠은 Python 3.12, torch 2.9, CUDA 12, linux x86_64 조합 전용이다.
# 실행 순서
---
게이트, 데이터셋, 학습, 하드웨어 순서다. 모든 명령은 저장소 루트에서 실행한다.
```bash
# 0) 규약 게이트. ALL PASS를 확인한 뒤에만 데이터셋을 만든다
python verification/verify_equivalence.py          # heavyhex
python verification/verify_rotatedSurface.py       # surface (d=3, d=5)
# 1) 데이터셋 (--code 기본 heavyhex, --distance 기본 3)
python dataset_generation/make_dataset.py --smoke
python dataset_generation/make_dataset.py --code surface
# 2) 학습
python train.py --model cnn --smoke
python train.py --code surface --model gnn
# 3) MWPM 기준선만 따로
python baseline/mwpm.py -n realistic/dp0.001_mf0.01_rf0.01_gd0.008 -p 0.005
# 4) 하드웨어 (keys.json 필요)
python hardware/run_hw.py submit --backend ibm_pittsburgh --code heavyhex --dry-run
python hardware/run_hw.py collect
python hardware/run_hw.py analyze --job-id <ID>
# 테스트
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider tests/
```
- `submit`에 `--dry-run`을 붙이면 제출 직전까지만 수행한다. 빼면 실제 QPU 잡이 제출된다.
- `analyze`는 raw LER, MWPM, 그리고 `checkpoint/`에서 코드와 distance가 맞는 모든 체크포인트를 한 표로 낸다.
# config.json
---
- `noise_profiles` 노이즈 프로파일 레지스트리다. 4-파라미터 프로파일(`realistic/dp…_mf…_rf…_gd…`)은 직접 편집하고, `qpu/…` 항목은 `make_qpu_avg_profile.py`가 이 섹션만 갱신한다.
- `train`, `dataset`, 최상위 `cycles` 학습 하이퍼파라미터와 데이터셋 샘플 수의 기본값이다. `cycles`는 양쪽이 함께 쓴다.
- `sweep` 스윕 정의다. `runs`의 각 항목이 (모델, 노이즈, p, 하이퍼파라미터 덮어쓰기) 한 벌이고, 섹션이 있으면 `make_dataset.py`와 `train.py`가 자동으로 따른다. `--config none`으로 끈다.
- `pipeline` 파이프라인 고정 설정이다. `conda_env`, `codes`, `distances`(코드별 distance 목록), `backends`(코드별 백엔드), `qpu_pubs`, `profile_runs`, `shots`, `train_args`가 있다.
- 우선순위는 `config.json` 기본값, CLI 인자, 스윕 run 항목 순으로 뒤가 이긴다.
# 결과 위치
---
- 데이터셋 `dataset/<code>/<노이즈태그>/{train,test}_<dtag>_c<cycles>_p<p>_X.npz`
- 학습 로그 `results/train/{MODEL}_<code>_<dtag>_c<cycles>_p<p>_<노이즈태그>.csv`
- 체크포인트 `checkpoint/{MODEL}_<code>_<dtag>_c<cycles>_p<p>_<노이즈태그>.pt` (검증 head-LER이 가장 낮을 때만 갱신)
- QPU 런 `hardware/runs/<backend>_<타임스탬프>/` (job.json, 캘리브레이션 스냅샷, 제출 회로, raw.npz)
- QPU LER 리포트 `results/hardware/<backend>_<코드>_<타임스탬프>.csv`
- 위 경로는 전부 `.gitignore` 대상이다.
# 고정 규약
---
Stim 회로와 QPU 회로는 check-value 수준에서 비트 단위로 일치해야 하고, 게이트 스크립트가 이를 검사한다.
- Stim은 ancilla를 측정 후 리셋으로 처리하고, QPU는 리셋 없는 ancilla의 raw 비트를 `check_values()`의 XOR chain으로 복원한다.
- detector는 Z-check가 cycle 0부터, X-check가 cycle 1부터(cycle 간 XOR)이고 마지막에 final-data 기반 Z detector가 붙는다. observable은 logical Z다.
- 데이터셋 텐서는 `(2*cycles, H, W)`이고 채널은 cycle마다 Z 평면, X 평면 순서다. 모델 입력에는 final-Z detector 채널 하나가 붙어 `2*cycles + 1` 채널이 된다. CNN, 그래프 모델, MWPM이 같은 detector 정보를 본다.
- per-qubit 라벨은 final-data의 측정 flip(`stim.FlipSimulator`)이다. X-check 측정 때문에 final data의 개별 비트는 무작위이고 parity만 결정론적이기 때문이다. logical 라벨은 그 flip의 logical Z parity이며 실제 측정 비트의 parity와 같다.
- 모델 인터페이스는 `forward(x) -> (qubit_logits, logical_logits)`, 모듈 수준 `compute_loss(...) -> (total, loss_logical, loss_qubit)`, 모듈 수준 `prepare_features(features, final_bits, code, distance)`다.
# keys.json
---
QPU 접근 정보는 저장소 루트의 `keys.json`에서만 읽는다. `keys.example.json`을 복사해 `ibm_API`와 `ibm_instance`(CRN)를 채운다. `keys.json`은 `.gitignore` 대상이고, 없으면 파이프라인의 QPU 수거와 제출 단계를 건너뛴다.
# Slurm
---
```bash
sbatch train.sbatch                 # 학습만
sbatch pipeline.sbatch              # 게이트, QPU 수거, QPU 제출, 프로파일, 데이터셋, 학습 1회
LOOPS=5 sbatch pipeline.sbatch      # 5회 반복 (루프 끝에 자기 재제출)
SMOKE=1 sbatch pipeline.sbatch      # 리허설 (스모크 데이터셋과 학습, QPU는 dry-run)
```
- 두 스크립트의 `#SBATCH --partition` 값은 사용하는 서버의 파티션 이름과 맞아야 한다.
- 환경 이름은 `config.json`의 `pipeline.conda_env`(기본 `KCS2`)에서 읽고, `train.sbatch`는 `CONDA_ENV=이름`으로 덮어쓸 수 있다.
