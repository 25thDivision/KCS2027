# 기준
---
- 대상은 `./Ising-decoding`(NVIDIA/Ising-Decoding clone, 커밋 `12bbeb13c5cfc1a064171ac91b07aea61f2eaae8`)의 `code/`다. 아래 내용은 전부 이 커밋의 파일을 읽거나 실행해서 확인한 것이다.
- 이 저장소 쪽 래퍼는 `decoders/nvidia_predecoder.py`, 테스트는 `tests/test_nvidia.py`다.
# 체크포인트
---
- 저장소에는 가중치가 없다. HuggingFace gated 모델로 배포되고 저장소당 fp16 safetensors 파일 하나다.
| 이름 | HuggingFace 저장소 | 파일 | R | model_id |
|---|---|---|---|---|
| Fast | `nvidia/Ising-Decoder-SurfaceCode-1-Fast` | `ising_decoder_surface_code_1_fast_r9_v1.0.77_fp16.safetensors` | 9 | 1 |
| Accurate | `nvidia/Ising-Decoder-SurfaceCode-1-Accurate` | `ising_decoder_surface_code_1_accurate_r13_v1.0.86_fp16.safetensors` | 13 | 4 |
- 받는 방법은 `conda activate KCS2` 후 `hf auth login`, 이어서 `hf download <저장소> --local-dir Ising-decoding/models/`다. 두 저장소 각각에서 접근 승인을 받아야 한다.
- Fast 파일의 메타데이터는 `model_id 1`, `num_filters [128, 128, 128, 4]`, `kernel_size [3, 3, 3, 3]`, `receptive_field 9`, `quant_format fp16`, `model_version predecoder_memory_v1`이다. 파라미터 수는 912772개다.
- Accurate의 model_id 4는 `code/model/registry.py`의 R 계산식(R = 1 + 합(k - 1))으로 6층 k=3이 R 13이 되는 것에서 대응시킨 것이다. 파일 메타데이터로는 아직 확인하지 못했다(접근 승인 전).
# 모델과 로더
---
- 레지스트리는 `code/model/registry.py`의 `get_model_spec(model_id)`다. model_id는 1 ~ 5와 "B"(color 전용 cascade)다. 반환값 `PublicModelSpec`에 `num_filters`, `kernel_size`, `receptive_field`, `model_version`이 있다.
- 모델 클래스는 `code/model/predecoder.py`의 `PreDecoderModelMemory_v1`이고 `code/model/factory.py`의 `ModelFactory.create_model(cfg)`가 `cfg.model.version`으로 고른다. 생성자는 `cfg.distance`, `cfg.n_rounds`, `cfg.model.{dropout_p, activation, num_filters, kernel_size, input_channels, out_channels}`를 읽는다.
- 숨은 기본값은 `code/workflows/config_validator.py`에 있다. surface는 `dropout_p 0.05`, `activation gelu`, `input_channels 4`, `out_channels 4`다.
- safetensors 로더는 `code/export/safetensors_utils.py`의 `load_safetensors(path, model_id=None, device)`다. 파일 메타데이터의 model_id로 최소 cfg를 만들고(`omegaconf` 필요), 모델을 만든 뒤 fp16이면 `model.half()`를 하고 `strict=True`로 가중치를 넣는다. `(model, metadata)`를 돌려준다.
- 환경변수 `PREDECODER_SAFETENSORS_CHECKPOINT`와 `PREDECODER_MODEL_CHECKPOINT_FILE`은 NVIDIA 자체 실행 스크립트용이고 이 저장소 래퍼는 쓰지 않는다. 래퍼는 경로를 직접 받는다.
- 이 저장소 래퍼 `load_predecoder`는 `load_safetensors`를 그대로 부른 뒤 `model.float()`로 fp32로 바꾼다. CPU에서도 돌게 하기 위해서다. fp16 값은 fp32로 정확히 표현되므로 가중치는 같고, 연산 정밀도만 NVIDIA 실행(autocast fp16)과 다르다.
## KCS 래퍼가 쓰던 이름
---
| 이름 | 현재 상태 |
|---|---|
| `model.registry.get_model_spec` | 그대로 |
| `PreDecoderModelMemory_v1` | 그대로 |
| dropout 0.05, gelu | 그대로 |
| `.pt` 체크포인트 `torch.load` | 공개 배포는 safetensors로 바뀜 |
| measurement 기반 datapipe | detector 기반으로 바뀜 |
- 로드 방식과 datapipe가 바뀌었으므로 KCS 래퍼는 복사하지 않고 다시 썼다.
## 패키지 이름 충돌
---
- `Ising-decoding/code`의 최상위 패키지 `model`, `evaluation`이 이 저장소의 `model/`, `evaluation/`과 이름이 같다. 한 프로세스에서 둘을 그냥 import하면 먼저 불린 쪽이 다른 쪽을 가린다.
- 래퍼의 `nvidia_imports()` 컨텍스트가 이 저장소 모듈을 잠시 치우고 NVIDIA 쪽을 import하게 한 뒤 되돌린다. NVIDIA 코드의 import는 함수 안의 지연 import까지 이 블록 안에서 일어나야 한다.
# 입력 텐서
---
- 형태는 `(B, 4, T, D, D)` float32다. 채널은 순서대로 `x_type`, `z_type`, `x_present`, `z_present`다.
- 현재 코드는 measurement가 아니라 detector 벡터에서 입력을 만든다. 구현이 두 군데 있고 결과는 같다. `code/data/predecoder_transform.py`의 `dets_to_predecoder_inputs`와 `code/qec/surface_code/detector_input.py`의 `SurfaceDetectorInputTransform`이다.
- detector 벡터의 순서는 `half = (D*D - 1) / 2`로 두고, 기준 basis의 첫 라운드 detector `half`개, 이어서 라운드 1 ~ T-1마다 X `half`개와 Z `half`개, 마지막에 경계 detector `half`개다. 전체 길이는 `2 * T * half`다.
- basis Z에서 `z_type[t]`는 라운드 t의 Z detector다. `x_type[t]`는 라운드 1 ~ T-2의 X detector이고 라운드 0과 T-1은 0으로 가린다. 마지막 라운드의 X detector는 입력에 들어가지 않는다.
- `x_present`, `z_present`는 stabilizer가 놓인 칸에 가중치를 둔 `D x D` 지도다. weight-4는 1.0, weight-2는 0.5다. basis Z에서 `x_present`의 라운드 0과 T-1은 0이다.
- 경계 detector는 모델 입력에 들어가지 않고 잔차로 그대로 넘어간다.
## 격자 배치
---
- stabilizer는 자기 지지 data 칸 중 하나(anchor)에 놓인다. `code/qec/surface_code/data_mapping.py`의 규칙은 아래와 같다. 행이 작은 쪽이 위, 열이 작은 쪽이 왼쪽이다.
| 종류 | weight-4 | 가로 쌍 | 세로 쌍 |
|---|---|---|---|
| X | 왼쪽 위 | 왼쪽 | 위 |
| Z | 오른쪽 위 | 오른쪽 | 위 |
- stabilizer 번호에서 칸으로 가는 표는 `compute_stabX_to_data_index_map(D, rotation)`, `compute_stabZ_to_data_index_map(D, rotation)`이고 순서는 `SurfaceCode.hx`, `SurfaceCode.hz`의 행 순서와 같다.
# orientation
---
- `data.code_rotation`은 XV, XH, ZV, ZH 네 값이고 공개 별칭은 순서대로 O1, O2, O3, O4다. 기본값은 O1이다.
- 이 저장소의 surface 격자는 data 번호를 `행 * D + 열`, 행 = (y - 1) / 2, 열 = (x - 1) / 2로 두면 ZH(O4)와 칸 단위로 같다. X 지지, Z 지지, logical Z(0열)가 재배열 없이 일치한다. O1, O2, O3은 일치하지 않는다(`tests/test_nvidia.py`의 `test_orientation_is_o4`).
- 좌우 또는 상하로 뒤집으면 O1과 같아지지만 래퍼는 뒤집지 않고 O4를 쓴다.
- 공개 Fast 체크포인트는 orientation에 따라 성능이 다르다. NVIDIA 자체 회로, datapipe, 평가 모듈, DEM만으로 잰 값이다(basis Z, 임계값 0, seed 0).
| D, T | p | orientation | shot | MWPM | pre-decoder + MWPM |
|---|---|---|---|---|---|
| 3, 3 | 0.003 | XV | 200000 | 0.00619 | 0.00572 |
| 3, 3 | 0.003 | XH | 200000 | 0.00643 | 0.04404 |
| 3, 3 | 0.003 | ZV | 200000 | 0.00660 | 0.03052 |
| 3, 3 | 0.003 | ZH | 200000 | 0.00615 | 0.00632 |
| 9, 9 | 0.003 | XV | 50000 | 0.00132 | 0.00082 |
| 9, 9 | 0.003 | XH | 50000 | 0.00110 | 0.00916 |
| 9, 9 | 0.003 | ZV | 50000 | 0.00102 | 0.01144 |
| 9, 9 | 0.003 | ZH | 50000 | 0.00110 | 0.00076 |
- XV와 ZH에서는 MWPM 단독과 같거나 낮고, XH와 ZV에서는 크게 나빠진다. 이 저장소 격자가 ZH이므로 O4를 그대로 쓰는 것이 맞다.
- 이 표는 2026-11-17에 임시 스크립트로 한 번 잰 값이고 결과 파일로 남기지 않았다. 다시 재려면 `QCDataPipePreDecoder_Memory_inference`와 `PreDecoderMemoryEvalModule`을 orientation별로 돌리면 된다.
# 출력과 잔차
---
- 출력은 `(B, 4, T, D, D)` logit이다. 채널은 순서대로 data의 Z 보정, data의 X 보정, X check 측정 보정, Z check 측정 보정이다. 이진화는 `logit >= 0`이다(`sample_predictions`, `th_data`, `th_syn` 기본 0).
- 잔차 계산은 `code/evaluation/logical_error_rate.py`의 `PreDecoderMemoryEvalModule.forward`에 있다. stabilizer 종류별로 아래와 같다. `syn`은 측정 보정, `S`는 data 보정이 만드는 신드롬(`H @ data`)이다.
$$R_0 = d_0 + syn_0 + S_0$$
$$R_t = d_t + syn_t + syn_{t-1} + S_t \quad (t \ge 1)$$
- logical 보정 `pre_L`은 basis Z에서 logical Z 지지 위의 X 보정을 모든 라운드에 걸쳐 XOR한 값이다. 최종 logical 예측은 잔차를 global 디코더로 푼 결과와 `pre_L`의 XOR이다.
- NVIDIA 모듈은 basis Z에서 마지막 라운드의 X detector를 잔차에서도 뺀다(입력에서 가린 값을 그대로 쓴다). 이 저장소 래퍼는 원래 detector에 보정을 XOR하므로 그 detector를 남긴다. 나머지는 비트 단위로 같다(`test_residual_matches_nvidia_eval_module`).
# 검증 범위
---
- 모델 카드(`Ising-decoding/models/README.md`)는 receptive field보다 작은 distance를 검증 범위 밖이라고 적는다. Fast는 R 9, Accurate는 R 13이고 이 저장소의 d 3, d 5는 둘 다 그보다 작다.
- 모델 카드는 노이즈가 학습 분포와 맞는 회로 수준 Pauli 채널이어야 하고, 다른 노이즈 모델이나 하드웨어의 신드롬에는 다시 학습한 체크포인트가 필요하다고 적는다.
- 이 저장소의 추상 surface d 3 회로(cycles 3, 200000 shot, seed 11)에서 Fast로 잰 값은 아래와 같다. 여섯 조합 모두 pre-decoder를 붙인 쪽이 MWPM 단독보다 높다. 수용 기준이 아니라 기록이다.
| 노이즈 프로파일 | p | MWPM | Fast + MWPM |
|---|---|---|---|
| dp0.001_mf0.01_rf0.01_gd0.008 | 0.005 | 0.0356 | 0.0550 |
| dp0.001_mf0.01_rf0.01_gd0.008 | 0.05 | 0.0911 | 0.1121 |
| dp0.005_mf0.02_rf0.02_gd0.015 | 0.005 | 0.1095 | 0.1408 |
| dp0.005_mf0.02_rf0.02_gd0.015 | 0.05 | 0.1693 | 0.1993 |
| dp0.01_mf0.05_rf0.05_gd0.01 | 0.005 | 0.1838 | 0.2316 |
| dp0.01_mf0.05_rf0.05_gd0.01 | 0.05 | 0.2435 | 0.2822 |
- 이 값도 임시 스크립트로 한 번 잰 것이고 결과 파일은 없다. 경로는 `decoders/nvidia_predecoder.nvidia_mwpm_decode`와 `baseline/mwpm.build_matching`이다.
# 의존성
---
- `code/requirements_public_inference.txt`의 목록 중 `KCS2`에 없던 것은 omegaconf, hydra-core, ldpc, beliefmatching, onnx, chromobius였고 2026-11-17에 설치했다. 딸려 온 패키지까지 10개가 추가됐고 기존 핀(torch, numpy, stim, PyMatching, scipy, qiskit)은 바뀌지 않았다.
- 버전은 `environment.yml`에 고정했다.
