# 목적
---
heavyhex (3,5)와 (5,3)에 대해, (3,3) 경로가 지키는 계약이 이 저장소에 어디까지 있는지 정리한다. 2026-11-17에 `circuits/heavyhex/`의 파일을 읽고 `diamond_generator.build_code`, `heavyhex_general.assign`을 실행해 확인한 내용이다. 수치의 출처는 `circuits/heavyhex/sol_33.json`, `sol_35.json`, `sol_53.json`과 위 두 함수의 반환값이다.
# 코드 정의
---
| 항목 | (3,3) | (3,5) | (5,3) |
|---|---|---|---|
| data 큐빗 | 17 | 29 | 29 |
| Z check | 7 | 12 | 14 |
| X check | 9 | 16 | 14 |
| logical Z 무게 | 3 | 5 | 3 |
| logical X 무게 | 3 | 3 | 5 |
| bridge 큐빗 | 12 | 22 | 22 |
| rung ancilla | 8 | 14 | 14 |
| 전체 큐빗 | 37 | 65 | 65 |
| relay check | X8 | X13, X15 | Z12, X13 |
- 정의는 `diamond_generator.build_code(dx, dz)`가 만든 격자 위에서 `heavyhex_general.solve`가 찾은 해이고 `sol_<dx><dz>.json`에 저장돼 있다. Z 지지, X 지지, logical Z(`zl`), logical X(`xl`)가 들어 있다.
- 세 경우 모두 X check와 Z check가 서로 commute하고, 큐빗 하나가 걸리는 Z check는 1개 또는 2개다. 그래서 `pymatching.Matching(H_z)`를 쓰는 하이브리드가 그대로 적용된다.
- rung ancilla는 하나가 한 cycle에 check 2개를 잰다 (세 경우 모두).
- data의 물리 번호는 `(r + 1) * 20 + (c + 1)` 규칙이다. (3,5)는 29 ~ 147, (5,3)은 25 ~ 151 범위다.
## (3,3) 일반해와 실제 (3,3) 경로의 차이
---
- `sol_33.json`의 (3,3)은 Z 7개, X 9개다. 지금 학습과 하드웨어에 쓰는 (3,3) 경로(`circuits/heavyhex/heavyhex_37q.py`의 `CHECK_DEFS`)는 Z 8개, X 8개다. data 17개의 물리 번호는 같지만 stabilizer 구성이 다르다.
- 즉 일반 경로(`heavyhex_general.py`)는 지금의 (3,3) 경로를 일반화한 것이 아니라 별도의 해다. (3,5), (5,3)을 일반 경로로 넣으면 (3,3)과 stabilizer 선택 규칙이 다른 코드가 된다.
# 계약 항목별 현황
---
| 항목 | (3,3) | (3,5), (5,3) |
|---|---|---|
| 코드 정의 (지지, logical) | `heavyhex_37q.py` | `sol_35.json`, `sol_53.json` |
| 하드웨어 회로, 정확성 우선 | `heavyhex_37q.py` | `heavyhex_general.GeneralDiamondCircuit` |
| 하드웨어 회로, depth 최적화 | `heavyhex_depth7_opt_for_37q.py` | `heavyhex_depth_opt.DepthOptDiamond` |
| check_values (XOR chain) | 있음 | 있음, relay 잔여 보정 포함 |
| 고정 CYCLE_ORDER 상수 | 있음 | 없음, 이름순 정렬로 정함 |
| 무노이즈 Aer 검증 스크립트 | `test_37q.py` 등 | `test_general.py` |
| 추상 Stim 생성기 | `heavyhex33_stim.py` | 없음 |
| 텐서 격자 (ANC_COORD) | 4x5 | 없음 |
| 하드웨어형 Stim (QPU 프로파일) | `heavyhex37_qpu_stim.py` | 없음 |
| 검증 게이트 (Stim, Aer 등가) | `verify_equivalence.py` | 없음 |
| detector 그래프 (`code_adapter`) | 있음 | 없음 |
| 모델 규격 (`CODE_SPECS`) | 있음 | 없음 |
| MWPM 기준선 | 있음 | 없음 |
| 배치 (`embedding_for`, 자동 배치) | 있음 | 회로 생성기 안의 번호 규칙만 있음 |
| `run_hw.py` 제출, 분석 | 있음 | 없음 |
- (3,5), (5,3)에 있는 것은 코드 정의와 하드웨어 회로 생성기 두 종류, 그리고 그 회로의 check_values와 무노이즈 검증 스크립트다. 데이터셋, 학습, MWPM, 게이트, 제출 경로는 없다.
- 하드웨어 회로 생성기는 coupling map 파일(`coupling_ibm_boston.json` 등)이 있어야 생성된다. rung ancilla의 물리 번호를 coupling map에서 찾기 때문이다. `fetch_coupling.fetch`로 받은 ibm_pittsburgh와 ibm_boston의 coupling map(둘 다 156큐빗, 방향 포함 352 edge)으로 두 생성기를 실행해 보았고 (3,5), (5,3) 모두 생성됐다 (2026-11-17).
- `fetch_coupling.py`의 설명문은 `heavyhex_depthopt`라는 모듈 이름을 적고 있는데 실제 파일은 `heavyhex_depth_opt.py`다.
# 하드웨어에 들어가는지
---
- 65큐빗이고 data 물리 번호가 최대 151이다. 156큐빗 Heron 장비의 번호 범위 안이다.
- ibm_pittsburgh와 ibm_boston의 coupling map에서 생성기가 필요한 연결을 assert로 확인하며 회로를 만들었다. 2 cycle 회로를 만들어 2로 나눈 값은 아래와 같고 (3,5)와 (5,3)이 같다.
| 생성기 | 큐빗 | depth | CX |
|---|---|---|---|
| `GeneralDiamondCircuit` | 65 | 48 | 247 |
| `DepthOptDiamond` | 65 | 26 | 223 |
- 캘리브레이션을 반영한 배치(불량 큐빗 회피)는 확인하지 않았다. 생성기는 번호 규칙으로 정한 한 위치만 쓴다.
# 범위 선택지
---
2026-11-17에 석현님이 (3,5), (5,3) 구현을 보류하기로 정했다. 아래는 다시 시작할 때 고를 선택지다.
계획서는 최소 목표를 Stim 데이터셋 생성과 학습, MWPM 기준선, 게이트로 적었고 하드웨어 제출은 확인 뒤 결정으로 남겼다.
## 가. 시뮬레이션만
---
- `sol_35.json`, `sol_53.json`에서 추상 Stim 생성기, 텐서 격자, detector 그래프, 모델 규격, MWPM을 만든다. 게이트는 Stim 수준(무노이즈 결정성, 단일 오류 서명이 지지와 일치)까지만 한다.
- 장점은 coupling map 없이 지금 할 수 있고 학습 격자에 바로 들어간다는 것이다.
- 단점은 하드웨어 회로와의 등가 검증이 없어 (3,3)과 같은 수준의 계약이 아니라는 것이다. 추상 회로의 cycle 안 순서를 제가 정해야 한다 ((3,3)처럼 Z 라운드 뒤 X 라운드로 두는 안).
## 나. 시뮬레이션과 하드웨어 경로
---
- 가에 더해 `DepthOptDiamond`를 하드웨어 회로로 붙이고, Stim과 Aer의 등가 게이트, 65큐빗 배치, QPU 프로파일 경로, `run_hw.py` 제출과 분석을 만든다.
- 장점은 (3,3)과 같은 계약이 된다는 것이다.
- 단점은 양이 가의 두세 배라는 것이다. 65큐빗 무노이즈 Aer 검증은 Clifford 회로라 가능하지만 실제로 돌려 확인해야 한다.
## 다. (3,5), (5,3) 제외
---
- 범위에서 빼고 (3,3)만 둔다. 확정 결정 9와 어긋나므로 계획서 변경이 필요하다.
## 함께 정할 것
---
- (3,3)을 지금의 8 Z, 8 X 경로로 그대로 두고 (3,5), (5,3)만 일반해를 쓸지, 아니면 세 경우를 같은 규칙의 해로 맞출지. 후자는 (3,3)의 데이터셋과 체크포인트를 다시 만드는 일이 된다.
