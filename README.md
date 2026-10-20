# decap_loop: 목표 임피던스를 최소 decap 개수로 맞추는 배치 정책

PDN 판 위에 decap을 놓아 측정 지점의 |Z(f)|가 모든 주파수에서 목표 임피던스 이하가 되게 하되, decap 개수를 최소로 하는 문제입니다.
작은 판에서 학습한 정책이 크거나 모양이 불규칙한 판에서도 동작하는지 검증합니다.

설계는 검증 보고서([reports/Decap 이중 루프 제안 검증.md](reports/Decap%20이중%20루프%20제안%20검증.md))의 수정안을 따릅니다.
- **회로 계산은 고전 솔버가 합니다.** 주파수마다 sparse LU로 한 번 분해하고, decap을 넣고 뺄 때는 rank-1 공식으로 갱신합니다. 전체 Z 행렬은 만들지 않습니다.
- **학습은 결정에만 씁니다.** 정책은 고전 솔버가 계산한 정확한 물리 특징을 입력으로 받아 add / remove / 종류 교체 / stop을 고릅니다.
- 원래 제안의 "회로 풀이를 흉내 내는 안쪽 루프"는 뺐습니다. Jacobi식 반복이 이 회로에서 수렴하지 않기 때문입니다(보고서 3절).

## 구성

| 파일 | 내용 |
|---|---|
| `decap_loop/pdn.py` | PDN 생성기. 단위 셀 RLGC 격자 + VRM/패키지 경로. 크기·모양(rect, L, holes), keep-out, 측정 지점 1~3개, 목표 임피던스(평탄형, 계단형)를 무작위로 정함. `ZBank`가 Z0의 필요한 열과 대각만 보관 |
| `decap_loop/sim.py` | 환경. Woodbury 공식으로 decap이 놓인 회로를 계산하고, 모든 후보 행동의 효과를 rank-1 식으로 한 번에 평가. 행동 규칙과 정책 입력 특징 |
| `decap_loop/baselines.py` | greedy, greedy+prune, random+prune, 국소 탐색(LS), 개수 최소화 GA. 모두 같은 환경을 씀 |
| `decap_loop/model.py` | 판 크기와 무관한 격자 GNN 정책. 이웃 메시지 전달 + 전역 평균 + (선택) 전역 attention |
| `decap_loop/rollout.py` | 정책 실행. 학습 때는 기울기 없이 실행하고 나중에 묶어서 log-확률을 다시 계산 |
| `train.py` | REINFORCE(같은 문제의 M개 샘플 평균을 기준선으로 사용) + 최고 해 모방 + 라그랑주 실패 벌점 |
| `evaluate.py` | 크기·모양별 비교. 성공률, 개수, greedy+prune 대비 비율, 최고 기록 대비 격차, 시간 |
| `sanity_check.py` | 시뮬레이터와 환경의 정확성 검사 |
| `run_all.sh` | 본 실험, ablation, 평가 전체 실행 |

## 설치와 실행

```bash
pip install -r requirements.txt
python sanity_check.py
bash run_all.sh
```

`python sanity_check.py --big`은 50×50, 100×100 판의 준비 시간과 greedy+prune 시간도 잽니다.

## 문제 정의

**회로**: 전원면 한 쌍을 셀 격자로 나눕니다.
- 각 셀은 판 정전용량 C와 유전 손실 G를 갖습니다.
- 이웃 셀은 R + L로 연결됩니다.
- 전원 공급 셀(1~4개)은 R_vrm + L_vrm을 거쳐 VRM에 연결됩니다.

**목표 임피던스** Zt(f) = L · s(f)입니다.
- s(f)는 평탄형이면 1이고, 계단형이면 무작위 주파수에서 2~10배로 올라갑니다.
- L은 두 기준 사이에서 정합니다. 위쪽은 "decap이 없어도 만족하는 수준", 아래쪽은 "놓을 수 있는 모든 셀에 한 종류를 채우면 만족하는 수준"입니다. 그래서 만든 문제는 모두 풀 수 있습니다.
- `--difficulty`(기본 0.02~0.2)가 작을수록 아래쪽에 가까워져 어려워집니다.

**행동 규칙**
- 목표를 아직 못 맞췄으면: add(셀, 종류)만 할 수 있습니다.
- 목표를 맞췄으면: remove 또는 종류 교체를 할 수 있습니다. 단, 그 뒤에도 목표가 유지될 때만 가능합니다. 바꾸기 전 종류는 그 셀에서 다시 쓸 수 없습니다(tabu). stop도 할 수 있습니다.
- 할 수 있는 행동이 없거나 `max_steps`에 도달하면 끝납니다.

**보상**: R = −개수 / N_ref − λ·(실패 + 위반량)
- N_ref는 같은 문제에서 greedy+prune이 쓴 개수입니다. 판 크기가 달라도 같은 기준이 됩니다.
- 보고서 실험에서 "물리 하한"은 너무 느슨해서 쓰지 않았습니다(greedy가 하한의 1~9배).
- λ는 실패율이 `--fail_target`(2%)보다 높으면 커지고 낮으면 작아집니다(라그랑주 방식).

## 정책 입력 (좌표 없음)

모든 값은 목표 임피던스에 대한 log 비율이라 판 크기에 묶이지 않습니다.
- 셀마다: 자기 임피던스, 측정 지점까지의 전달 임피던스(주파수 8개), 행동별로 "하면 위반이 얼마나 줄어드는지"와 "하고 나서 최악 여유"
- 전체: 현재 위반 곡선, f / f1(판의 첫 공진 주파수 대비 주파수), 목표 만족 여부, 남은 단계, 놓은 비율
- 플래그: 유효 셀, keep-out, 측정 지점, 전원 공급 셀, 놓인 종류

`--feat coord`는 비교용입니다. 물리 특징 대신 좌표와 측정 지점까지의 오프셋을 씁니다(Park / DevFormer 방식).

## 가설과 run 대응 (보고서의 수정된 가설)

| 가설 | 비교 |
|---|---|
| H1′ 물리 특징 정책이 좌표 정책보다 큰 판에 잘 일반화된다 | `gnn_phys` vs `gnn_coord`, 학습 8~12칸 → 평가 15/20/30/50칸 |
| H2′ 메시지 전달 깊이 고정이 큰 판에서 문제인가 | `gnn_k3` vs `gnn_phys`(6층) vs `gnn_k12`, `gnn_glob` |
| H3′ add/remove/stop이 greedy+prune 이상, LS·GA에 근접 | `gnn_phys` vs `gnn_noprune`, 기준선 greedy / greedy+prune / LS / GA |
| H4′ 같은 계산기를 쓸 때 시간이 짧다 | `summary.csv`의 `sec_per_inst` (판 준비 시간 포함) |
| 학습 방법 | `gnn_noimit`(모방 없음), `gnn_lsteach`(LS를 선생으로) |

## 지표

- **success**: 목표를 만족한 비율
- **count**: 만족한 문제에서 쓴 decap 개수
- **/g+p**: greedy+prune 대비 개수 비율 (1보다 작으면 더 적게 씀)
- **gap%**: 그 문제의 모든 방법 중 최소 개수 대비 초과 비율
- **sec**: 문제 하나당 시간. sparse LU 분해와 diag(Z0) 계산 같은 판 준비 시간을 모든 방법에 포함

## 확인한 결과 (M4 Pro CPU, `sanity_check.py --big`)

| 항목 | 결과 |
|---|---|
| Woodbury 계산 vs decap을 Y에 직접 넣고 푼 값 | 상대 오차 1e-14 (dense, sparse 둘 다) |
| rank-1 행동 효과(add / 교체 / remove) vs 다시 계산 | 오차 1e-15 |
| 94단계 동안 누적 갱신한 상태 vs 처음부터 계산 | 오차 4e-10 (기본값은 64단계마다 다시 계산) |
| greedy+prune 해를 직접 풀어서 확인 | 8/8 목표 만족 |
| 50×50 판 준비 / greedy+prune | 11.4초 / 0.2초 |
| 100×100 판 준비 / greedy+prune | 217초 / 0.5초 (준비 시간 대부분이 diag(Z0)) |

10×10 문제 8개에서 greedy+prune과 국소 탐색의 차이가 큰 경우가 있습니다(31 → 3, 25 → 12). H3′를 시험할 여지가 있다는 뜻입니다.

학습 확인(작은 설정 d=64, 3층, B=8, M=4, CPU 600 iter, 약 25분)에서 검증 세트 개수 비율(정책 / greedy+prune)은 이렇게 내려갔습니다. 본 실험이 아니라 학습 루프가 동작하는지 본 것입니다.

| iter | 10×10 | 16×16 |
|---|---|---|
| 25 | 7.61 | 9.48 |
| 100 | 3.27 | 3.64 |
| 300 | 1.51 | 1.61 |
| 500 | 1.22 | 1.35 |

CPU에서는 iter당 약 2~3초입니다. 기본 설정(d=128, 6층, B=16, M=8)은 GPU를 권장합니다.

## MoTT (공진 모드 토큰 정책) — 2026-10-20 추가

설계 근거: `reports/최종 모델 설계.md`, `reports/모달 토큰 트랜스포머 제안.md`.

| 파일 | 내용 |
|---|---|
| `decap_loop/verify.py` | 정밀 검사: decap을 Y에 직접 넣고 700점 + 피크 정밀화로 최악 \|Z\|/Zt 계산. `cutting_plane`: 위반 주파수 주변에 점을 추가하며 다시 푸는 적응형 격자 |
| `decap_loop/pdn.py` | 문제마다 자기 주파수 격자(`batch.freqs[b]`), `add_freqs`, 목표 임피던스를 정밀 검사 기준으로 설정 |
| `decap_loop/sim.py` | `feat_mode="mott"`: 상위 8개 공진 피크(모드 토큰), 칸–모드 엣지(결합도 g, \|Z_ii\|/Zt, 종류별 놓은 뒤 \|Z_pp\|/Zt), decap 토큰(SRF 등), 칸 특징(정확한 1단계 효과 = 물리 사전 점수) |
| `decap_loop/model.py` | `MoTT`: 모드 self-attention, 칸↔모드·decap↔모드 cross-attention(칸끼리 attention 없음). 출력 = 학습된 가중치 × 물리 사전 점수 + 학습 보정. `GridPolicy(prior=True)`: 같은 사전 점수를 붙인 대조군 |
| `decap_loop/bank.py` | 문제 은행 저장/불러오기 (격자까지 저장) |
| `decap_loop/baselines.py` | `modal_heuristic`: 가장 큰 위반 피크에 결합도가 가장 큰 칸 + SRF가 가장 가까운 종류 |
| `build_pool.py` | 인증된 문제 은행: 학습 PCB 8/10/12칸, 시험 PCB 10/16칸, 인터포저형 10/16칸(난이도 0.2~0.5) |
| `train.py` | `--pool` 문제 은행 사용, `--model mott`, `--prior`, `--certify`(매 반복 최고 샘플 정밀 검사, 위반이면 실패 처리 + 격자 보강), `--hours` |
| `eval_mott.py` | 모든 방법을 cutting-plane 안에서 실행하고 정밀 검사로 성공 판정 |
| `smoke_mott.py` | 작은 판으로 전체 과정 스모크 테스트 |
| `run_mott.sh` | GPU 2대: 스모크 테스트 → 문제 은행 → 두 모델 학습 → 평가 |
| `h0_check.py`, `baseline_table.py`, `proxy_h1.py` | H0, 기준선 표, 학습 없는 대리 실험 |

스모크 테스트 (작은 6칸 판으로 전체 과정 확인, CPU 약 20초):

```bash
python smoke_mott.py
```

문제 은행 → Woodbury와 직접 풀이 비교 → 두 모델 롤아웃·역전파 → `train.py` 2회 → `eval_mott.py` 순서로 확인하고, 하나라도 실패하면 멈춘다. `run_mott.sh`가 본 실행 전에 GPU에서 먼저 돌린다.

실행 (GPU 2대):

```bash
HOURS=6 bash run_mott.sh
```

CPU 확인용 (작은 설정):

```bash
python train.py --pool runs/pool --model mott --d 64 --layers 3 --heads 4 --B 8 --M 8 --iters 50 --val_every 25 --out runs/mott_cpu
```

주의:
- 정밀 검사 없이 격자로만 보상을 주면 정책이 격자 사이 공진 피크로 decap을 줄이는 법을 배운다. 학습 전 정책의 정밀 검사 실패율이 88%였다. 그래서 `--certify`를 기본으로 켠다.
- dense bank는 Z0 전체를 주파수마다 저장한다. 격자가 커지면 메모리가 B·F·n²에 비례한다. 문제 은행은 complex64로 올리고, 12칸 묶음 하나가 약 0.6GB다. 메모리가 부족하면 `--pool_glob`로 묶음 수를 줄인다.
- GPU에서 아직 실행해 보지 않았다. 정밀 검사(`verify.py`)는 CPU에서 돈다.

## 꼭 알아야 할 한계

1. **물리 값은 임시입니다.** `PDNConfig`의 셀 크기, 유전체 두께, VRM 경로, decap 라이브러리를 실제 보드 값으로 바꿔야 합니다.
2. **diag(Z0)이 큰 판의 병목입니다.** 셀마다 한 번씩 풀어야 해서 100×100에서 수 분이 걸립니다. 선택적 역행렬(selected inversion)을 쓰면 줄일 수 있지만 아직 넣지 않았습니다.
3. **목표가 쉬우면 greedy+prune이 거의 최적입니다.** difficulty가 낮은(어려운) 문제에서 greedy와 LS·GA의 차이가 커집니다. 실험 전에 크기별 기준선 표부터 확인하세요.
4. **PAGA는 포함하지 않았습니다.** 공개 코드(github.com/SoFoolish250/Physics-Assisted-Genetic-Algorithm-PAGA-for-Decap-Optimization)를 이 환경의 계산기에 연결해 비교해야 합니다.
5. 독립 검증(ngspice AC 해석, RL4CO 10×10 데이터 대조)은 아직 하지 않았습니다.
