# 서버 에이전트 프롬프트 (github.com/jeehoo0507/R2E 에서 실행)

사전 조건: 로컬 `R&E` 폴더가 이 저장소 main에 push되어 있어야 한다. 서버에서 `git clone` 후 저장소 루트에서 Claude Code를 열고 아래를 첫 메시지로 붙여 넣는다.

---

너는 이 저장소에서 **R&E 연구보고서(제출 11/16, 실험·그림 완료 11/10)의 핵심 결과를 만드는 연구 에이전트**다. 매일 사람이 아침에 `DAILY.md`를 읽고 방향을 정하고, 너는 그 사이 밤낮으로 실험을 돌린다. 오늘은 첫날이고 24시간 동안 자유롭게 탐색해도 된다. 단, 아래 목적과 조건은 바꾸지 마라.

## 목적: 보고서에 들어갈 그림 3장을 만든다

**그림 1 (메인):** PCB형·패키지형·인터포저형 **세 종류의 판을 섞어 학습한 모델 하나**가, 세 종류 모두에서 목표 임피던스를 만족하는 decap 배치를 낸다. 판 위 배치 + 임피던스 곡선이 목표 아래로 내려가는 모습.
**그림 2 (와우 숫자):** 큰 판(150~400포트)에서 우리 모델 vs greedy+prune vs LS (가능하면 GA): **시간**과 **decap 개수**. "수백 배 빠르고 개수는 비슷"이 목표이되, 결과가 어떻든 숫자 그대로 보고한다.
**그림 3 (엄밀함):** 사다리 ablation. 같은 학습·같은 데이터에서 모델만 한 단씩 쌓는다.
- 0단: 기존 정책 + 좌표 입력 (`--feat coord`)
- 1단: + 공진 참여도·정확한 1수 효과 포트 특징 (새 `--feat modal`)
- 2단: + 모드 토큰 + 포트↔모드 cross-attention (멀티 토큰, `reports/최종 모델 설계.md` §3)
- 3단: + 루프 (2단 블록 가중치 공유, 학습 때 R~U{2..8}, 테스트 때 R 2→16)
각 단은 **독립적으로 보고 가능**해야 한다. 2단이 안 돌면 1단까지로 그림 1·2를 그린다.

**보너스 (주장 아님, 고찰용):** 학습에서 뺀 네 번째 종류(예: 칩형 파라미터)에 zero-shot. 되면 좋고, 안 되면 "안 됨 + 어느 특징이 전이됐나"로 보고.

## 마감
- **모든 실험과 그림 3장은 11/10까지 끝낸다.** 실험 동결 11/8, 11/9~10은 평가·그림·결과표만. 11/10 이후 새 실험 금지.
- 하루 단위로 "오늘 끝까지 가면 보고서에 들어갈 수 있는 것"을 우선한다. 큰 실험 하나보다 작은 비교 여러 개.

## 참고 논문 (아키텍처와 학습의 출발점. 베끼지 말고 "무엇을 가져오는지"를 명확히)
| 논문 | 가져올 것 | 어디에 |
|---|---|---|
| Park et al., "Transformer Network-Based RL Method for PDN Optimization of HBM", IEEE TMTT 2022 (arXiv 2203.15722) | 포인터식 디코더(후보마다 점수), 문맥 임베딩, REINFORCE + 롤아웃 기준선. **좌표 입력 = 0단 기준선** | 0단, 디코더 출력 |
| Kool, van Hoof, Welling, "Attention, Learn to Solve Routing Problems!", ICLR 2019 / Kwon et al., "POMO", NeurIPS 2020 | 인코더-디코더 정책, 같은 문제 여러 번 풀어 평균을 기준선으로 (POMO). `train.py`가 이미 이 방식 | 학습 |
| Manoharan, Wang, Hwang, "RL for Decap Optimization in Postlayout PDNs Using Transformer Architecture", IEEE TSIPI 2026 (박사논문 Paper I, Scholars' Mine 3459) | Z 입력, 매 단계 Schur/Woodbury 갱신, decap 토큰(ESL·ESR·C·크기), 주파수 한 점 loop-L. **고정 주파수 입력 = `--feat phys` 기준선** | 기준선, decap 토큰 |
| Kim et al., "DevFormer: A Symmetric Transformer for Context-Aware Device Placement", ICML 2023 | 행동 순열 대칭성, 오프라인 전문가 데이터 학습 | 모방 학습 설계 |
| Kwon et al., "Matrix Encoding Networks for Neural Combinatorial Optimization (MatNet)", NeurIPS 2021 | **두 집합 사이의 행렬을 cross-attention 편향으로** 넣는 구조. 우리의 모드×포트 참여도 행렬이 그 자리 | **2단 핵심** |
| Park & Sengupta, "Neural Modal Decomposition: Architectural Priors from Observables", arXiv 2609.14402 | 공유 pole + 포트별 residue 표현, 포트 수 변화에 불변 | 모드 토큰 정의 |
| Dehghani et al., "Universal Transformers", ICLR 2019 / Fan et al., "Looped Transformers for Length Generalization", ICLR 2025 / Bansal et al., "End-to-end Algorithm Synthesis with Recurrent Networks (Deep Thinking)", NeurIPS 2022 | 가중치 공유 반복, 매 반복 입력 재주입(recall), 반복 횟수 랜덤 학습(progressive), 테스트 때 반복 늘리기 | 3단 |
| Lee et al., "Set Transformer", ICML 2019 | 순서·개수 무관 집합 입력 | 토큰 집합 처리 |
| Anderson, Smith, Chun, US 6789241 (Sun, 2004) | 공진 피크 위치에 SRF 맞는 decap을 놓고 재스윕하는 **비학습 규칙** → 모달 휴리스틱 기준선 | 기준선 |
| Gustavsen & Semlyen, "Rational approximation of frequency domain responses by Vector Fitting", 1999 / Nakatsukasa, Sète, Trefethen, "The AAA algorithm", 2018 | pole 추출 (`scipy.interpolate.AAA` 사용) | 모달 추출 |

서지가 불확실하면 `PROPOSALS.md`에 "확인 필요"로 적고 보고서용 인용은 사람이 확정한다.

## 먼저 읽을 것 (순서대로, 40분 이내)
1. `README.md`
2. `reports/방향 확정.md` — 범위, 사다리, 일정
3. `reports/최종 모델 설계.md` §2~§3 — 모달 추출과 토큰 설계
4. `research_notes/토큰 데이터 분석/결과.md` — 학습 없는 분석: 같은 종류 안에서는 공진+멀티 토큰이 AUC 0.93, 종류를 넘으면 참여비만 전이, 좌표·loop L은 뒤집힘. **그래서 섞어서 학습한다.**
5. `research_notes/토큰 데이터 분석/token_analysis2.py` — 공진 추출·참여비·1수 효과 계산 코드 (재사용)
6. `train.py`, `decap_loop/model.py`, `decap_loop/verify.py`, `sanity_check.py`

## 첫날 권장 순서 (바꿔도 되지만 이유를 DAILY.md에 적어라)
1. 환경 확인: `python sanity_check.py` 통과, GPU torch 동작, `git status` 기록
2. **종류 3개의 PDNConfig 정의** (PCB형=기본, 패키지형, 인터포저형: 셀 크기·유전체 두께·손실·VRM·decap 라이브러리·검사 대역을 종류답게). `research_notes/토큰 데이터 분석/token_analysis2.py`의 `TECH`를 출발점으로. 평가 세트 고정 (종류별 × 크기별 각 ≥50문제, 시드 고정, `data/eval/`에 저장)
3. 기준선 E0: greedy+prune, LS를 평가 세트 전체에 실행 (CPU 병렬) → 결과표
4. 1단 `--feat modal` 구현 (포트 특징 채널 추가, 모델 구조는 그대로) → 섞어 학습 짧게(≤2h) → 세 종류 평가
5. 0단 같은 조건 → 1단과 비교
6. 2단 멀티 토큰 모델 구현 (새 파일 `decap_loop/mott.py`) → 작은 판 sanity → 섞어 학습 ≤2h
7. 시간이 남으면 3단, 큰 판 시간 측정(그림 2), zero-shot 보너스

## 반드시 지킬 것
1. **환경 파일 수정 금지:** `decap_loop/pdn.py`, `sim.py`, `verify.py`, `bank.py`. 종류별 설정은 새 파일(`decap_loop/techs.py`)로.
2. **성공 판정은 `verify.py`의 700점 정확 검사만.** 32점 격자 성공률은 보고서용 숫자가 아니다 (이미 "주장 1.00 / 실제 0.00"이 확인됨).
3. 모든 실행: `runs/<날짜>/<이름>/`에 `config.json`(인자·seed·git hash·종류 설정), `log.txt`, `metrics.json`. `RESULTS.md`의 표는 **스크립트**(`scripts/collect_results.py`, 네가 만들어라)로 로그에서 생성. 손으로 숫자를 쓰지 마라. **추정·예시·"대략" 숫자 금지.**
4. 학습 1회 ≤ 2시간 (첫날). 넘으면 중단하고 그 시점 체크포인트로 평가.
5. 한 비교 안에서 **한 번에 하나만** 바꿔라. 모델 폭·학습 iter·데이터·보상은 사다리 전체에서 동일.
6. 기준선(greedy+prune, LS)보다 나쁜 결과도 그대로 기록한다. 좋은 결과를 만들려고 평가 세트·목표·검사를 바꾸는 것이 가장 나쁜 행동이다.
7. 막히면 30분 이상 매달리지 말고 `DAILY.md`에 기록하고 넘어가라. 환경 파일을 고쳐야만 풀리는 문제는 `PROPOSALS.md`에 적고 멈춘다.
8. **GitHub 보고 (사람이 원격에서 지켜본다):**
   - 실험 하나가 끝날 때마다 `git commit` + **`git push r2e main`** (하루 끝까지 미루지 마라). 커밋 메시지는 영어, 무엇을 왜.
   - **4시간마다** `PROGRESS.md`를 갱신하고 push: 지금 돌고 있는 것, 끝난 것(숫자 + 로그 경로), 막힌 것, 다음 4시간 계획. 10줄 이내, 맨 위에 갱신 시각.
   - 학습이 2시간 넘게 도는 동안에도 중간 체크포인트 평가 결과를 `PROGRESS.md`에 적고 push.
   - 큰 파일(체크포인트 `*.pt`, `runs/`)은 push하지 않는다 (.gitignore). 로그는 `metrics.json`과 `RESULTS.md`로 올린다.
   - push가 실패하면 `git pull --rebase r2e main` 후 재시도. 충돌이 나면 사람 파일(`reports/`, `EXPERIMENTS.md`)은 원격 것을 우선.

## 24시간 뒤 제출물
`DAILY.md` (맨 위에 10줄 요약) + `RESULTS.md` (자동 표) + `PROGRESS.md` 이력. 모두 push된 상태여야 한다.

```
## 요약 (10줄 이내)
- 사다리 몇 단까지 동작 확인 (각 단: 세 종류 평가 성공률·개수/greedy+prune, 로그 경로)
- 그림 1·2·3 중 지금 그릴 수 있는 것
- 가장 큰 발견 2개, 효과 없던 것 2개
## 결과표  (RESULTS.md 링크)
## 발견한 문제 (버그·불안정·메모리·시간)
## 내일 추천 큐 (5개, 각 1줄, 이유)
## 하지 말아야 할 것
```

## 성공의 기준
그림 3장을 **정직한 숫자**로 채우는 것. "세 종류 모두 성공률 100%, 개수는 LS와 같음, 속도 300배"가 나오면 최고지만, "2단까지 동작, 인터포저형 성공률 70%, LS보다 개수 20% 많음"도 좋은 결과다. 보고서는 숫자가 아니라 **과정의 엄밀함과 해석의 정확함**으로 평가된다.
