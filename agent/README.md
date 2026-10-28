# 서버에서 에이전트 무인 실행

## 1회 준비 (서버)
```bash
git clone https://github.com/jeehoo0507/R2E.git && cd R2E
git remote rename origin r2e                      # loop.sh는 원격 이름 r2e를 쓴다
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python sanity_check.py                  # 통과해야 시작
gh auth login                                     # push 권한 (또는 git credential 설정)
claude                                            # 한 번 열어 로그인만 하고 종료
chmod +x agent/loop.sh
```

## 시작 (터미널 닫아도 계속 돈다)
```bash
tmux new -d -s agent 'bash agent/loop.sh'
```

## 보기 / 끄기
```bash
tmux attach -t agent          # 화면 보기 (빠져나오기: Ctrl-b 다음 d)
tail -f agent/logs/loop.log   # 라운드 로그
touch agent/STOP              # 현재 라운드가 끝나면 종료
tmux kill-session -t agent    # 즉시 종료
```

## 사람이 지시하기
`agent/QUEUE.md`에 한 줄씩 적고 push하면 다음 라운드가 최우선으로 읽는다. 예:
```
- 2단 학습을 멈추고 1단으로 세 종류 평가부터 끝내라
- 인터포저형 decap 라이브러리를 (50p,2p,0.1) 추가해 다시 평가
```

## 동작 방식
- 라운드마다 새 `claude -p` 세션. 기억은 파일로만 이어진다: `PROGRESS.md`, `DAILY.md`, `RESULTS.md`, `runs/`, git log.
- 라운드 끝에 자동으로 commit + push (에이전트가 잊어도 `loop.sh`가 한다).
- `END_DATE`(기본 2026-11-11) 이후 또는 `agent/STOP`이 있으면 멈춘다.
- 모델·턴 수 바꾸기: `MODEL=claude-sonnet-5-5 MAX_TURNS=200 bash agent/loop.sh`
