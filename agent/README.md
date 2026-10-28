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
- 모델 바꾸기: `MODEL=claude-sonnet-5-5 bash agent/loop.sh`

## 모델 선택 (loop.sh가 자동으로)
| 상황 | 모델 |
|---|---|
| 보통 라운드 | `claude-opus-5-5` (`MODEL` 환경변수로 변경) |
| 하루의 첫 라운드 (DAILY.md 작성 + 계획) | `claude-fable-5-1` |
| 직전 라운드가 `STUCK:`을 남겼거나 비정상 종료 | `claude-fable-5-1` 한 번 |
| 사람이 강제 | `echo claude-fable-5-1 > agent/MODEL` (지우면 자동으로 복귀) |

## 질문 없이 도는가
- `claude -p`는 무인 모드라 질문할 수 없다. 프롬프트에 "질문 금지, 보수적 선택 후 `결정:` 기록" 규칙을 넣었다.
- 허용 도구(`--allowedTools`) 밖의 명령은 거부되고 에이전트는 다른 방법을 찾는다. 환경 파일 수정·네트워크·삭제는 허용 목록에 없다.
- 사람의 개입 통로는 `agent/QUEUE.md`와 `agent/STOP` 둘뿐이다.
