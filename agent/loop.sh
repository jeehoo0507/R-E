#!/usr/bin/env bash
# Unattended agent loop. Run inside tmux (see agent/README.md) so it survives a closed terminal.
#
#   bash agent/loop.sh            # runs until agent/STOP exists or $END_DATE passes
#
# Each round starts a fresh `claude -p` session with a fixed tool allowlist (repo files, python,
# git, tmux-free shell basics). Continuity lives in files, not in the chat: PROGRESS.md, DAILY.md,
# RESULTS.md, runs/, git log. Every round must leave those updated.
set -u
cd "$(dirname "$0")/.."
END_DATE="${END_DATE:-2026-11-11}"     # no new rounds on/after this date (experiments freeze 11/10)
MAX_TURNS="${MAX_TURNS:-400}"          # tool calls per round
MODEL_DEFAULT="${MODEL:-claude-opus-5-5}"   # routine rounds
MODEL_HARD="${MODEL_HARD:-claude-fable-5-1}" # planning / stuck rounds (see pick_model)
pick_model() {
  # 1) human override: agent/MODEL holds a model id
  if [ -s agent/MODEL ]; then cat agent/MODEL; return; fi
  # 2) first round of a calendar day -> hard model (writes DAILY.md, plans the day)
  if [ ! -f agent/.last_day ] || [ "$(cat agent/.last_day)" != "$(date +%F)" ]; then echo "$MODEL_HARD"; return; fi
  # 3) previous round flagged STUCK in PROGRESS.md, or exited non-zero -> hard model once
  if grep -q '^STUCK' PROGRESS.md 2>/dev/null || [ "${last_rc:-0}" != "0" ]; then echo "$MODEL_HARD"; return; fi
  echo "$MODEL_DEFAULT"
}
mkdir -p agent/logs
ROUND_PROMPT='이 세션은 무인 실행이다. 사람에게 질문하거나 확인을 요청하지 마라. 결정이 필요하면 가장 보수적인 선택(기존 설정 유지, 작은 실험 먼저)을 하고 PROGRESS.md에 "결정:"으로 이유와 함께 기록하라. 같은 문제로 두 번 막히면 PROGRESS.md 첫 줄에 "STUCK: <문제>"를 쓰고 다른 작업으로 넘어가라. 먼저 git log -20, PROGRESS.md, DAILY.md(있으면), RESULTS.md(있으면), PROPOSALS.md(있으면)를 읽고 어디까지 왔는지 파악하라. 그다음 agent/AGENT_PROMPT.md의 목적·규칙·사다리에 따라 이어서 작업하라. 이번 라운드가 끝날 때 반드시: PROGRESS.md 갱신, 끝난 실험은 RESULTS.md 반영, git commit, git push r2e main. 24시간마다 DAILY.md를 새로 쓴다. 사람이 agent/QUEUE.md에 지시를 적어 두었으면 그것을 최우선으로 따른다.'
# Only what the research loop needs. Anything else prompts (and, headless, is refused).
ALLOWED='Read,Edit,Write,Glob,Grep,Bash(python:*),Bash(python3:*),Bash(.venv/bin/python:*),Bash(.venv/bin/pip:*),Bash(git:*),Bash(ls:*),Bash(cat:*),Bash(head:*),Bash(tail:*),Bash(wc:*),Bash(mkdir:*),Bash(nvidia-smi:*),Bash(date:*)'

round=0
while true; do
  if [ -f agent/STOP ]; then echo "[loop] STOP file found, exiting"; break; fi
  if [[ ! "$(date +%F)" < "$END_DATE" ]]; then echo "[loop] reached $END_DATE, exiting"; break; fi
  round=$((round + 1))
  ts="$(date +%Y%m%d_%H%M%S)"
  log="agent/logs/round_${ts}.log"
  MODEL="$(pick_model)"
  echo "[loop] round $round start $ts model=$MODEL" | tee -a agent/logs/loop.log
  date +%F > agent/.last_day
  git pull --rebase r2e main >>"$log" 2>&1 || true
  claude -p "$ROUND_PROMPT" \
    --model "$MODEL" \
    --max-turns "$MAX_TURNS" \
    --permission-mode acceptEdits \
    --allowedTools "$ALLOWED" \
    --output-format text >>"$log" 2>&1
  rc=$?; last_rc=$rc
  echo "[loop] round $round end rc=$rc $(date +%Y%m%d_%H%M%S)" | tee -a agent/logs/loop.log
  # safety net: push whatever the round left behind, even if it forgot
  git add -A . >>"$log" 2>&1
  git diff --cached --quiet || git commit -q -m "agent: round $round autosave ($ts)" >>"$log" 2>&1
  git push r2e main >>"$log" 2>&1 || true
  sleep 120
done
