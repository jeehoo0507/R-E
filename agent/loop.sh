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
MODEL="${MODEL:-claude-opus-5-5}"
mkdir -p agent/logs
ROUND_PROMPT='먼저 git log -20, PROGRESS.md, DAILY.md(있으면), RESULTS.md(있으면), PROPOSALS.md(있으면)를 읽고 어디까지 왔는지 파악하라. 그다음 agent/AGENT_PROMPT.md의 목적·규칙·사다리에 따라 이어서 작업하라. 이번 라운드가 끝날 때 반드시: PROGRESS.md 갱신, 끝난 실험은 RESULTS.md 반영, git commit, git push r2e main. 24시간마다 DAILY.md를 새로 쓴다. 사람이 agent/QUEUE.md에 지시를 적어 두었으면 그것을 최우선으로 따른다.'
# Only what the research loop needs. Anything else prompts (and, headless, is refused).
ALLOWED='Read,Edit,Write,Glob,Grep,Bash(python:*),Bash(python3:*),Bash(.venv/bin/python:*),Bash(.venv/bin/pip:*),Bash(git:*),Bash(ls:*),Bash(cat:*),Bash(head:*),Bash(tail:*),Bash(wc:*),Bash(mkdir:*),Bash(nvidia-smi:*),Bash(date:*)'

round=0
while true; do
  if [ -f agent/STOP ]; then echo "[loop] STOP file found, exiting"; break; fi
  if [[ ! "$(date +%F)" < "$END_DATE" ]]; then echo "[loop] reached $END_DATE, exiting"; break; fi
  round=$((round + 1))
  ts="$(date +%Y%m%d_%H%M%S)"
  log="agent/logs/round_${ts}.log"
  echo "[loop] round $round start $ts" | tee -a agent/logs/loop.log
  git pull --rebase r2e main >>"$log" 2>&1 || true
  claude -p "$ROUND_PROMPT" \
    --model "$MODEL" \
    --max-turns "$MAX_TURNS" \
    --permission-mode acceptEdits \
    --allowedTools "$ALLOWED" \
    --output-format text >>"$log" 2>&1
  rc=$?
  echo "[loop] round $round end rc=$rc $(date +%Y%m%d_%H%M%S)" | tee -a agent/logs/loop.log
  # safety net: push whatever the round left behind, even if it forgot
  git add -A . >>"$log" 2>&1
  git diff --cached --quiet || git commit -q -m "agent: round $round autosave ($ts)" >>"$log" 2>&1
  git push r2e main >>"$log" 2>&1 || true
  sleep 120
done
