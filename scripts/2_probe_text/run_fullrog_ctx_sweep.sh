#!/usr/bin/env bash
# Overnight OR context sweep on FULLROG (ROG test n=2233).
# Flash 0shot, 7 cells: 3cls at W={10,20,40} + 6cls at W={10,20,30,40}.
# Run from repo root INSIDE an already-open tmux pane:
#   bash scripts/2_probe_text/run_fullrog_ctx_sweep.sh
# Non-zero exit on any cell failure (so the pane stays visible).

set -uo pipefail
cd "$(dirname "$0")/../.."

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="runs/_sweep_logs/fullrog_ctx_${STAMP}"
mkdir -p "$LOGDIR"
MASTER="${LOGDIR}/_master.log"

EVAL_SET="data/subset_rog_test_full.jsonl"
EVAL_TAG="FULLROG"
SCRIPT="scripts/2_probe_text/03_run_openrouter_text_context.py"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$MASTER"; }

log "SWEEP START  log dir: $LOGDIR"
log "Plan: 3cls at W={10,20,40} + 6cls at W={10,20,40}. Flash 0shot. Eval: $EVAL_SET tag=$EVAL_TAG"
log "Est. total cost ~\$15. Est. runtime ~14 h sequential."

FAILED=()

run_cell() {
  local label="$1"; shift
  local logfile="${LOGDIR}/${label}.log"
  log ">> START cell=${label}  args: $*"
  if python "$SCRIPT" "$@" --eval-set "$EVAL_SET" --eval-tag "$EVAL_TAG" \
        > "$logfile" 2>&1; then
    log "<< OK    cell=${label}  log: $logfile"
  else
    log "<< FAIL  cell=${label}  log: $logfile"
    FAILED+=("$label")
  fi
}

# --- 3cls leg: W=10, W=20, W=40 (skipping W=30; already in CTXW30 batch) ---
for W in 10 20 40; do
  run_cell "flash_0shot_3cls_w${W}" \
    --models flash --shots 0shot --schemes 3cls --context-words "$W"
done

# --- 6cls leg: full W={10,20,40} (prompt-sensitivity / fine-grain probe) ---
for W in 10 20 40; do
  run_cell "flash_0shot_6cls_w${W}" \
    --models flash --shots 0shot --schemes 6cls --context-words "$W"
done

log "SWEEP END  cells failed: ${#FAILED[@]}"
if ((${#FAILED[@]})); then
  log "Failed cells: ${FAILED[*]}"
  exit 1
fi

log "Next: results land as runs/batch_FULLROG_CTXW<W>_<stamp>/"
log "  Each W has two batch dirs (different stamps, same W): one 3cls, one 6cls."
log "Analyze each: python3 scripts/analyze_probe.py runs/batch_FULLROG_CTXW<W>_<stamp>"
