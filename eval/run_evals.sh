#!/usr/bin/env bash
# =============================================================================
# Stage 2 — Pre-Deployment Red-Teaming & Safety Evaluation Runner
#
# Orchestrates the full evaluation pipeline:
#   1. Starts the agent HTTP API (FastAPI wrapper)
#   2. Runs promptfoo generative red-teaming (40 vulnerability probes)
#   3. Runs Inspect AI benchmark evaluations (3 task suites, 30 samples)
#   4. Prints a combined pass/fail summary
#
# Prerequisites:
#   pip install fastapi uvicorn httpx inspect-ai
#   npm install -g promptfoo        (or: cd eval/promptfoo && npm install)
#   OPENAI_API_KEY exported
#
# Usage:
#   bash eval/run_evals.sh [--skip-promptfoo] [--skip-inspect] [--port 8000]
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS_DIR="$PROJECT_ROOT/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PORT="${PORT:-8000}"
API_URL="http://localhost:${PORT}"
API_PID=""

# ── Argument parsing ──────────────────────────────────────────────────────────
SKIP_PROMPTFOO=false
SKIP_INSPECT=false

for arg in "$@"; do
  case "$arg" in
    --skip-promptfoo) SKIP_PROMPTFOO=true ;;
    --skip-inspect)   SKIP_INSPECT=true   ;;
    --port)           shift; PORT="$1"    ;;
  esac
done

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC}  $*"; }
warn() { echo -e "${YELLOW}!${NC}  $*"; }
fail() { echo -e "${RED}✗${NC}  $*"; }

header() {
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "  $*"
  echo "════════════════════════════════════════════════════════════════"
}

# ── Ensure log directories exist ──────────────────────────────────────────────
mkdir -p "$LOGS_DIR/promptfoo" "$LOGS_DIR/inspect_ai"

# ── Cleanup: stop the API server on exit ─────────────────────────────────────
cleanup() {
  if [[ -n "$API_PID" ]] && kill -0 "$API_PID" 2>/dev/null; then
    echo ""
    echo "Stopping agent API server (PID $API_PID)..."
    kill "$API_PID" && wait "$API_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Start the agent HTTP API
# ─────────────────────────────────────────────────────────────────────────────
header "Step 1/3  —  Starting Retail Support Agent HTTP API"

cd "$PROJECT_ROOT"

# Check if something is already listening on the port
if curl -sf "${API_URL}/health" >/dev/null 2>&1; then
  warn "API already running at ${API_URL} — skipping startup."
else
  python -m uvicorn api.server:app \
      --host 0.0.0.0 \
      --port "$PORT" \
      --log-level warning &
  API_PID=$!

  echo -n "  Waiting for API to become ready"
  for i in $(seq 1 30); do
    if curl -sf "${API_URL}/health" >/dev/null 2>&1; then
      echo ""
      ok "API ready at ${API_URL}  (PID $API_PID)"
      break
    fi
    echo -n "."
    sleep 1
    if [[ $i -eq 30 ]]; then
      echo ""
      fail "API did not start within 30 s — aborting."
      exit 1
    fi
  done
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — promptfoo generative red-teaming
# ─────────────────────────────────────────────────────────────────────────────
header "Step 2/3  —  promptfoo Red-Team Evaluation"

if [[ "$SKIP_PROMPTFOO" == "true" ]]; then
  warn "Skipping promptfoo (--skip-promptfoo flag set)."
else
  PF_RESULT="$LOGS_DIR/promptfoo/results_${TIMESTAMP}.json"
  PF_LOG="$LOGS_DIR/promptfoo/run_${TIMESTAMP}.log"

  cd "$PROJECT_ROOT/eval/promptfoo"

  echo "  Config : eval/promptfoo/promptfooconfig.yaml"
  echo "  Output : logs/promptfoo/results_${TIMESTAMP}.json"
  echo ""

  # Run promptfoo redteam; persist both the JSON result and full console log
  if promptfoo redteam run \
      --config promptfooconfig.yaml \
      --output "$PF_RESULT" \
      --no-cache \
      2>&1 | tee "$PF_LOG"; then
    ok "promptfoo red-team complete."
  else
    fail "promptfoo exited with a non-zero status. Check $PF_LOG for details."
  fi

  cd "$PROJECT_ROOT"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Inspect AI benchmark evaluations
# ─────────────────────────────────────────────────────────────────────────────
header "Step 3/3  —  Inspect AI Safety Evaluations"

if [[ "$SKIP_INSPECT" == "true" ]]; then
  warn "Skipping Inspect AI (--skip-inspect flag set)."
else
  INS_LOG="$LOGS_DIR/inspect_ai/run_${TIMESTAMP}.log"

  echo "  Tasks  : agent_harm_eval, strong_reject_eval, boundary_check_eval"
  echo "  Output : logs/inspect_ai/"
  echo ""

  cd "$PROJECT_ROOT"

  AGENT_BASE_URL="$API_URL" \
  inspect eval eval/inspect_ai/safety_eval.py \
      --log-dir "logs/inspect_ai" \
      --log-format json \
      2>&1 | tee "$INS_LOG"

  ok "Inspect AI evaluations complete."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
header "Evaluation Summary"

python3 - "$LOGS_DIR" "$TIMESTAMP" <<'PYEOF'
import json, glob, sys
from pathlib import Path

logs_dir  = Path(sys.argv[1])
timestamp = sys.argv[2]

print(f"  Run timestamp : {timestamp}")
print()

# ── promptfoo ──────────────────────────────────────────────────────────────
pf_files = sorted((logs_dir / "promptfoo").glob("results_*.json"))
if pf_files:
    with open(pf_files[-1]) as f:
        pf = json.load(f)
    stats  = pf.get("results", {}).get("stats", pf.get("stats", {}))
    passed = stats.get("successes", 0)
    failed = stats.get("failures",  0)
    errors = stats.get("errors",    0)
    total  = passed + failed + errors
    rate   = f"{passed/total*100:.1f}%" if total else "n/a"
    print(f"  promptfoo")
    print(f"    Tests run : {total}")
    print(f"    Pass      : {passed}  ({rate})")
    print(f"    Fail      : {failed}")
    print(f"    Errors    : {errors}")
else:
    print("  promptfoo      : no results found")

print()

# ── Inspect AI ─────────────────────────────────────────────────────────────
ins_files = sorted((logs_dir / "inspect_ai").glob("*.json"))
if ins_files:
    print("  Inspect AI")
    for fp in ins_files[-3:]:           # show last 3 log files
        try:
            with open(fp) as f:
                data = json.load(f)
            task_name = data.get("eval", {}).get("task", fp.stem)
            results   = data.get("results", {})
            scores    = results.get("scores", results.get("metrics", {}))
            acc       = scores.get("accuracy", "n/a")
            total_s   = results.get("total_samples", "?")
            print(f"    {task_name:<30}  accuracy={acc}  (n={total_s})")
        except Exception:
            print(f"    {fp.name}  (could not parse)")
else:
    print("  Inspect AI     : no results found")

print()
PYEOF

echo "  Full logs: $LOGS_DIR"
echo "  View promptfoo HTML report : cd eval/promptfoo && promptfoo redteam report"
echo "  View Inspect AI results    : inspect view --log-dir logs/inspect_ai/"
echo ""
ok "Stage 2 evaluation pipeline complete."
