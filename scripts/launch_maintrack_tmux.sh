#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/maintrack.full.yaml}"
SESSION="${MAINTRACK_TMUX_SESSION:-cartography-maintrack}"
GPU_IDS="${MAINTRACK_GPU_IDS:-0,1}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required; enter the Nix server shell first." >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  echo "Attach with: tmux attach -t $SESSION" >&2
  exit 1
fi

results_dir="$(
  while read -r key value _; do
    if [ "$key" = "results_dir:" ]; then
      printf '%s\n' "$value"
      break
    fi
  done < "$CONFIG"
)"
if [ -z "$results_dir" ]; then
  echo "Could not read suite.results_dir from $CONFIG" >&2
  exit 1
fi
mkdir -p "$results_dir/logs" "$results_dir/status"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
tmux_log="$results_dir/logs/tmux-${timestamp}.log"
runner_file="$results_dir/status/tmux_runner.sh"

{
  printf '#!/usr/bin/env bash\n'
  printf 'set -euo pipefail\n'
  printf 'cd %q\n' "$PWD"
  printf 'export MAINTRACK_GPU_IDS=%q\n' "$GPU_IDS"
  printf 'exec scripts/run_maintrack_week.sh %q\n' "$CONFIG"
} > "$runner_file"
chmod +x "$runner_file"

tmux new-session -d -s "$SESSION" -n experiments bash
tmux pipe-pane -o -t "$SESSION:experiments.0" "exec cat >> $(printf '%q' "$tmux_log")"
tmux send-keys -t "$SESSION:experiments.0" "exec bash $(printf '%q' "$runner_file")" Enter

printf '%s\n' "$SESSION" > "$results_dir/status/tmux_session.txt"
printf '%s\n' "$tmux_log" > "$results_dir/status/tmux_log_path.txt"

echo "Started tmux session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Full pane log: $tmux_log"
echo "Status: uv run --no-sync python scripts/maintrack_status.py --config $CONFIG --watch 10"
