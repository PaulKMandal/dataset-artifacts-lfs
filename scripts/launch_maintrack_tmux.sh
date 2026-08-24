#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/maintrack.full.yaml}"
SESSION="${MAINTRACK_TMUX_SESSION:-cartography-maintrack}"
GPU_IDS="${MAINTRACK_GPU_IDS:-0,1}"
REPO_ROOT="$(pwd -P)"

case "$CONFIG" in
  /*) config_path="$CONFIG" ;;
  *) config_path="$REPO_ROOT/$CONFIG" ;;
esac

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required; enter the Nix server shell first." >&2
  exit 1
fi
if ! command -v nix >/dev/null 2>&1; then
  echo "nix is required to create an isolated runtime inside tmux." >&2
  exit 1
fi
NIX_BIN="$(command -v nix)"
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
  done < "$config_path"
)"
if [ -z "$results_dir" ]; then
  echo "Could not read suite.results_dir from $config_path" >&2
  exit 1
fi
case "$results_dir" in
  /*) ;;
  *) results_dir="$REPO_ROOT/$results_dir" ;;
esac
mkdir -p "$results_dir/logs" "$results_dir/status"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
tmux_log="$results_dir/logs/tmux-${timestamp}.log"
pane_log="$results_dir/logs/tmux-pane-${timestamp}.log"
runner_file="$results_dir/status/tmux_runner-${timestamp}.sh"
supervisor_file="$results_dir/status/tmux_supervisor-${timestamp}.sh"
supervisor_exit_file="$results_dir/status/tmux_supervisor_exit_code.txt"

{
  printf '#!/usr/bin/env bash\n'
  printf 'set -Eeuo pipefail\n'
  printf 'cd %q\n' "$REPO_ROOT"
  printf 'export MAINTRACK_GPU_IDS=%q\n' "$GPU_IDS"
  printf 'exec scripts/run_maintrack_week.sh %q\n' "$config_path"
} > "$runner_file"
chmod +x "$runner_file"

{
  printf '#!/usr/bin/env bash\n'
  printf 'set -uo pipefail\n'
  printf 'exec > >(tee -a %q) 2>&1\n' "$tmux_log"
  printf 'cd %q\n' "$REPO_ROOT"
  printf 'printf "Main-track supervisor started at %%s.\\n" "$(date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ)"\n'
  printf 'set +e\n'
  printf '%q develop --no-update-lock-file --no-write-lock-file %q --command bash %q\n' \
    "$NIX_BIN" "$REPO_ROOT#server" "$runner_file"
  printf 'runner_exit_code=$?\n'
  printf 'set -e\n'
  printf 'printf "%%s\\n" "$runner_exit_code" > %q\n' "$supervisor_exit_file"
  printf 'printf "\\nMain-track runner exited with code %%s at %%s.\\n" "$runner_exit_code" "$(date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ)"\n'
  printf 'printf "This tmux pane is intentionally being kept open for inspection.\\n"\n'
  printf 'exit "$runner_exit_code"\n'
} > "$supervisor_file"
chmod +x "$supervisor_file"

: > "$tmux_log"
: > "$pane_log"
tmux new-session -d -s "$SESSION" -n experiments -c "$REPO_ROOT" bash --noprofile --norc
tmux pipe-pane -o -t "$SESSION:experiments.0" "exec cat >> $(printf '%q' "$pane_log")"
printf '%s\n' "$SESSION" > "$results_dir/status/tmux_session.txt"
printf '%s\n' "$tmux_log" > "$results_dir/status/tmux_log_path.txt"
printf '%s\n' "$pane_log" > "$results_dir/status/tmux_pane_log_path.txt"
tmux send-keys -t "$SESSION:experiments.0" "bash $(printf '%q' "$supervisor_file")" Enter

echo "Started tmux session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Durable runner log: $tmux_log"
echo "Raw pane log: $pane_log"
echo "Status: uv run --no-sync python scripts/maintrack_status.py --config $config_path --watch 10"
