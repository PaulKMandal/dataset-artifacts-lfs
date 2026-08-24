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
TMUX_BIN="$(command -v tmux)"
NIX_BIN="$(command -v nix)"

# A supervised run deliberately sets remain-on-exit so a failed pane stays
# inspectable.  Older launcher revisions started the supervisor as a child of
# an interactive bash.  When that child exited, tmux still reported the idle
# parent bash as a live pane even though the experiment was gone.  Treat a
# childless bash pane as a stale inspection shell, but refuse to touch any pane
# that is dead=0 and either has children or is running a non-shell command.
if "$TMUX_BIN" has-session -t "$SESSION" 2>/dev/null; then
  live_panes=0
  pane_summary=""
  while IFS='|' read -r pane_id pane_dead pane_status pane_command pane_pid; do
    [ -n "$pane_id" ] || continue
    pane_summary+="  pane=$pane_id dead=$pane_dead status=${pane_status:-?} command=${pane_command:-?} pid=${pane_pid:-?}"$'\n'
    if [ "$pane_dead" = "1" ]; then
      continue
    fi

    if [ "$pane_command" = "bash" ] && [ -n "${pane_pid:-}" ]; then
      pane_children="$(ps -o pid= --ppid "$pane_pid" 2>/dev/null || true)"
      if [ -z "${pane_children//[[:space:]]/}" ]; then
        continue
      fi
    fi

    live_panes=$((live_panes + 1))
  done < <(
    "$TMUX_BIN" list-panes -s -t "$SESSION" \
      -F '#{pane_id}|#{pane_dead}|#{pane_dead_status}|#{pane_current_command}|#{pane_pid}'
  )

  if [ "$live_panes" -gt 0 ]; then
    echo "tmux session already exists with active pane(s): $SESSION" >&2
    printf '%s' "$pane_summary" >&2
    echo "Attach with: $TMUX_BIN attach -t $SESSION" >&2
    exit 1
  fi

  echo "Removing stale tmux inspection session: $SESSION" >&2
  printf '%s' "$pane_summary" >&2
  "$TMUX_BIN" kill-session -t "$SESSION"
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
validation_log="$results_dir/logs/launch-validation-${timestamp}.log"
tmux_log="$results_dir/logs/tmux-${timestamp}.log"
pane_log="$results_dir/logs/tmux-pane-${timestamp}.log"
runner_file="$results_dir/status/tmux_runner-${timestamp}.sh"
supervisor_file="$results_dir/status/tmux_supervisor-${timestamp}.sh"
supervisor_exit_file="$results_dir/status/tmux_supervisor_exit_code.txt"

: > "$validation_log"
printf '%s\n' "$validation_log" > "$results_dir/status/launch_validation_log_path.txt"
echo "Running the complete data and end-to-end GPU smoke gate before tmux launch."
set +e
"$NIX_BIN" develop \
  --no-update-lock-file \
  --no-write-lock-file \
  "$REPO_ROOT#server" \
  --command scripts/validate_maintrack_launch.sh "$config_path" \
  2>&1 | tee -a "$validation_log"
validation_exit=${PIPESTATUS[0]}
set -e
if [ "$validation_exit" -ne 0 ]; then
  echo "Launch validation failed; no tmux session or full batch was started." >&2
  echo "Durable validation log: $validation_log" >&2
  exit "$validation_exit"
fi

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
"$TMUX_BIN" new-session -d -s "$SESSION" -n experiments -c "$REPO_ROOT" bash --noprofile --norc
"$TMUX_BIN" set-option -w -t "$SESSION:experiments" remain-on-exit on
"$TMUX_BIN" pipe-pane -o -t "$SESSION:experiments.0" "exec cat >> $(printf '%q' "$pane_log")"
printf '%s\n' "$SESSION" > "$results_dir/status/tmux_session.txt"
printf '%s\n' "$tmux_log" > "$results_dir/status/tmux_log_path.txt"
printf '%s\n' "$pane_log" > "$results_dir/status/tmux_pane_log_path.txt"
"$TMUX_BIN" send-keys -t "$SESSION:experiments.0" "exec bash $(printf '%q' "$supervisor_file")" Enter

echo "Started tmux session: $SESSION"
echo "Attach: $TMUX_BIN attach -t $SESSION"
echo "Foreground validation log: $validation_log"
echo "Durable runner log: $tmux_log"
echo "Raw pane log: $pane_log"
echo "Status: uv run --no-sync python scripts/maintrack_status.py --config $config_path --watch 10"
