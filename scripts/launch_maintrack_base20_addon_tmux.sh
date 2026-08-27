#!/usr/bin/env bash
set -Eeuo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

session="${MAINTRACK_BASE20_TMUX_SESSION:-cartography-base20}"
gpu_ids="${MAINTRACK_GPU_IDS:-0,1}"
results_dir="results/maintrack_week"
status_dir="$results_dir/status"
logs_dir="$results_dir/logs"
nix_bin="/nix/var/nix/profiles/default/bin/nix"
minimum_free_gib="${MAINTRACK_MIN_FREE_GIB:-20}"

mkdir -p "$status_dir" "$logs_dir"

if tmux has-session -t "$session" 2>/dev/null; then
  if tmux list-panes -s -t "$session" -F '#{pane_dead}' | grep -qx '0'; then
    echo "ERROR: live tmux session already exists: $session" >&2
    tmux list-panes -s -t "$session" \
      -F 'pane=#{pane_id} dead=#{pane_dead} exit=#{pane_dead_status} pid=#{pane_pid} cmd=#{pane_current_command}' >&2
    exit 1
  fi
  tmux kill-session -t "$session"
fi

# Refuse to overlap the add-on with any existing main-track model/controller job.
./.venv/bin/python3 - "$repo" <<'PYPROC'
import os
import sys
from pathlib import Path

repo = str(Path(sys.argv[1]).resolve())
found = []
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        if entry.stat().st_uid != os.getuid():
            continue
        argv = [
            item.decode(errors="replace")
            for item in (entry / "cmdline").read_bytes().split(b"\0")
            if item
        ]
        if not argv:
            continue
        try:
            cwd = str((entry / "cwd").resolve())
        except OSError:
            cwd = ""
    except (OSError, PermissionError, ProcessLookupError, FileNotFoundError):
        continue

    text = " ".join(argv)
    suite = any(arg.endswith("scripts/run_maintrack_suite.py") for arg in argv) and repo in text
    addon = any(arg.endswith("scripts/run_maintrack_base20_addon.py") for arg in argv) and repo in text
    worker = cwd == repo and any(arg == "run.py" for arg in argv)
    if suite or addon or worker:
        found.append((int(entry.name), text))

if found:
    print("ERROR: main-track Python work is already active:", file=sys.stderr)
    for pid, text in sorted(found):
        print(f"  {pid}: {text}", file=sys.stderr)
    raise SystemExit(2)
PYPROC

free_bytes="$(df -PB1 "$results_dir" | awk 'NR==2 {print $4}')"
minimum_bytes="$(./.venv/bin/python3 -c 'import sys; print(int(float(sys.argv[1]) * 1024**3))' "$minimum_free_gib")"
if [ "$free_bytes" -lt "$minimum_bytes" ]; then
  ./scripts/reclaim_maintrack_space.py \
    --config configs/maintrack.full.yaml \
    --warn-free-gib "$minimum_free_gib" \
    --require-free-gib "$minimum_free_gib"
  free_bytes="$(df -PB1 "$results_dir" | awk 'NR==2 {print $4}')"
fi
if [ "$free_bytes" -lt "$minimum_bytes" ]; then
  echo "ERROR: insufficient free space for add-on: $(numfmt --to=iec "$free_bytes") free; need ${minimum_free_gib} GiB" >&2
  exit 1
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
runner="$status_dir/base20_runner-$timestamp.sh"
supervisor="$status_dir/base20_supervisor-$timestamp.sh"
log="$logs_dir/base20-$timestamp.log"

cat > "$runner" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
cd $repo
export MAINTRACK_GPU_IDS=$gpu_ids
export MAINTRACK_HF_SCRATCH_MIN_FREE_GIB=8
exec ./.venv/bin/python3 scripts/run_maintrack_base20_addon.py --gpu-ids $gpu_ids
EOF
chmod +x "$runner"

cat > "$supervisor" <<EOF
#!/usr/bin/env bash
set -uo pipefail
exec > >(tee -a $log) 2>&1
cd $repo
printf 'ELECTRA-base 20%% add-on supervisor started at %s.\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
set +e
$nix_bin develop --no-update-lock-file --no-write-lock-file $repo#server --command bash $repo/$runner
exit_code=\$?
set -e
printf '%s\n' "\$exit_code" > $status_dir/base20_runner_exit_code.txt
printf 'ELECTRA-base 20%% add-on exited with code %s at %s.\n' "\$exit_code" "\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "\$exit_code"
EOF
chmod +x "$supervisor"

printf '%s\n' "$log" > "$status_dir/base20_tmux_log_path.txt"
printf '%s\n' "$session" > "$status_dir/base20_tmux_session.txt"

tmux set-option -g remain-on-exit on
tmux new-session -d -s "$session" "exec bash '$repo/$supervisor'"

sleep 1
if ! tmux has-session -t "$session" 2>/dev/null; then
  echo "ERROR: failed to create tmux session $session" >&2
  exit 1
fi

printf 'Started add-on tmux session: %s\n' "$session"
printf 'Durable add-on log: %s\n' "$repo/$log"
printf 'Attach: tmux attach -t %s\n' "$session"
