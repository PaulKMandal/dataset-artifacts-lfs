import os
import subprocess
import sys
from pathlib import Path


def test_remote_scripts_exist_and_are_executable():
    root = Path(__file__).resolve().parents[1]
    for rel in [
        "scripts/remote_run.sh",
        "scripts/pull_results.sh",
        "scripts/run_qa_smoke.sh",
        "scripts/pull_review_results.sh",
        "scripts/tar_review_bundle.sh",
        "scripts/tar_code_bundle.sh",
        "scripts/recover_adversarial_squad_sources.sh",
        "scripts/validate_maintrack_launch.sh",
    ]:
        path = root / rel
        assert path.exists()
        assert path.stat().st_mode & 0o111


def test_maintrack_entrypoints_resolve_repo_imports(tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    for rel in [
        "scripts/materialize_sentence_classification.py",
        "scripts/build_mechanism_features.py",
        "scripts/analyze_subset_coverage.py",
        "scripts/combine_cartography_maps.py",
    ]:
        subprocess.run(
            [sys.executable, str(root / rel), "--help"],
            check=True,
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
        )


def test_tmux_launcher_runs_foreground_gate_before_creating_session():
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "scripts" / "launch_maintrack_tmux.sh").read_text(encoding="utf-8")
    gate = 'scripts/validate_maintrack_launch.sh "$config_path"'
    create_session = '"$TMUX_BIN" new-session -d -s "$SESSION"'
    assert gate in launcher
    assert create_session in launcher
    assert launcher.index(gate) < launcher.index(create_session)
    assert "Launch validation failed; no tmux session or full batch was started." in launcher


def _fake_tmux_environment(tmp_path: Path, pane_line: str) -> dict[str, str]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    state = tmp_path / "tmux.state"
    panes = tmp_path / "tmux.panes"
    log = tmp_path / "tmux.log"
    state.write_text("exists\n", encoding="utf-8")
    panes.write_text(pane_line + "\n", encoding="utf-8")

    tmux = fake_bin / "tmux"
    tmux.write_text(
        """#!/usr/bin/env bash
set -eu
cmd="$1"
shift
case "$cmd" in
  has-session)
    [ -f "$FAKE_TMUX_STATE" ]
    ;;
  list-panes)
    cat "$FAKE_TMUX_PANES"
    ;;
  kill-session)
    echo kill-session >> "$FAKE_TMUX_LOG"
    rm -f "$FAKE_TMUX_STATE"
    ;;
  new-session)
    echo new-session >> "$FAKE_TMUX_LOG"
    : > "$FAKE_TMUX_STATE"
    ;;
  set-option|pipe-pane|send-keys)
    echo "$cmd" >> "$FAKE_TMUX_LOG"
    ;;
  *)
    echo "unexpected fake tmux command: $cmd" >&2
    exit 64
    ;;
esac
""",
        encoding="utf-8",
    )
    tmux.chmod(0o755)

    nix = fake_bin / "nix"
    nix.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    nix.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "FAKE_TMUX_STATE": str(state),
            "FAKE_TMUX_PANES": str(panes),
            "FAKE_TMUX_LOG": str(log),
            "MAINTRACK_TMUX_SESSION": "cartography-maintrack",
        }
    )
    return environment


def test_tmux_launcher_removes_all_dead_stale_session(tmp_path):
    root = Path(__file__).resolve().parents[1]
    launcher = root / "scripts" / "launch_maintrack_tmux.sh"
    config = tmp_path / "maintrack.yaml"
    config.write_text("results_dir: results/maintrack_week\n", encoding="utf-8")
    environment = _fake_tmux_environment(tmp_path, "%1|1|143|bash")

    completed = subprocess.run(
        ["bash", str(launcher), str(config)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Removing stale tmux inspection session" in completed.stderr
    actions = Path(environment["FAKE_TMUX_LOG"]).read_text(encoding="utf-8").splitlines()
    assert actions[0:2] == ["kill-session", "new-session"]


def test_tmux_launcher_removes_legacy_idle_bash_session(tmp_path):
    root = Path(__file__).resolve().parents[1]
    launcher = root / "scripts" / "launch_maintrack_tmux.sh"
    config = tmp_path / "maintrack.yaml"
    config.write_text("results_dir: results/maintrack_week\n", encoding="utf-8")
    # Use an impossible PID so the launcher sees the legacy bash pane as
    # childless. This reproduces the old launcher state after its supervised
    # runner exited and returned control to the parent shell.
    environment = _fake_tmux_environment(tmp_path, "%1|0||bash|999999")

    completed = subprocess.run(
        ["bash", str(launcher), str(config)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Removing stale tmux inspection session" in completed.stderr
    actions = Path(environment["FAKE_TMUX_LOG"]).read_text(encoding="utf-8").splitlines()
    assert actions[0:2] == ["kill-session", "new-session"]


def test_tmux_launcher_refuses_legacy_bash_with_live_child(tmp_path):
    root = Path(__file__).resolve().parents[1]
    launcher = root / "scripts" / "launch_maintrack_tmux.sh"
    config = tmp_path / "maintrack.yaml"
    config.write_text("results_dir: results/maintrack_week\n", encoding="utf-8")

    parent = subprocess.Popen(["bash", "-c", "sleep 60 & wait"])
    try:
        environment = _fake_tmux_environment(tmp_path, f"%1|0||bash|{parent.pid}")
        completed = subprocess.run(
            ["bash", str(launcher), str(config)],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        parent.terminate()
        parent.wait(timeout=5)

    assert completed.returncode == 1
    assert "tmux session already exists with active pane(s)" in completed.stderr
    log = Path(environment["FAKE_TMUX_LOG"])
    assert not log.exists() or "kill-session" not in log.read_text(encoding="utf-8")



def test_tmux_launcher_execs_supervisor_into_pane():
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "scripts" / "launch_maintrack_tmux.sh").read_text(encoding="utf-8")
    assert '"$TMUX_BIN" send-keys -t "$SESSION:experiments.0" "exec bash ' in launcher
    assert '"$TMUX_BIN" send-keys -t "$SESSION:experiments.0" "bash ' not in launcher


def test_tmux_launcher_refuses_session_with_live_pane(tmp_path):
    root = Path(__file__).resolve().parents[1]
    launcher = root / "scripts" / "launch_maintrack_tmux.sh"
    config = tmp_path / "maintrack.yaml"
    config.write_text("results_dir: results/maintrack_week\n", encoding="utf-8")
    environment = _fake_tmux_environment(tmp_path, "%1|0||python3|12345")

    completed = subprocess.run(
        ["bash", str(launcher), str(config)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "tmux session already exists with active pane(s)" in completed.stderr
    log = Path(environment["FAKE_TMUX_LOG"])
    assert not log.exists() or "kill-session" not in log.read_text(encoding="utf-8")
