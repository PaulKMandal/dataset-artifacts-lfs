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
