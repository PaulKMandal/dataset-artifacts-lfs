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
