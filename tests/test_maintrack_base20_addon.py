from pathlib import Path

from scripts import run_maintrack_base20_addon as addon_runner
from scripts import run_maintrack_suite as suite

ROOT = Path(__file__).resolve().parents[1]


def test_addon_exactly_fills_electra_base_20pct_budget_gap():
    config = suite.load_config(ROOT / "configs" / "maintrack.full.yaml")
    addon = addon_runner.load_addon(ROOT / "configs" / "maintrack.electra_base_20pct_addon.yaml")
    specs = addon_runner.addon_specs(config, addon)

    assert len(specs) == 12
    assert {spec.model_key for spec in specs} == {"electra_base"}
    assert {spec.task for spec in specs} == {"qa", "classification"}
    assert {spec.subset_fraction for spec in specs} == {0.20}
    assert {spec.seed for spec in specs} == {42}

    for task in ("qa", "classification"):
        task_specs = [spec for spec in specs if spec.task == task]
        assert len(task_specs) == 6
        assert sorted(
            spec.subset_draw_id for spec in task_specs if spec.train_subset == "random"
        ) == [0, 1, 2]
        assert {
            spec.train_subset for spec in task_specs if spec.train_subset != "random"
        } == {"easy", "ambiguous", "hard"}


def test_addon_does_not_overlap_original_budget_panel():
    config = suite.load_config(ROOT / "configs" / "maintrack.full.yaml")
    addon = addon_runner.load_addon(ROOT / "configs" / "maintrack.electra_base_20pct_addon.yaml")
    addon_ids = {spec.run_id for spec in addon_runner.addon_specs(config, addon)}
    original_ids = {spec.run_id for spec in suite.build_budget_specs(config)}

    assert addon_ids.isdisjoint(original_ids)
    assert len(original_ids) == 60


def test_addon_launcher_is_present():
    launcher = ROOT / "scripts" / "launch_maintrack_base20_addon_tmux.sh"
    assert launcher.exists()
    assert launcher.stat().st_mode & 0o111
