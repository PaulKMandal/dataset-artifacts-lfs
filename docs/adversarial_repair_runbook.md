# Adversarial repair runbook

This runbook explains how to repair AddSent/AddOneSent robustness metrics without retraining.

The repair computes split metrics from row-level predictions when available. If predictions are missing and checkpoints exist, the repair can regenerate evaluation predictions only. It never trains models.

Primary robustness metrics must be adversarial-only and paired; all-row AddSent/AddOneSent metrics are retained only as legacy comparability numbers.
