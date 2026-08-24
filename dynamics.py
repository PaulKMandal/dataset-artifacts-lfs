#!/usr/bin/env python3
"""Dataset-cartography aggregation and plotting for training dynamics.

The loader supports both formats:

1. Legacy rows with full probability vectors:
   - NLI: {idx, label, prob: [...]}
   - QA:  {idx, start_position, end_position, start_prob: [...], end_prob: [...]}

2. Scalar rows emitted by the fast DynamicsLogger:
   - {idx, confidence, correctness, ...}

The output cartography coordinates are one row per original example index.
"""

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

CATEGORY_LABELS = {
    "easy": "Easy-to-learn",
    "ambiguous": "Ambiguous",
    "hard": "Hard-to-learn",
}


def load_training_dynamics(td_dir_or_file):
    """Load dynamics rows from a JSONL file or a directory containing them."""
    path = Path(td_dir_or_file)
    if path.is_dir():
        candidates = sorted(path.glob("training_dynamics*.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No training_dynamics*.jsonl files found in {path}")
    else:
        candidates = [path]

    dynamics = defaultdict(list)
    for candidate in candidates:
        with candidate.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                idx = int(record["idx"])
                dynamics[idx].append(record)
    return dynamics


def _metrics_from_record(record, confidence_field="confidence"):
    """Return (confidence, correctness) for one legacy or scalar dynamics row."""
    if confidence_field in record and "correctness" in record:
        return float(record[confidence_field]), float(record["correctness"])

    if confidence_field == "negative_gold_span_loss" and "correctness" in record:
        if "start_logp" not in record or "end_logp" not in record:
            raise ValueError("negative_gold_span_loss requires start_logp and end_logp fields")
        # Cross-entropy loss is -(log p_start_gold + log p_end_gold).
        # Use its negative so higher values remain better for selection.
        return float(record["start_logp"] + record["end_logp"]), float(record["correctness"])

    # Backward-compatible NLI path.
    if "prob" in record:
        probs = np.array(record["prob"], dtype=np.float64)
        true_label = int(record["label"])
        pred_label = int(np.argmax(probs))
        return float(probs[true_label]), float(pred_label == true_label)

    # Backward-compatible QA path.
    if "start_prob" in record and "end_prob" in record:
        start_probs = np.array(record["start_prob"], dtype=np.float64)
        end_probs = np.array(record["end_prob"], dtype=np.float64)
        true_start = int(record["start_position"])
        true_end = int(record["end_position"])
        pred_start = int(np.argmax(start_probs))
        pred_end = int(np.argmax(end_probs))
        confidence = (start_probs[true_start] + end_probs[true_end]) / 2.0
        correctness = float(pred_start == true_start and pred_end == true_end)
        return float(confidence), correctness

    raise ValueError(f"Unrecognized dynamics row schema: {sorted(record.keys())}")


AGGREGATIONS = {
    "records",
    "epoch_mean",
    "answer_only_records",
    "answer_only_epoch_mean",
}


def _aggregate_record_metrics(records, confidence_field, aggregation):
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"Unknown aggregation {aggregation!r}; choose from {sorted(AGGREGATIONS)}")
    answer_only = aggregation.startswith("answer_only_")
    if answer_only:
        if not any("answer_in_window" in record for record in records):
            raise ValueError(
                "answer-only aggregation requires dynamics logged with answer_in_window"
            )
        records = [record for record in records if record.get("answer_in_window") is True]

    values = []
    for record in records:
        try:
            values.append(_metrics_from_record(record, confidence_field=confidence_field))
        except (KeyError, ValueError):
            if confidence_field == "confidence":
                raise
            values.append(_metrics_from_record(record, confidence_field="confidence"))

    if aggregation.endswith("epoch_mean"):
        by_epoch = defaultdict(list)
        for record, value in zip(records, values, strict=False):
            epoch = record.get("epoch")
            if epoch is None:
                step = record.get("step")
                epoch_key = ("step", int(step) if step is not None else 0)
            else:
                # TrainerState.epoch is fractional progress (for example,
                # 0.42), not an epoch identifier. Bucket all feature records
                # observed during the same pass through the data together.
                epoch_key = ("epoch", math.floor(float(epoch)))
            by_epoch[epoch_key].append(value)
        values = [
            (mean(v[0] for v in epoch_values), mean(v[1] for v in epoch_values))
            for _, epoch_values in sorted(by_epoch.items(), key=lambda item: item[0])
        ]
    return values


def compute_metrics(dynamics, confidence_field="confidence", aggregation="records"):
    """Compute cartography metrics per example index."""
    metrics = {}
    for idx, records in dynamics.items():
        values = _aggregate_record_metrics(records, confidence_field, aggregation)
        confidences = [value[0] for value in values]
        correctnesses = [value[1] for value in values]
        if confidences:
            metrics[idx] = {
                "n_records": len(confidences),
                "avg_confidence": float(mean(confidences)),
                "variability": float(pstdev(confidences)) if len(confidences) > 1 else 0.0,
                "correctness": float(mean(correctnesses)),
            }
    return metrics


def categorize_examples(metrics):
    """Categorize examples by correctness, retaining the original labels."""
    categories = {}
    for idx, m in metrics.items():
        correctness = m["correctness"]
        if correctness == 1.0:
            categories[idx] = CATEGORY_LABELS["easy"]
        elif correctness >= 0.5:
            categories[idx] = CATEGORY_LABELS["ambiguous"]
        else:
            categories[idx] = CATEGORY_LABELS["hard"]
    return categories


def save_cartography_csv(
    metrics,
    categories,
    output_dir,
    *,
    confidence_field="confidence",
    aggregation="records",
):
    out_path = Path(output_dir) / "cartography_scores.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "n_records",
                "confidence",
                "variability",
                "correctness",
                "region",
                "confidence_field",
                "aggregation",
            ],
        )
        writer.writeheader()
        for idx in sorted(metrics):
            m = metrics[idx]
            writer.writerow(
                {
                    "idx": idx,
                    "n_records": m["n_records"],
                    "confidence": m["avg_confidence"],
                    "variability": m["variability"],
                    "correctness": m["correctness"],
                    "region": categories[idx],
                    "confidence_field": confidence_field,
                    "aggregation": aggregation,
                }
            )
    return out_path


def plot_scatter(metrics, categories, output_dir, limit_scatter_samples=None, seed=None):
    indices = list(metrics.keys())
    if limit_scatter_samples is not None and len(indices) > limit_scatter_samples:
        rng = random.Random(seed)
        indices = rng.sample(indices, limit_scatter_samples)

    category_colors = {
        "Easy-to-learn": "green",
        "Ambiguous": "orange",
        "Hard-to-learn": "red",
    }
    x = [metrics[idx]["variability"] for idx in indices]
    y = [metrics[idx]["avg_confidence"] for idx in indices]
    colors = [category_colors[categories[idx]] for idx in indices]

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c=colors, alpha=0.7, s=1)
    plt.xlabel("Variability (Std Dev of Confidence)")
    plt.ylabel("Average Confidence in True Class")
    plt.title("Confidence vs. Variability Scatter Plot")
    plt.grid(True)

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=color, markersize=8)
        for label, color in category_colors.items()
    ]
    plt.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "confidence_vs_variability_scatter.png")
    plt.close()


def plot_histogram(data, xlabel, title, output_path):
    if not data:
        return
    plt.figure(figsize=(8, 5))
    vals = np.array(data, dtype=np.float64)
    if len(vals) >= 2 and np.min(vals) != np.max(vals):
        density = gaussian_kde(vals)
        xs = np.linspace(np.min(vals), np.max(vals), 200)
        density.covariance_factor = lambda: 0.25
        density._compute_covariance()
        plt.plot(xs, density(xs))
        plt.fill_between(xs, density(xs), alpha=0.5)
    else:
        plt.hist(vals, bins=1)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot and aggregate training dynamics.")
    parser.add_argument(
        "--td_dir", type=str, required=True, help="Directory or JSONL file containing dynamics."
    )
    parser.add_argument(
        "--output_dir", type=str, default=".", help="Directory for plots and outputs."
    )
    parser.add_argument("--limit_scatter_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--confidence_field",
        type=str,
        default="confidence",
        choices=["confidence", "joint_confidence", "negative_gold_span_loss"],
        help=(
            "For scalar QA rows, choose endpoint-average confidence, joint span "
            "confidence, or negative gold-span loss. Negative loss is higher-is-better."
        ),
    )
    parser.add_argument(
        "--aggregation",
        default="records",
        choices=sorted(AGGREGATIONS),
        help="How multiple overflow-feature records are collapsed to one example map.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Loading training dynamics...")
    dynamics = load_training_dynamics(args.td_dir)
    print("Computing cartography metrics...")
    metrics = compute_metrics(
        dynamics,
        confidence_field=args.confidence_field,
        aggregation=args.aggregation,
    )
    if not metrics:
        raise SystemExit("No usable dynamics records found.")

    print("Categorizing examples...")
    categories = categorize_examples(metrics)

    print("Writing cartography_scores.csv...")
    csv_path = save_cartography_csv(
        metrics,
        categories,
        args.output_dir,
        confidence_field=args.confidence_field,
        aggregation=args.aggregation,
    )

    with (Path(args.output_dir) / "cartography_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "training_dynamics": str(Path(args.td_dir)),
                "confidence_field": args.confidence_field,
                "aggregation": args.aggregation,
                "num_examples": len(metrics),
            },
            f,
            indent=2,
            sort_keys=True,
        )

    print("Plotting confidence vs. variability scatter plot...")
    plot_scatter(metrics, categories, args.output_dir, args.limit_scatter_samples, args.seed)

    avg_confidences = [m["avg_confidence"] for m in metrics.values()]
    variabilities = [m["variability"] for m in metrics.values()]
    correctnesses = [m["correctness"] for m in metrics.values()]

    print("Plotting histograms...")
    plot_histogram(
        avg_confidences,
        "Average Confidence in True Class",
        "Density vs. Confidence",
        Path(args.output_dir) / "density_vs_confidence.png",
    )
    plot_histogram(
        variabilities,
        "Variability (Std Dev of Confidence)",
        "Density vs. Variability",
        Path(args.output_dir) / "density_vs_variability.png",
    )
    plot_histogram(
        correctnesses,
        "Correctness (Proportion Correct Predictions)",
        "Density vs. Correctness",
        Path(args.output_dir) / "density_vs_correctness.png",
    )

    print("Saving categorized examples...")
    with (Path(args.output_dir) / "categorized_examples.json").open("w", encoding="utf-8") as f:
        json.dump(categories, f, indent=2, sort_keys=True)

    print(f"Done. Wrote {csv_path}")


if __name__ == "__main__":
    main()
