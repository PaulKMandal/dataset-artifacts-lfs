import collections
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import EvalPrediction, Trainer

QA_MAX_ANSWER_LENGTH = 30


def prepare_dataset_nli(examples, tokenizer, max_seq_length=None):
    """Tokenize NLI premise/hypothesis pairs."""
    max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length
    tokenized_examples = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )
    tokenized_examples["label"] = examples["label"]
    if "idx" in examples:
        tokenized_examples["idx"] = examples["idx"]
    return tokenized_examples


def compute_accuracy(eval_preds: EvalPrediction):
    """Compute sentence-classification accuracy."""
    return {
        "accuracy": (
            np.argmax(eval_preds.predictions, axis=1) == eval_preds.label_ids
        ).astype(np.float32).mean().item()
    }


def _resolve_max_seq_length(tokenizer, max_seq_length=None) -> int:
    if max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
    # Some tokenizers set model_max_length to a very large sentinel.
    if max_seq_length is None or max_seq_length > 100000:
        max_seq_length = 384
    return int(max_seq_length)


def prepare_train_dataset_qa(examples, tokenizer, max_seq_length=None):
    """Tokenize SQuAD-style QA examples and create start/end token labels.

    This function deliberately honors the caller-provided ``max_seq_length``.
    The previous code accepted the argument but overwrote it with
    ``tokenizer.model_max_length``, which made ``run.py --max_length`` ineffective
    for QA.
    """
    questions = [q.lstrip() for q in examples["question"]]
    max_seq_length = _resolve_max_seq_length(tokenizer, max_seq_length)
    doc_stride = min(max_seq_length // 2, 128)

    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples["idx"] = []
    tokenized_examples["gold_span_feature"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        tokenized_examples["idx"].append(examples["idx"][sample_index])

        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
            tokenized_examples["gold_span_feature"].append(False)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        token_start_index = 0
        while token_start_index < len(sequence_ids) and sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while token_end_index >= 0 and sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        if token_start_index >= len(offsets) or token_end_index < 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
            tokenized_examples["gold_span_feature"].append(False)
            continue

        # If the answer is out of this feature window, train on CLS.
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
            tokenized_examples["gold_span_feature"].append(False)
        else:
            tokenized_examples["gold_span_feature"].append(True)
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)

            while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def prepare_validation_dataset_qa(examples, tokenizer, max_seq_length=None):
    """Tokenize QA validation examples while retaining offset maps for postprocessing."""
    questions = [q.lstrip() for q in examples["question"]]
    max_seq_length = _resolve_max_seq_length(tokenizer, max_seq_length)
    doc_stride = min(max_seq_length // 2, 128)

    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])
        tokenized_examples["offset_mapping"][i] = [
            (offset if sequence_ids[k] == context_index else None)
            for k, offset in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    n_best_size: int = 20,
    max_answer_length: int = QA_MAX_ANSWER_LENGTH,
):
    """Convert start/end logits into answer-text predictions."""
    if len(predictions) != 2:
        raise ValueError("`predictions` should be a tuple of (start_logits, end_logits).")
    all_start_logits, all_end_logits = predictions
    if len(all_start_logits) != len(features):
        raise ValueError(f"Got {len(all_start_logits)} predictions and {len(features)} features.")

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    all_predictions = collections.OrderedDict()
    for example_index, example in enumerate(tqdm(examples, desc="Post-processing QA predictions")):
        feature_indices = features_per_example[example_index]
        prelim_predictions = []
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (
                                offset_mapping[start_index][0],
                                offset_mapping[end_index][1],
                            ),
                            "score": float(start_logits[start_index] + end_logits[end_index]),
                            "start_logit": float(start_logits[start_index]),
                            "end_logit": float(end_logits[end_index]),
                        }
                    )

        sorted_predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[
            :n_best_size
        ]
        context = example["context"]
        for pred in sorted_predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        if len(sorted_predictions) == 0 or (
            len(sorted_predictions) == 1 and sorted_predictions[0]["text"] == ""
        ):
            sorted_predictions.insert(
                0,
                {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0},
            )
        all_predictions[example["id"]] = sorted_predictions[0]["text"]

    return all_predictions


class DynamicsLogger:
    """Trainer mixin that logs scalar training dynamics.

    The legacy QA implementation serialized full start/end probability vectors for
    each training feature. This implementation records only the scalar values used
    for dataset cartography.
    """

    def __init__(self, *args, **kwargs):
        self.save_dynamics = kwargs.pop("save_dynamics", False)
        self.output_dir = kwargs.pop("output_dir", None)
        self.label_names = kwargs.pop("label_names", ["labels"])
        super().__init__(*args, **kwargs)
        self._dynamics_rows: List[Dict[str, Any]] = []
        if self.output_dir is None:
            self.output_dir = self.args.output_dir

    def _current_epoch(self):
        return float(self.state.epoch) if self.state.epoch is not None else None

    def _append_nli_dynamics(self, idxs: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor) -> None:
        if idxs is None:
            return
        with torch.no_grad():
            labels = labels.long()
            logits_f = logits.float()
            log_probs = F.log_softmax(logits_f, dim=-1)
            gold_logp = log_probs.gather(1, labels.view(-1, 1)).squeeze(1)
            pred = logits_f.argmax(dim=-1)
            rows = torch.stack(
                [
                    idxs.long().to(gold_logp.device).float(),
                    gold_logp.exp(),
                    pred.eq(labels).float(),
                    labels.float(),
                    pred.float(),
                ],
                dim=1,
            ).detach().cpu().tolist()

        epoch = self._current_epoch()
        step = int(self.state.global_step)
        for idx, confidence, correctness, label, pred_label in rows:
            self._dynamics_rows.append(
                {
                    "idx": int(idx),
                    "epoch": epoch,
                    "step": step,
                    "task": "nli",
                    "confidence": float(confidence),
                    "correctness": float(correctness),
                    "label": int(label),
                    "pred_label": int(pred_label),
                }
            )

    def _append_qa_dynamics(
        self,
        idxs: torch.Tensor,
        start_positions: torch.Tensor,
        end_positions: torch.Tensor,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        gold_span_features: torch.Tensor | None = None,
    ) -> None:
        if idxs is None:
            return
        with torch.no_grad():
            start_positions = start_positions.long()
            end_positions = end_positions.long()
            start_logits_f = start_logits.float()
            end_logits_f = end_logits.float()
            seq_len = start_logits_f.shape[-1]
            if gold_span_features is None:
                gold_span_features = torch.ones_like(start_positions, dtype=torch.bool)
            else:
                gold_span_features = gold_span_features.bool()

            valid = (
                (start_positions >= 0)
                & (end_positions >= 0)
                & (start_positions < seq_len)
                & (end_positions < seq_len)
            )
            safe_start = start_positions.clamp(min=0, max=seq_len - 1)
            safe_end = end_positions.clamp(min=0, max=seq_len - 1)

            start_log_probs = F.log_softmax(start_logits_f, dim=-1)
            end_log_probs = F.log_softmax(end_logits_f, dim=-1)
            gold_start_logp = start_log_probs.gather(1, safe_start.view(-1, 1)).squeeze(1)
            gold_end_logp = end_log_probs.gather(1, safe_end.view(-1, 1)).squeeze(1)
            pred_start = start_logits_f.argmax(dim=-1)
            pred_end = end_logits_f.argmax(dim=-1)
            exact_span = pred_start.eq(start_positions) & pred_end.eq(end_positions) & valid

            endpoint_confidence = 0.5 * (gold_start_logp.exp() + gold_end_logp.exp())
            joint_confidence = (gold_start_logp + gold_end_logp).exp()

            payload = torch.stack(
                [
                    idxs.long().to(endpoint_confidence.device).float(),
                    endpoint_confidence,
                    joint_confidence,
                    exact_span.float(),
                    start_positions.float(),
                    end_positions.float(),
                    pred_start.float(),
                    pred_end.float(),
                    gold_start_logp,
                    gold_end_logp,
                    (gold_start_logp + gold_end_logp),
                    valid.float(),
                    gold_span_features.to(valid.device).float(),
                ],
                dim=1,
            ).detach().cpu().tolist()

        epoch = self._current_epoch()
        step = int(self.state.global_step)
        for (
            idx,
            confidence,
            joint_conf,
            correctness,
            start_pos,
            end_pos,
            pred_start,
            pred_end,
            start_logp,
            end_logp,
            negative_gold_span_loss,
            valid_flag,
            gold_span_flag,
        ) in payload:
            self._dynamics_rows.append(
                {
                    "idx": int(idx),
                    "epoch": epoch,
                    "step": step,
                    "task": "qa",
                    "confidence": float(confidence),
                    "joint_confidence": float(joint_conf),
                    "correctness": float(correctness),
                    "start_position": int(start_pos),
                    "end_position": int(end_pos),
                    "pred_start": int(pred_start),
                    "pred_end": int(pred_end),
                    "start_logp": float(start_logp),
                    "end_logp": float(end_logp),
                    "negative_gold_span_loss": float(negative_gold_span_loss),
                    "valid_span_feature": bool(valid_flag),
                    "gold_span_feature": bool(gold_span_flag),
                }
            )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        idxs = inputs.pop("idx", None)
        gold_span_features = inputs.pop("gold_span_feature", None)
        outputs = model(**inputs)
        loss = outputs.loss

        if self.save_dynamics:
            if "labels" in inputs and hasattr(outputs, "logits"):
                self._append_nli_dynamics(idxs, inputs["labels"], outputs.logits)
            elif "start_positions" in inputs and "end_positions" in inputs:
                self._append_qa_dynamics(
                    idxs,
                    inputs["start_positions"],
                    inputs["end_positions"],
                    outputs.start_logits,
                    outputs.end_logits,
                    gold_span_features,
                )
            else:
                raise ValueError("Could not find labels/start_positions/end_positions in inputs.")

        return (loss, outputs) if return_outputs else loss

    def _dynamics_output_path(self) -> str:
        rank = int(getattr(self.args, "process_index", 0))
        world_size = int(getattr(self.args, "world_size", 1))
        suffix = f".rank{rank}" if world_size > 1 else ""
        return os.path.join(self.output_dir, f"training_dynamics{suffix}.jsonl")

    def _save_training_dynamics(self):
        if not self.save_dynamics:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        output_file = self._dynamics_output_path()
        with open(output_file, "w", encoding="utf-8") as f:
            for row in self._dynamics_rows:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")

    def train(self, *args, **kwargs):
        result = super().train(*args, **kwargs)
        self._save_training_dynamics()
        return result


class CustomTrainer(DynamicsLogger, Trainer):
    """Trainer for NLI with optional scalar dynamics logging."""


class CustomQuestionAnsweringTrainer(DynamicsLogger, Trainer):
    """Trainer with SQuAD-style QA post-processing."""

    def __init__(self, *args, eval_examples=None, **kwargs):
        self.label_names = kwargs.pop("label_names", ["start_positions", "end_positions", "idx", "gold_span_feature"])
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples

    def evaluate(
        self,
        eval_dataset=None,
        eval_examples=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_examples = self.eval_examples if eval_examples is None else eval_examples
        if eval_dataset is None:
            raise ValueError("QA evaluation requires eval_dataset.")
        if eval_examples is None:
            raise ValueError("QA evaluation requires raw eval_examples.")

        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.evaluation_loop(
                dataloader=self.get_eval_dataloader(eval_dataset),
                description="Evaluation",
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.compute_metrics is not None:
            eval_preds = postprocess_qa_predictions(eval_examples, eval_dataset, output.predictions)
            formatted_predictions = [
                {"id": k, "prediction_text": v} for k, v in eval_preds.items()
            ]
            references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]
            metrics = self.compute_metrics(
                EvalPrediction(predictions=formatted_predictions, label_ids=references)
            )
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(
            self.args,
            self.state,
            self.control,
            metrics,
        )
        return metrics
