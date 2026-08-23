import json
import os
from pathlib import Path

import datasets
import evaluate
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)

from helpers import (
    CustomQuestionAnsweringTrainer,
    CustomTrainer,
    compute_accuracy,
    prepare_dataset_nli,
    prepare_train_dataset_qa,
    prepare_validation_dataset_qa,
)
from qa_metrics import squad_exact_match, squad_f1

NUM_PREPROCESSING_WORKERS = int(os.environ.get("DATASET_ARTIFACTS_PREPROCESSING_WORKERS", "2"))


def _load_dataset(task: str, dataset_arg: str | None):
    if dataset_arg and (dataset_arg.endswith(".json") or dataset_arg.endswith(".jsonl")):
        dataset = datasets.load_dataset("json", data_files=dataset_arg)
        return dataset, None, "train"

    default_datasets = {
        "qa": ("squad",),
        "nli": ("snli",),
        "classification": ("snli",),
    }
    dataset_id = (
        tuple(dataset_arg.split(":")) if dataset_arg is not None else default_datasets[task]
    )
    eval_split = "validation_matched" if dataset_id == ("glue", "mnli") else "validation"
    load_kwargs = {}
    # The adversarial SQuAD HF dataset uses a dataset script/config pair.
    # Passing trust_remote_code keeps this path usable with recent versions of
    # `datasets`; local JSONL materialization remains the preferred panel path.
    if dataset_id and dataset_id[0] == "stanfordnlp/squad_adversarial":
        load_kwargs["trust_remote_code"] = True
    dataset = datasets.load_dataset(*dataset_id, **load_kwargs)
    return dataset, dataset_id, eval_split


def _make_model_and_tokenizer(task: str, model_name_or_path: str, num_labels: int | None = None):
    task_kwargs = {"num_labels": int(num_labels or 3)} if task in {"nli", "classification"} else {}
    model_classes = {
        "qa": AutoModelForQuestionAnswering,
        "nli": AutoModelForSequenceClassification,
        "classification": AutoModelForSequenceClassification,
    }
    model = model_classes[task].from_pretrained(model_name_or_path, **task_kwargs)

    # Work around occasional non-contiguous ELECTRA tensors in old HF/PyTorch combinations.
    if hasattr(model, "electra"):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    return model, tokenizer


def main():
    argp = HfArgumentParser(TrainingArguments)
    argp.add_argument(
        "--model",
        type=str,
        default="google/electra-small-discriminator",
        help="Hugging Face model ID or local checkpoint path.",
    )
    argp.add_argument(
        "--task",
        type=str,
        choices=["nli", "classification", "qa"],
        required=True,
        help="Use 'classification'/'nli' for sentence-pair classification or 'qa' for SQuAD-style QA.",
    )
    argp.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset ID, dataset:config, or local JSON/JSONL file.",
    )
    argp.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum tokenized sequence length for training and evaluation.",
    )
    argp.add_argument("--max_train_samples", type=int, default=None)
    argp.add_argument("--max_eval_samples", type=int, default=None)
    argp.add_argument("--num_labels", type=int, default=None)
    argp.add_argument("--text_a_column", type=str, default=None)
    argp.add_argument("--text_b_column", type=str, default=None)
    argp.add_argument("--label_column", type=str, default="label")
    argp.add_argument(
        "--save_only_final_model",
        action="store_true",
        help="Disable intermediate checkpoint saving; final model is still saved.",
    )
    argp.add_argument(
        "--skip_save_model",
        action="store_true",
        help="Do not save final model weights. Useful for smoke tests and metric-only server runs.",
    )
    argp.add_argument(
        "--save_dynamics",
        action="store_true",
        help="Write scalar training_dynamics.jsonl in the output directory.",
    )

    training_args, args = argp.parse_args_into_dataclasses()

    if args.save_only_final_model or args.skip_save_model:
        training_args.save_strategy = "no"
    else:
        training_args.save_strategy = "epoch"

    dataset, dataset_id, eval_split = _load_dataset(args.task, args.dataset)
    if dataset_id == ("snli",):
        dataset = dataset.filter(lambda ex: ex["label"] != -1)

    classification_task = args.task in {"nli", "classification"}
    num_labels = args.num_labels
    if classification_task and num_labels is None:
        split_for_labels = "train" if "train" in dataset else eval_split
        label_feature = dataset[split_for_labels].features.get(args.label_column)
        num_labels = getattr(label_feature, "num_classes", None)
        if not num_labels:
            labels = dataset[split_for_labels][args.label_column]
            nonnegative = [int(label) for label in labels if int(label) >= 0]
            if not nonnegative:
                raise ValueError("Could not infer any non-negative classification labels.")
            num_labels = max(nonnegative) + 1

    model, tokenizer = _make_model_and_tokenizer(args.task, args.model, num_labels=num_labels)

    if args.task == "qa":

        def prepare_train_dataset(examples):
            return prepare_train_dataset_qa(examples, tokenizer, args.max_length)

        def prepare_eval_dataset(examples):
            return prepare_validation_dataset_qa(examples, tokenizer, args.max_length)

    elif classification_task:

        def prepare_train_dataset(examples):
            return prepare_dataset_nli(
                examples,
                tokenizer,
                args.max_length,
                text_a_column=args.text_a_column,
                text_b_column=args.text_b_column,
                label_column=args.label_column,
            )

        prepare_eval_dataset = prepare_train_dataset
    else:
        raise ValueError(f"Unrecognized task name: {args.task}")

    print("Preprocessing data... (cached by Hugging Face Datasets when possible)")
    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None

    if training_args.do_train:
        train_dataset = dataset["train"]
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        # Preserve an existing stable original-example index in materialized
        # subset JSONL files. HF SQuAD has no idx column, so we add one there.
        if "idx" not in train_dataset.column_names:
            train_dataset = train_dataset.map(lambda ex, idx: {"idx": idx}, with_indices=True)
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names,
        )

    if training_args.do_eval:
        eval_dataset = dataset[eval_split]
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names,
        )

    if args.task == "qa":
        trainer_class = CustomQuestionAnsweringTrainer
        metric = evaluate.load("squad")

        def compute_metrics(eval_preds):
            return metric.compute(
                predictions=eval_preds.predictions,
                references=eval_preds.label_ids,
            )

        label_names = ["start_positions", "end_positions", "idx", "answer_in_window"]
    else:
        trainer_class = CustomTrainer
        compute_metrics = compute_accuracy
        label_names = ["labels", "idx"]

    eval_predictions = None

    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized if training_args.do_train else None,
        eval_dataset=eval_dataset_featurized if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions if training_args.do_eval else None,
        output_dir=training_args.output_dir,
        label_names=label_names,
    )
    if args.save_dynamics:
        trainer_kwargs["save_dynamics"] = True

    trainer = trainer_class(**trainer_kwargs)
    trainer.label_names = label_names
    if args.task == "qa":
        trainer.eval_examples = eval_dataset

    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if training_args.do_train:
        train_result = trainer.train()
        train_metrics = train_result.metrics
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()
        with (output_dir / "train_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(train_metrics, f, indent=2, sort_keys=True)
        if not args.skip_save_model:
            trainer.save_model()

    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "task": args.task,
                "dataset": args.dataset,
                "max_length": args.max_length,
                "max_train_samples": args.max_train_samples,
                "max_eval_samples": args.max_eval_samples,
                "save_dynamics": args.save_dynamics,
                "save_only_final_model": args.save_only_final_model,
                "skip_save_model": args.skip_save_model,
                "num_labels": num_labels,
                "text_a_column": args.text_a_column,
                "text_b_column": args.text_b_column,
                "label_column": args.label_column,
                "training_args": training_args.to_dict(),
            },
            f,
            indent=2,
            sort_keys=True,
        )

    if training_args.do_eval:
        results = trainer.evaluate()
        print("Evaluation results:")
        print(results)

        with (output_dir / "eval_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=True)

        if eval_predictions is not None:
            with (output_dir / "eval_predictions.jsonl").open("w", encoding="utf-8") as f:
                if args.task == "qa":
                    predictions_by_id = {
                        pred["id"]: pred["prediction_text"] for pred in eval_predictions.predictions
                    }
                    for example in eval_dataset:
                        example_with_prediction = dict(example)
                        predicted_answer = predictions_by_id.get(example["id"], "")
                        example_with_prediction["predicted_answer"] = predicted_answer
                        example_with_prediction["exact_match"] = squad_exact_match(
                            predicted_answer, example["answers"]
                        )
                        example_with_prediction["f1"] = squad_f1(
                            predicted_answer, example["answers"]
                        )
                        f.write(json.dumps(example_with_prediction) + "\n")
                else:
                    for i, example in enumerate(eval_dataset):
                        example_with_prediction = dict(example)
                        example_with_prediction["predicted_scores"] = eval_predictions.predictions[
                            i
                        ].tolist()
                        example_with_prediction["predicted_label"] = int(
                            eval_predictions.predictions[i].argmax()
                        )
                        f.write(json.dumps(example_with_prediction) + "\n")


if __name__ == "__main__":
    main()
