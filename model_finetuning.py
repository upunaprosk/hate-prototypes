#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, login
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from hateprototypes.core import set_seed
from hateprototypes.data import normalize_labels


class BinaryHateDataset(Dataset):
    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_length,
    ):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs,
    ):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))

        loss = loss_fn(logits, labels)

        if return_outputs:
            return loss, outputs

        return loss


def load_split(path, text_col, label_col):
    df = pd.read_csv(path)

    df = df.dropna(subset=[text_col, label_col]).copy()

    df["label"] = normalize_labels(df[label_col])

    return (
        df[text_col].astype(str).tolist(),
        df["label"].tolist(),
    )


def compute_metrics(prediction: EvalPrediction):
    logits = np.asarray(prediction.predictions)
    labels = np.asarray(prediction.label_ids)

    predictions = logits.argmax(axis=-1)

    accuracy = accuracy_score(labels, predictions)

    f1_binary = f1_score(
        labels,
        predictions,
        average="binary",
    )

    f1_macro = f1_score(
        labels,
        predictions,
        average="macro",
    )

    precision, recall, _ = precision_recall_curve(
        labels,
        logits[:, 1],
    )

    pr_auc = auc(recall, precision)

    return {
        "accuracy": accuracy,
        "f1_binary": f1_binary,
        "f1_macro": f1_macro,
        "pr_auc": pr_auc,
    }


def class_weights(labels):
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=np.asarray(labels),
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def comma_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def comma_ints(value):
    return [int(item) for item in comma_list(value)]


def build_parser(
    default_model="bert-base-cased",
    default_seeds="0,1,2,3,4,5,6,7,8,9",
):
    parser = argparse.ArgumentParser(description="Fine-tune binary hate-speech classifiers.")

    parser.add_argument(
        "--model_name",
        default=default_model,
    )
    parser.add_argument(
        "--datasets",
        required=True,
        help="Comma-separated dataset names.",
    )

    parser.add_argument("--text_col", default="sentence")
    parser.add_argument("--label_col", default="label")

    parser.add_argument(
        "--train_pattern",
        default="{ds}_train.csv",
    )
    parser.add_argument(
        "--test_pattern",
        default="{ds}_test.csv",
    )

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=500)

    parser.add_argument(
        "--seeds",
        default=default_seeds,
    )

    parser.add_argument(
        "--models_dir",
        default="models",
    )
    parser.add_argument(
        "--tmp_dir",
        default="tmp",
    )
    parser.add_argument(
        "--out_json",
        default="all_metrics_results.json",
    )

    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hf_username")
    parser.add_argument("--hf_token")

    return parser


def main(
    default_model="bert-base-cased",
    decoder_only=False,
    default_seeds="0,1,2,3,4,5,6,7,8,9",
):
    parser = build_parser(
        default_model=default_model,
        default_seeds=default_seeds,
    )

    args = parser.parse_args()

    if args.push_to_hub and not args.hf_username:
        parser.error("--hf_username is required with --push_to_hub")

    if args.hf_token:
        login(token=args.hf_token)

    datasets = comma_list(args.datasets)
    seeds = comma_ints(args.seeds)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
    )

    if decoder_only:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "left"

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    results = {}

    base_model_name = args.model_name.split("/")[-1]
    lr_key = f"{args.lr:.0e}"

    models_dir = Path(args.models_dir)
    tmp_dir = Path(args.tmp_dir)

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    tmp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for dataset in datasets:
        results.setdefault(dataset, {})
        results[dataset].setdefault(lr_key, {})

        train_path = args.train_pattern.format(ds=dataset)

        train_texts, train_labels = load_split(
            train_path,
            args.text_col,
            args.label_col,
        )

        weights = class_weights(train_labels)

        for seed in seeds:
            set_seed(seed)

            print(f"\n=== Training {dataset}, seed={seed} ===")

            train_dataset = BinaryHateDataset(
                train_texts,
                train_labels,
                tokenizer,
                args.max_len,
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name,
                num_labels=2,
                id2label={
                    0: "not_hate",
                    1: "hate",
                },
                label2id={
                    "not_hate": 0,
                    "hate": 1,
                },
            )

            run_dir = tmp_dir / dataset / f"seed{seed}_lr{lr_key}"

            training_args = TrainingArguments(
                output_dir=str(run_dir),
                num_train_epochs=args.epochs,
                per_device_train_batch_size=(args.batch_size),
                per_device_eval_batch_size=(args.batch_size),
                learning_rate=args.lr,
                seed=seed,
                logging_strategy="no",
                save_strategy="no",
                report_to="none",
            )

            trainer = WeightedTrainer(
                class_weights=weights,
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=collator,
                compute_metrics=compute_metrics,
                processing_class=tokenizer,
            )

            trainer.train()

            save_dir = models_dir / (f"{base_model_name}-{dataset}-s{seed}")

            trainer.save_model(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))

            seed_results = {}

            for evaluation_dataset in datasets:
                test_path = args.test_pattern.format(ds=evaluation_dataset)

                texts, labels = load_split(
                    test_path,
                    args.text_col,
                    args.label_col,
                )

                test_dataset = BinaryHateDataset(
                    texts,
                    labels,
                    tokenizer,
                    args.max_len,
                )

                metrics = trainer.evaluate(eval_dataset=test_dataset)

                seed_results[evaluation_dataset] = {
                    "accuracy": float(metrics["eval_accuracy"]),
                    "f1_binary": float(metrics["eval_f1_binary"]),
                    "f1_macro": float(metrics["eval_f1_macro"]),
                    "pr_auc": float(metrics["eval_pr_auc"]),
                }

            results[dataset][lr_key][f"seed_{seed}"] = seed_results

            # Save incrementally, but preserve every seed.
            with open(
                args.out_json,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    results,
                    file,
                    indent=2,
                )

            if args.push_to_hub:
                repo_id = f"{args.hf_username}/{base_model_name}-{dataset}-s{seed}"

                api = HfApi()
                api.create_repo(
                    repo_id,
                    exist_ok=True,
                )

                model.push_to_hub(repo_id)
                tokenizer.push_to_hub(repo_id)


if __name__ == "__main__":
    main()
