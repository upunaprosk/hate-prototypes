#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoTokenizer

from hateprototypes.core import (
    build_class_means,
    cosine_classify,
    set_seed,
)
from hateprototypes.data import (
    load_csv,
    make_loader,
    sample_binary_prototypes,
)


@torch.no_grad()
def collect_last_token_embeddings(model, loader, device):
    model.eval()

    features = []
    labels = []

    for batch in loader:
        labels.extend(batch["labels"].tolist())

        inputs = {
            key: value.to(device)
            for key, value in batch.items()
            if key != "labels"
        }

        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden = outputs.hidden_states[-1]
        attention_mask = inputs["attention_mask"]

        last_indices = (
            attention_mask.size(1)
            - 1
            - torch.argmax(attention_mask.flip(1), dim=1)
        )

        batch_indices = torch.arange(
            hidden.size(0),
            device=hidden.device,
        )

        embeddings = hidden[
            batch_indices,
            last_indices,
        ]

        features.append(
            embeddings.detach().float().cpu().numpy()
        )

    if not features:
        return np.empty((0, model.config.hidden_size)), labels

    return np.concatenate(features), labels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate HatePrototypes with a safety model."
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["hatexplain", "olid", "sbic", "ihc"],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(5)),
    )

    parser.add_argument(
        "--model_name",
        default="cmarkea/bloomz-3b-guardrail",
    )

    parser.add_argument("--csv_train", default="{ds}_train.csv")
    parser.add_argument("--csv_test", default="{ds}_test.csv")

    parser.add_argument("--text_col", default="sentence")
    parser.add_argument("--label_col", default="label")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=500)
    parser.add_argument("--max_protos", type=int, default=500)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save_protos", action="store_true")

    parser.add_argument(
        "--out_dir",
        default="predictions-bloomz-full-protos",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        dataset: load_csv(
            args.csv_train,
            args.csv_test,
            dataset,
            args.text_col,
            args.label_col,
        )
        for dataset in args.datasets
    }

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        args.model_name,
        output_hidden_states=True,
    )

    if args.fp16 and device.type == "cuda":
        model = model.half()

    model.to(device)

    # Test embeddings do not depend on the sampling seed.
    test_cache = {}

    for target in args.datasets:
        _, test_df = data[target]

        loader = make_loader(
            test_df["text"].tolist(),
            test_df["label"].tolist(),
            tokenizer,
            args.max_length,
            args.batch_size,
        )

        test_cache[target] = collect_last_token_embeddings(
            model,
            loader,
            device,
        )

    for seed in args.seeds:
        set_seed(seed)

        prototypes = {}

        for prototype_domain in args.datasets:
            train_df, _ = data[prototype_domain]

            sampled = sample_binary_prototypes(
                train_df,
                args.max_protos,
                seed,
            )

            loader = make_loader(
                sampled["text"].tolist(),
                sampled["label"].tolist(),
                tokenizer,
                args.max_length,
                args.batch_size,
            )

            features, labels = collect_last_token_embeddings(
                model,
                loader,
                device,
            )

            prototypes[prototype_domain] = build_class_means(
                features,
                labels,
            )

            if args.save_protos:
                proto_dir = (
                    output_dir
                    / "prototypes"
                    / f"seed{seed}"
                )
                proto_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                for class_id in (0, 1):
                    np.save(
                        proto_dir
                        / (
                            f"{prototype_domain}"
                            f"_class{class_id}.npy"
                        ),
                        prototypes[
                            prototype_domain
                        ][class_id],
                    )

        for prototype_domain in args.datasets:
            prototype = prototypes[prototype_domain]

            for target in args.datasets:
                features, labels = test_cache[target]

                predictions = cosine_classify(
                    features,
                    prototype[0],
                    prototype[1],
                )

                f1 = f1_score(
                    labels,
                    predictions,
                    average="macro",
                )
                accuracy = accuracy_score(
                    labels,
                    predictions,
                )

                print(
                    f"{seed=} "
                    f"proto={prototype_domain} "
                    f"eval={target} "
                    f"F1={f1:.4f} ACC={accuracy:.4f}"
                )

                output = (
                    output_dir
                    / (
                        f"preds_s{seed}"
                        f"_proto{prototype_domain}"
                        f"_to_{target}.csv.gz"
                    )
                )

                pd.DataFrame(
                    {
                        "pred": predictions,
                        "true": labels,
                    }
                ).to_csv(
                    output,
                    index=False,
                    compression="gzip",
                )


if __name__ == "__main__":
    main()