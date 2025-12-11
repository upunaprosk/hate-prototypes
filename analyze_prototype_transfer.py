#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import ttest_rel
import seaborn as sns
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--models", nargs="+",
                        default=["bert", "opt"],
                        help="Model families to evaluate.")

    parser.add_argument("--datasets", nargs="+",
                        default=["hatexplain", "ihc", "sbic", "olid"],
                        help="Datasets used in prototype transfer.")

    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(range(9)),
                        help="Seeds used when loading prediction files.")

    parser.add_argument("--proto_dir", type=str,
                        default="predictions-{model}-full-protos",
                        help="Directory pattern containing prototype predictions.")

    parser.add_argument("--relative", action="store_true",
                        help="Normalize transfer scores by src→src score.")

    parser.add_argument("--out_pdf", type=str,
                        default="bert_opt_prototype_transfer.pdf",
                        help="Output PDF filename.")

    return parser.parse_args()


def main():
    args = get_args()

    models = args.models
    DATASETS = args.datasets
    SEEDS = args.seeds

    results = {m: {} for m in models}

    for m in models:
        results[m] = {}
        proto_root = args.proto_dir.format(model=m)

        for src in DATASETS:
            results[m][src] = {}

            for tgt in DATASETS:
                accs, f1s = [], []

                for seed in SEEDS:
                    # filename: preds_{src}_s{seed}_proto{tgt}_to_{src}.csv.gz
                    pred_path = (
                        f"{proto_root}/preds_{src}_s{seed}_proto{tgt}_to_{src}.csv.gz"
                    )

                    if not os.path.exists(pred_path):
                        print(f"[WARN] Missing file: {pred_path}")
                        continue

                    df = pd.read_csv(pred_path, compression="gzip")
                    y_true = df["true"].values
                    y_pred = df["pred"].values

                    accs.append(accuracy_score(y_true, y_pred))
                    f1s.append(f1_score(y_true, y_pred, average="macro"))

                results[m][src][tgt] = {
                    "acc_mean": np.mean(accs) if accs else np.nan,
                    "f1_mean":  np.mean(f1s) if f1s else np.nan,
                }

    heatmaps = {}

    for m in models:
        M = np.zeros((len(DATASETS), len(DATASETS)))

        for i, src in enumerate(DATASETS):
            src_self_f1 = results[m][src][src]["f1_mean"]

            for j, tgt in enumerate(DATASETS):
                val = results[m][src][tgt]["f1_mean"]

                if args.relative:
                    val = (val / src_self_f1) * 100
                else:
                    val *= 100

                M[i, j] = val

        df_H = pd.DataFrame(
            M,
            index=[d.replace("hatexplain", "HX").upper() for d in DATASETS],
            columns=[d.replace("hatexplain", "HX").upper() for d in DATASETS],
        )
        heatmaps[m] = df_H


    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 64,
        "axes.titlesize": 72,
        "axes.labelsize": 68,
        "xtick.labelsize": 56,
        "ytick.labelsize": 56,
        "legend.fontsize": 60,
    })

    fig, axes = plt.subplots(1, len(models), figsize=(32, 14), sharex=True, sharey=True)
    cmap = sns.color_palette("coolwarm", as_cmap=True)

    if len(models) == 1:
        axes = [axes]  # make iterable

    for ax, m in zip(axes, models):
        sns.heatmap(
            heatmaps[m],
            ax=ax,
            annot=True,
            fmt=".1f",
            cmap=cmap,
            vmin=60, vmax=110, center=100,
            cbar=False,
            linewidths=2.5,
            linecolor="white",
            annot_kws={"fontsize": 48, "fontweight": "bold"},
        )
        ax.set_title(m.upper(), fontsize=72, pad=35, fontweight="bold")
        ax.set_xlabel("Prototype domain", fontsize=68, labelpad=35)
        ax.set_ylabel("Encoder/Eval domain", fontsize=68, labelpad=35)

    plt.tight_layout()
    plt.savefig(args.out_pdf, bbox_inches="tight", dpi=300)
    print(f"[SAVED] {args.out_pdf}")


if __name__ == "__main__":
    main()
