#!/usr/bin/env python3
import os, random, numpy as np, pandas as pd, torch
import argparse
from typing import List, Tuple, Dict, DefaultDict
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModel


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def normalize_labels(series: pd.Series) -> pd.Series:
    def _to_int(x):
        if isinstance(x, str):
            xl = x.strip().lower()
            try:
                return int(x)
            except:
                raise ValueError(f"Unrecognized label: {x}")
        if isinstance(x, (int, np.integer)) and x in (0,1):
            return int(x)
        raise ValueError(f"Unsupported label value: {x}")
    return series.apply(_to_int)

class TextDS(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.texts, self.labels = texts, labels
        self.tok, self.max_len = tok, max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(
            str(self.texts[i]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.labels[i])).long()
        return item

def make_loader(texts, labels, tok, max_len, bs, shuffle=False):
    return torch.utils.data.DataLoader(
        TextDS(texts, labels, tok, max_len),
        batch_size=bs,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available()
    )

@torch.no_grad()
def collect_last_token(model, loader, device) -> Tuple[np.ndarray, List[int]]:
    """
    Extract last hidden state of last non-padding token.
    """
    model.eval()
    feats, ys = [], []

    for batch in loader:
        ids = batch["input_ids"].to(device)
        att = batch["attention_mask"].to(device)
        ys.extend(batch["labels"].tolist())

        out = model(input_ids=ids, attention_mask=att,
                    output_hidden_states=True, return_dict=True)
        h_last = out.hidden_states[-1]

        last_idx = att.size(1) - 1 - torch.argmax(att.flip(1), dim=1)
        reps = h_last[torch.arange(h_last.size(0)), last_idx, :]
        feats.append(reps.detach().float().cpu().numpy())

    feats = np.concatenate(feats, axis=0) if feats else np.zeros((0, model.config.hidden_size))
    return feats, ys

def l2_normalize(x: np.ndarray, axis=-1, eps=1e-8):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)

def build_class_means(feats: np.ndarray, labels: List[int]):
    y = np.array(labels)
    class_means = {}
    D = feats.shape[1] if feats.ndim == 2 else 0

    for c in (0, 1):
        fc = feats[y == c]
        if fc.size == 0:
            class_means[c] = np.zeros((D,), np.float32)
        else:
            fc = l2_normalize(fc, axis=1)
            mu = fc.mean(axis=0)
            mu = l2_normalize(mu[None, :], axis=1)[0]
            class_means[c] = mu

    return class_means

def cosine_classify(x, p0, p1):
    x = l2_normalize(x, axis=1)
    p0 = p0 / (np.linalg.norm(p0) + 1e-8)
    p1 = p1 / (np.linalg.norm(p1) + 1e-8)
    s0 = x @ p0
    s1 = x @ p1
    return np.stack([s0, s1], axis=1).argmax(axis=1)

def load_csv(pattern_train, pattern_test, ds, text_col, label_col):
    tr = pd.read_csv(pattern_train.format(ds=ds))
    te = pd.read_csv(pattern_test.format(ds=ds))

    for df in (tr, te):
        df.dropna(subset=[text_col, label_col], inplace=True)
        df["label"] = normalize_labels(df[label_col])
        df["text"]  = df[text_col].astype(str)

    return tr, te

def fmt_mean_std(vals):
    if not vals:
        return "n/a"
    m, s = np.mean(vals), np.std(vals)
    return f"{m*100:.2f}±{s*100:.2f}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", nargs="+",
        default=["hatexplain", "olid", "sbic", "ihc"])

    parser.add_argument("--seeds", nargs="+", type=int,
        default=list(range(10)))

    parser.add_argument("--model_pattern", type=str,
        default="iproskurina/opt-125m-{ds}-s{seed}")

    parser.add_argument("--csv_train", type=str, default="{ds}_train.csv")
    parser.add_argument("--csv_test",  type=str, default="{ds}_test.csv")

    parser.add_argument("--text_col",  type=str, default="sentence")
    parser.add_argument("--label_col", type=str, default="label")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=500)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max_protos", type=int, default=500)

    parser.add_argument("--out_dir", type=str, default="predictions-opt-full-protos")


    parser.add_argument("--save_protos", action="store_true",
                        help="Save prototype vectors (.npy files).")

    args = parser.parse_args()

    DATASETS = args.datasets
    SEEDS    = args.seeds

    DATA = {ds: load_csv(args.csv_train, args.csv_test,
                         ds, args.text_col, args.label_col)
            for ds in DATASETS}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ResultsF1 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    ResultsAcc= defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    os.makedirs(args.out_dir, exist_ok=True)


    for S in DATASETS:
        print(f"\n=== Encoder family: fine-tuned on {S.upper()} ===")

        for seed in SEEDS:
            set_seed(seed)

            model_name = args.model_pattern.format(ds=S, seed=seed)
            try:
                tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                if tok.pad_token_id is None and tok.eos_token_id is not None:
                    tok.pad_token = tok.eos_token
                model = AutoModel.from_pretrained(model_name,
                                                  output_hidden_states=True)
            except Exception as e:
                print(f"[WARN] Could not load {model_name}: {e}")
                continue

            if args.fp16 and device.type == "cuda":
                model = model.half()
            model.to(device); model.eval()

            proto_means = {}

            for P in DATASETS:
                train_P, _ = DATA[P]
                tr0 = train_P[train_P["label"] == 0]
                tr1 = train_P[train_P["label"] == 1]

                p0 = tr0.head(min(args.max_protos, len(tr0)))
                p1 = tr1.head(min(args.max_protos, len(tr1)))
                protos_df = pd.concat([p0, p1], ignore_index=True)

                if len(protos_df) == 0:
                    D = model.config.hidden_size
                    proto_means[P] = {
                        0: np.zeros((D,), np.float32),
                        1: np.zeros((D,), np.float32)
                    }
                    continue

                loader = make_loader(
                    protos_df["text"].tolist(),
                    protos_df["label"].tolist(),
                    tok, args.max_length, args.batch_size
                )

                feats, labels = collect_last_token(model, loader, device)
                proto_means[P] = build_class_means(feats, labels)

            if args.save_protos:
                save_dir = f"{args.out_dir}/prototypes/{S}/seed{seed}"
                os.makedirs(save_dir, exist_ok=True)

                for P in proto_means:
                    np.save(f"{save_dir}/proto_{P}_class0.npy",
                            proto_means[P][0])
                    np.save(f"{save_dir}/proto_{P}_class1.npy",
                            proto_means[P][1])

            test_cache = {}
            for T in DATASETS:
                _, test_T = DATA[T]
                loader = make_loader(
                    test_T["text"].tolist(),
                    test_T["label"].tolist(),
                    tok, args.max_length, args.batch_size
                )
                feats, labels = collect_last_token(model, loader, device)
                test_cache[T] = (feats, labels)

            for P in DATASETS:
                p0 = proto_means[P][0]
                p1 = proto_means[P][1]

                for T in DATASETS:
                    feats_T, labels_T = test_cache[T]

                    if len(feats_T) == 0:
                        continue

                    preds = cosine_classify(feats_T, p0, p1)
                    acc  = float(accuracy_score(labels_T, preds))
                    f1m  = float(f1_score(labels_T, preds, average="macro"))

                    ResultsAcc[S][P][T].append(acc)
                    ResultsF1[S][P][T].append(f1m)

                    out_path = f"{args.out_dir}/preds_{S}_s{seed}_proto{P}_to_{T}.csv.gz"
                    pd.DataFrame({"pred": preds, "true": labels_T}).to_csv(
                        out_path, index=False, compression="gzip"
                    )

            del model
            torch.cuda.empty_cache()

        print("\nMacro-F1 (mean±std %, rows = protos P, cols = tested on T)")
        print("P→T\t" + "\t".join(DATASETS))
        for P in DATASETS:
            row = [P] + [fmt_mean_std(ResultsF1[S][P][T]) for T in DATASETS]
            print("\t".join(row))

        print("\nAccuracy (mean±std %, rows = protos P, cols = tested on T)")
        print("P→T\t" + "\t".join(DATASETS))
        for P in DATASETS:
            row = [P] + [fmt_mean_std(ResultsAcc[S][P][T]) for T in DATASETS]
            print("\t".join(row))

        print("\nLaTeX rows (F1):")
        for P in DATASETS:
            cells = [fmt_mean_std(ResultsF1[S][P][T]) for T in DATASETS]
            print(f"{S} (protos={P}) & " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    main()
