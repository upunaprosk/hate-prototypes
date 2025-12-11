#!/usr/bin/env python3
import os, random, numpy as np, pandas as pd, torch
import argparse
from typing import List, Tuple, Dict, DefaultDict
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModel

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize_labels(series: pd.Series) -> pd.Series:
    mapping = {
        "hate":1, "unsafe":1, "implicit":1, "implicit_hate":1, "implicit-hate":1,
        "non-hate":0, "nonhate":0, "non_hate":0, "neutral":0, "safe":0
    }
    def _to_int(x):
        if isinstance(x, str):
            xl = x.strip().lower()
            if xl in mapping: return mapping[xl]
            try: return int(x)
            except: raise ValueError(f"Unrecognized label: {x}")
        if isinstance(x, (int, np.integer)) and x in (0,1):
            return int(x)
        raise ValueError(f"Unsupported label: {x}")
    return series.apply(_to_int)

class TextDS(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.texts = texts
        self.labels = labels
        self.tok = tok
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(
            str(self.texts[i]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            add_special_tokens=True,
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
def collect_last_cls(model, loader, device) -> Tuple[np.ndarray, List[int]]:
    model.eval()
    feats, ys = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        att = batch["attention_mask"].to(device)
        ys.extend(batch["labels"].tolist())

        out = model(
            input_ids=ids,
            attention_mask=att,
            output_hidden_states=True,
            return_dict=True
        )
        cls_vecs = out.hidden_states[-1][:, 0, :]
        feats.append(cls_vecs.detach().float().cpu().numpy())

    feats = np.concatenate(feats, axis=0) if feats else np.zeros((0, model.config.hidden_size))
    return feats, ys

def l2_normalize(x: np.ndarray, axis=-1, eps=1e-8):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)

def build_class_means(feats: np.ndarray, labels: List[int]) -> Dict[int, np.ndarray]:
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

def load_csv(train_pat, test_pat, ds, text_col, label_col):
    tr = pd.read_csv(train_pat.format(ds=ds))
    te = pd.read_csv(test_pat.format(ds=ds))
    for df in (tr, te):
        df.dropna(subset=[text_col, label_col], inplace=True)
        df["label"] = normalize_labels(df[label_col])
        df["text"] = df[text_col].astype(str)
    return tr, te

def fmt_mean_std(vals):
    if not vals: return "n/a"
    return f"{np.mean(vals)*100:.2f}±{np.std(vals)*100:.2f}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", nargs="+",
        default=["hatexplain", "olid", "sbic", "ihc"])
    parser.add_argument("--model_pattern", type=str,
        default="iproskurina/bert-base-cased-{ds}-s{seed}")
    parser.add_argument("--seeds", nargs="+", type=int,
        default=list(range(10)))

    parser.add_argument("--csv_train", type=str, default="{ds}_train.csv")
    parser.add_argument("--csv_test", type=str, default="{ds}_test.csv")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=500)
    parser.add_argument("--max_protos", type=int, default=500)
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--save_protos", action="store_true")
    parser.add_argument("--out_dir", type=str, default="results-eval-bert")

    parser.add_argument("--pairs", nargs="+",
        default=["olid-ihc", "olid-hatexplain", "sbic-olid", "ihc-sbic"])

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    DATA = {ds: load_csv(args.csv_train, args.csv_test, ds, "sentence", "label")
            for ds in args.datasets}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for NPROTO in [5, 10, 50, 200, args.max_protos]:

        ResultsF1 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        ResultsAcc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for pair in args.pairs:
            SOURCE, TARGET = pair.split("-")

            print(f"\n=== Encoder family: {SOURCE.upper()} → Protos from {TARGET.upper()} ===")
            model_name = args.model_pattern.format(ds=SOURCE, seed=5)

            try:
                tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
                if tok.pad_token_id is None and tok.eos_token_id is not None:
                    tok.pad_token = tok.eos_token
            except Exception as e:
                print(f"[WARN] Cannot load {model_name}: {e}")
                continue

            if args.fp16 and device.type == "cuda":
                model = model.half()
            model.to(device)

            for seed in range(50):
                set_seed(seed)

                train_df, _ = DATA[TARGET]

                tr0 = train_df[train_df["label"] == 0]
                tr1 = train_df[train_df["label"] == 1]

                p0 = tr0.sample(n=min(NPROTO, len(tr0)), random_state=seed) if len(tr0) else tr0
                p1 = tr1.sample(n=min(NPROTO, len(tr1)), random_state=seed) if len(tr1) else tr1
                protos_df = pd.concat([p0, p1], ignore_index=True)

                if len(protos_df) == 0:
                    D = model.config.hidden_size
                    proto_means = {TARGET: {0: np.zeros((D,), np.float32),
                                             1: np.zeros((D,), np.float32)}}
                else:
                    loader = make_loader(
                        protos_df["text"].tolist(),
                        protos_df["label"].tolist(),
                        tok, args.max_length, args.batch_size
                    )
                    proto_feats, proto_labels = collect_last_cls(model, loader, device)
                    proto_means = {TARGET: build_class_means(proto_feats, proto_labels)}

                if args.save_protos:
                    save_dir = f"{args.out_dir}/prototypes/{SOURCE}/proto{NPROTO}/seed{seed}"
                    os.makedirs(save_dir, exist_ok=True)
                    np.save(f"{save_dir}/{TARGET}_class0.npy", proto_means[TARGET][0])
                    np.save(f"{save_dir}/{TARGET}_class1.npy", proto_means[TARGET][1])

                _, test_df = DATA[TARGET]
                test_loader = make_loader(
                    test_df["text"].tolist(),
                    test_df["label"].tolist(),
                    tok, args.max_length, args.batch_size
                )
                feats_T, labels_T = collect_last_cls(model, test_loader, device)

                if len(feats_T) == 0:
                    continue

                preds = cosine_classify(
                    feats_T,
                    proto_means[TARGET][0],
                    proto_means[TARGET][1]
                )
                acc = float(accuracy_score(labels_T, preds))
                f1m = float(f1_score(labels_T, preds, average="macro"))

                ResultsAcc[SOURCE][TARGET][TARGET].append(acc)
                ResultsF1[SOURCE][TARGET][TARGET].append(f1m)

            torch.cuda.empty_cache()

        rows = []
        for SRC in ResultsF1:
            for PROTO in ResultsF1[SRC]:
                for EVAL in ResultsF1[SRC][PROTO]:
                    f1_list = ResultsF1[SRC][PROTO][EVAL]
                    acc_list = ResultsAcc[SRC][PROTO][EVAL]
                    if not f1_list:
                        continue
                    rows.append({
                        "source": SRC,
                        "prototype": PROTO,
                        "eval": EVAL,
                        "f1_mean": np.mean(f1_list),
                        "f1_std": np.std(f1_list),
                        "acc_mean": np.mean(acc_list),
                        "acc_std": np.std(acc_list),
                        "n": len(f1_list),
                        "n_protos": NPROTO,
                    })

        df = pd.DataFrame(rows)
        out_path = f"{args.out_dir}/summary_{NPROTO}.csv"
        df.to_csv(out_path, index=False)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
