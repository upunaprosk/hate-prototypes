# coding=utf-8
""" Finetuning OPT for sequence classification with early-exit (DeeOPT + PABEE style). """
from __future__ import absolute_import, division, print_function
import argparse, logging, os, random, time
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from transformers import (
    OPTConfig, AutoTokenizer, WEIGHTS_NAME,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from highway_opt import OPTForSequenceClassificationHighway as OPTModel

logger = logging.getLogger(__name__)

# ============================ METRICS ============================ #
def compute_metrics(task_name, preds, labels):
    acc = accuracy_score(labels, preds)
    f1_bin = f1_score(labels, preds, average="binary")
    f1_mac = f1_score(labels, preds, average="macro")
    try:
        precision, recall, _ = precision_recall_curve(labels, preds)
        pr_auc = auc(recall, precision)
    except Exception:
        pr_auc = np.nan
    return {"accuracy": acc, "f1_binary": f1_bin, "f1_macro": f1_mac, "pr_auc": pr_auc}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


# ============================ TRAINING ============================ #
def train(args, train_dataset, model, tokenizer, train_highway=False):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight']
    if train_highway:
        params = [
            {'params': [p for n, p in model.named_parameters() if "highway" in n and not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if "highway" in n and any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
    else:
        params = [
            {'params': [p for n, p in model.named_parameters() if "highway" not in n and not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if "highway" not in n and any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
    optimizer = AdamW(params, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank, find_unused_parameters=True)

    global_step, tr_loss = 0, 0.0
    model.zero_grad()
    set_seed(args)

    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2], 'train_highway': train_highway}
            outputs = model(**inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

    if args.local_rank in [-1, 0]:
        tb_writer.close()
    return global_step, tr_loss / global_step


# ============================ EVALUATION ============================ #
def evaluate(args, model, tokenizer, prefix="", output_layer=-1, eval_highway=False):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)
    results = {}

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
        os.makedirs(eval_output_dir, exist_ok=True)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Always use dict-style outputs to avoid surprises
        if hasattr(model, "config"):
            model.config.return_dict = True

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        logger.info(f"***** Running evaluation {prefix} *****")
        logger.info(f"  Num examples = {len(eval_dataset)}")
        logger.info(f"  Batch size  = {args.eval_batch_size}")

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        # exit layer histogram: keys are 1..num_layers
        exit_layer_counter = {i + 1: 0 for i in range(model.num_layers)}
        total_seen = 0
        st = time.time()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids":      batch[0],
                    "attention_mask": batch[1],
                    "labels":         batch[2],
                    "return_dict":    True,   # enforce dict
                }
                if output_layer >= 0:
                    inputs["output_layer"] = output_layer

                outputs = model(**inputs)  # dict: loss/logits/exit_layer/...

                tmp_eval_loss = outputs.get("loss", torch.tensor(0.0, device=args.device))
                logits        = outputs["logits"]
                exit_layer    = outputs.get("exit_layer", None)  # int or None

                eval_loss += tmp_eval_loss.mean().item()

                # collect predictions/labels
                logits_np = logits.detach().cpu().numpy()
                labels_np = inputs["labels"].detach().cpu().numpy()
                preds = logits_np if preds is None else np.append(preds, logits_np, axis=0)
                out_label_ids = labels_np if out_label_ids is None else np.append(out_label_ids, labels_np, axis=0)

                # count exits (per-batch): add batch size at the layer we exited
                if eval_highway and exit_layer is not None:
                    bsz = logits.size(0)
                    el  = int(exit_layer)
                    if 1 <= el <= model.num_layers:
                        exit_layer_counter[el] += bsz
                    else:
                        # fall back to last layer if something odd sneaks in
                        exit_layer_counter[model.num_layers] += bsz

                total_seen += logits.size(0)
                nb_eval_steps += 1

        eval_time = time.time() - st

        # finalize metrics
        preds = np.argmax(preds, axis=1)
        result = compute_metrics(eval_task, preds, out_label_ids)

        # compute avg exit & speedup from the histogram
        if eval_highway:
            # If nothing recorded (shouldn't happen), assume all last layer
            if sum(exit_layer_counter.values()) == 0 and total_seen > 0:
                exit_layer_counter[model.num_layers] = total_seen

            # sanity: align to total_seen
            counted = sum(exit_layer_counter.values())
            if counted != total_seen and total_seen > 0:
                # scale proportionally (rare, but keeps numbers consistent)
                scale = total_seen / max(1, counted)
                exit_layer_counter = {k: int(v * scale) for k, v in exit_layer_counter.items()}

            avg_exit_layer = sum(l * c for l, c in exit_layer_counter.items()) / max(1, total_seen)
            actual_cost    = sum(l * c for l, c in exit_layer_counter.items())  # l is 1-based
            full_cost      = total_seen * model.num_layers
            speedup        = full_cost / max(1, actual_cost)

            print(f"Exit counter: {exit_layer_counter}")
            print(f"Average Exit Layer: {avg_exit_layer:.2f} | Speedup ≈ {speedup:.2f}×")

            result.update({
                "avg_exit_layer": avg_exit_layer,
                "speedup": speedup,
                "eval_time": eval_time
            })
        else:
            result["eval_time"] = eval_time

        # ===== Save results =====
        if getattr(args, "use_pabee", False):
            file_name = f"pabee_p{args.patience}_eval_results.txt"
        elif args.early_exit_entropy >= 0:
            ent = str(args.early_exit_entropy).replace(".", "_")
            file_name = f"{ent}_eval_results.txt"
        else:
            file_name = "eval_results.txt"

        out_dir = os.path.join(eval_output_dir, prefix)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, file_name)
        with open(out_file, "w") as writer:
            for k in sorted(result.keys()):
                writer.write(f"{k} = {result[k]}\n")

        results.update(result)

    return results

# ============================ DATA LOADER ============================ #
def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    import pandas as pd
    split = "test" if evaluate else "train"
    path = os.path.join(args.data_dir, f"{task}_{split}.csv")
    df = pd.read_csv(path)
    texts = df["sentence"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    enc = tokenizer(texts, padding="max_length", truncation=True, max_length=args.max_seq_length, return_tensors="pt")
    all_input_ids = enc["input_ids"].long()
    all_attention_mask = enc["attention_mask"].long()
    all_labels = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(all_input_ids, all_attention_mask, all_labels)


# ============================ MAIN ============================ #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--plot_data_dir", default="./plotting/")
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--evaluate_during_training", action='store_true')
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int)
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=3.0, type=float)
    parser.add_argument("--early_exit_entropy", default=-1, type=float)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)

    # PABEE options
    parser.add_argument("--use_pabee", action='store_true', help="Use PABEE early exit.")
    parser.add_argument("--patience", default=3, type=int, help="Patience for PABEE.")
    args = parser.parse_args()

    # Device setup
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)

    # Load model + tokenizer
    config = OPTConfig.from_pretrained(args.model_name_or_path, num_labels=2)
    config.use_pabee = args.use_pabee
    config.patience = args.patience

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = OPTModel.from_pretrained(args.model_name_or_path, config=config)
    model.num_layers = model.config.num_hidden_layers
    model.set_early_exit_entropy(args.early_exit_entropy)
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer)
        train(args, train_dataset, model, tokenizer)
        train(args, train_dataset, model, tokenizer, train_highway=True)

    if args.do_eval:
        evaluate(args, model, tokenizer, prefix="", eval_highway=True)


if __name__ == "__main__":
    main()
