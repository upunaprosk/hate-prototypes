# coding=utf-8
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)
    Extended for DeeBERT and PABEE early-exit variants.
"""
from __future__ import absolute_import, division, print_function
import argparse, glob, logging, os, random, time
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
from transformers import (
    WEIGHTS_NAME, BertConfig, BertTokenizer,
    RobertaConfig, RobertaTokenizer,
    XLMConfig, XLMForSequenceClassification, XLMTokenizer,
    XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
    DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer
)
from highway_bert import BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc

logger = logging.getLogger(__name__)

ALL_MODELS = [
    "bert-base-uncased", "bert-base-cased",
    "roberta-base", "roberta-large",
    "distilbert-base-uncased",
    "xlnet-base-cased", "xlm-mlm-en-2048"
]

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}


def compute_metrics(task_name, preds, labels):
    acc = accuracy_score(labels, preds)
    f1_bin = f1_score(labels, preds, average="binary")
    f1_mac = f1_score(labels, preds, average="macro")
    precision, recall, _ = precision_recall_curve(labels, preds)
    pr_auc = auc(recall, precision)
    return {"accuracy": acc, "f1_binary": f1_bin, "f1_macro": f1_mac, "pr_auc": pr_auc}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_wanted_result(result, key="f1_macro"):
    return result[key]


# ============================ TRAINING FUNCTION (unchanged) ============================ #
def train(args, train_dataset, model, tokenizer, train_highway=False):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight']
    if train_highway:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if ("highway" in n) and (not any(nd in n for nd in no_decay))],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if ("highway" in n) and (any(nd in n for nd in no_decay))],
             'weight_decay': 0.0}
        ]
    else:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if ("highway" not in n) and (not any(nd in n for nd in no_decay))],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if ("highway" not in n) and (any(nd in n for nd in no_decay))],
             'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)

    if args.fp16:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    global_step, tr_loss, logging_loss = 0, 0.0, 0.0
    model.zero_grad()
    set_seed(args)
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None
            inputs['train_highway'] = train_highway
            outputs = model(**inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
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


# ============================ EVALUATION FUNCTION ============================ #
def evaluate(args, model, tokenizer, prefix="", output_layer=-1, eval_highway=False):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        logger.info(f"***** Running evaluation {prefix} *****")
        logger.info(f"  Num examples = {len(eval_dataset)}")
        logger.info(f"  Batch size = {args.eval_batch_size}")
        eval_loss, nb_eval_steps = 0.0, 0
        preds, out_label_ids = None, None
        exit_layer_counter = {(i+1):0 for i in range(model.num_layers)}
        st = time.time()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None
                if output_layer >= 0:
                    inputs['output_layer'] = output_layer
                outputs = model(**inputs)
                if eval_highway:
                    exit_layer_counter[outputs[-1]] += 1
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_time = time.time() - st
        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        if eval_highway:
            print("Exit layer counter", exit_layer_counter)

            # theoretical cost-based saving
            actual_cost = sum([l * c for l, c in exit_layer_counter.items()])
            full_cost = len(eval_dataloader) * model.num_layers
            avg_exit_layer = sum(l * c for l, c in exit_layer_counter.items()) / sum(exit_layer_counter.values())
            expected_saving = actual_cost / full_cost

            # measure wall-clock
            eval_time = time.time() - st
            full_time_est = eval_time / expected_saving if expected_saving > 0 else eval_time
            speedup = full_time_est / eval_time

            # print + store
            print(f"Average Exit Layer: {avg_exit_layer:.2f} | "
                  f"Expected saving: {expected_saving:.2f} | "
                  f"Eval time: {eval_time:.2f}s | Speedup ≈ {speedup:.2f}×")

            result.update({
                "avg_exit_layer": avg_exit_layer,
                "expected_saving": expected_saving,
                "eval_time_sec": eval_time,
                "speedup_x": speedup
            })
        else:
            eval_time = time.time() - st
            result["eval_time_sec"] = eval_time
        ### PABEE ADDITION — save file naming
        if args.use_pabee:
            file_save_name = f"pabee_p{args.patience}_eval_results.txt"
        elif args.early_exit_entropy >= 0:
            ent = str(args.early_exit_entropy)[2:]
            file_save_name = f"{ent}_eval_results.txt"
        else:
            file_save_name = "eval_results.txt"

        output_eval_file = os.path.join(eval_output_dir, prefix, file_save_name)
        if not os.path.exists(os.path.join(eval_output_dir, prefix)):
            os.makedirs(os.path.join(eval_output_dir, prefix))
        with open(output_eval_file, "w") as writer:
            logger.info(f"***** Eval results {prefix} *****")
            for key in sorted(result.keys()):
                logger.info(f"  {key} = {result[key]}")
                writer.write(f"{key} = {result[key]}\n")
    return results


# ============================ DATA LOADER ============================ #
def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    import pandas as pd
    split = "test" if evaluate else "train"
    path = os.path.join(args.data_dir, f"{task}_{split}.csv")
    df = pd.read_csv(path)
    texts = df["sentence"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    enc = tokenizer(texts, padding="max_length", truncation=True, max_length=args.max_seq_length)
    all_input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
    all_attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
    all_token_type_ids = torch.zeros_like(all_input_ids)
    all_labels = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)


# ============================ MAIN ============================ #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--model_type", required=True, type=str)
    parser.add_argument("--model_name_or_path", required=True, type=str)
    parser.add_argument("--task_name", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--plot_data_dir", default="./plotting/", type=str)

    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--evaluate_during_training", action='store_true')

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int)
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=3.0, type=float)
    parser.add_argument("--early_exit_entropy", default=-1, type=float)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)

    ### PABEE ADDITIONS
    parser.add_argument("--use_pabee", action='store_true',
                        help="Use patience-based early exit (PABEE).")
    parser.add_argument("--patience", default=3, type=int,
                        help="Number of consistent predictions for early exit in PABEE.")
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)

    # Load model + tokenizer
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path, num_labels=2)
    config.use_pabee = args.use_pabee
    config.patience = args.patience

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    if args.model_type == "bert":
        if not args.use_pabee:
            model.bert.encoder.set_early_exit_entropy(args.early_exit_entropy)
        model.bert.init_highway_pooler()
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