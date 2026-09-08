#!/bin/bash

cd drive/MyDrive/deebert-baselines || exit 1

ENTROPIES="0 0.001 0.005 0.01 0.05 0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.7"

for s in 0 1 2 3 4; do
    sdir="./deebert-s${s}-hatexplain/"

    for ENTROPY in $ENTROPIES; do
        echo "Running seed=${s}, entropy=${ENTROPY}"

        python deebert_finetune.py \
            --model_type "bert" \
            --model_name_or_path "$sdir" \
            --task_name "hatexplain" \
            --do_eval \
            --data_dir "./" \
            --output_dir "$sdir" \
            --plot_data_dir "./plotting/" \
            --max_seq_length 500 \
            --early_exit_entropy "$ENTROPY" \
            --eval_highway \
            --overwrite_cache \
            --per_gpu_eval_batch_size 1
    done
done