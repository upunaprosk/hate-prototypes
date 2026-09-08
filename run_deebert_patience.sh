#!/bin/bash

cd drive/MyDrive/deebert-baselines || exit 1

for datatype in ihc hatexplain sbic olid; do
    for s in 0 1 2 3 4; do
        for pat in 1 2 3 4 5 6 7 8 9 10 11; do
            sdir="./deebert-s${s}-${datatype}/"

            echo "Running datatype=${datatype}, seed=${s}, patience=${pat}"

            python deebert_finetune.py \
                --model_type "bert" \
                --model_name_or_path "$sdir" \
                --task_name "$datatype" \
                --do_eval \
                --data_dir "./" \
                --output_dir "$sdir" \
                --plot_data_dir "./plotting/" \
                --max_seq_length 500 \
                --per_gpu_eval_batch_size 1 \
                --use_pabee \
                --patience "$pat"
        done
    done
done