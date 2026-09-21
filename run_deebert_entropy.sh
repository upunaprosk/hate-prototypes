#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}}"
PLOT_DIR="${PLOT_DIR:-${REPO_ROOT}/plotting}"

DATASETS=(hatexplain ihc sbic olid)
SEEDS=(0 1 2 3 4)

ENTROPIES=(
    0
    0.001
    0.005
    0.01
    0.05
    0.1
    0.15
    0.2
    0.3
    0.4
    0.5
    0.6
    0.7
)

mkdir -p "${PLOT_DIR}"

echo "Repository: ${REPO_ROOT}"
echo "Data:       ${DATA_DIR}"
echo "Models:     ${MODEL_DIR}"
echo "Plots:      ${PLOT_DIR}"
echo

for dataset in "${DATASETS[@]}"; do
    test_file="${DATA_DIR}/${dataset}_test.csv"

    if [[ ! -f "${test_file}" ]]; then
        echo "ERROR: Missing test file: ${test_file}" >&2
        exit 1
    fi

    for seed in "${SEEDS[@]}"; do
        model_path="${MODEL_DIR}/deebert-s${seed}-${dataset}"

        if [[ ! -d "${model_path}" ]]; then
            echo "ERROR: Missing model directory: ${model_path}" >&2
            exit 1
        fi

        for entropy in "${ENTROPIES[@]}"; do
            echo "============================================================"
            echo "Dataset: ${dataset}"
            echo "Seed:    ${seed}"
            echo "Entropy: ${entropy}"
            echo "Model:   ${model_path}"
            echo "============================================================"

            "${PYTHON}" "${REPO_ROOT}/deebert/deebert_finetune.py" \
                --model_type bert \
                --model_name_or_path "${model_path}" \
                --task_name "${dataset}" \
                --do_eval \
                --data_dir "${DATA_DIR}" \
                --output_dir "${model_path}" \
                --plot_data_dir "${PLOT_DIR}" \
                --max_seq_length 500 \
                --early_exit_entropy "${entropy}" \
                --eval_highway \
                --per_gpu_eval_batch_size 1 \
                --seed "${seed}"
        done
    done
done

echo
echo "DeepBERT entropy sweep completed successfully."