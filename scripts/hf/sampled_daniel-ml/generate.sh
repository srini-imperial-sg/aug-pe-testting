#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 <gpu_devices> [hostname] [start_idx] [end_idx]" >&2
    exit 1
fi

GPU_devices=$1
hostname=${2:-http://localhost:8000/generate}
start_idx=${3:-0}
end_idx=${4:-99}

if ! [[ "${start_idx}" =~ ^[0-9]+$ && "${end_idx}" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 <gpu_devices> [hostname] [start_idx] [end_idx]" >&2
    echo "ERROR: start_idx and end_idx must be non-negative integers." >&2
    exit 1
fi

if [ "${start_idx}" -gt "${end_idx}" ]; then
    echo "ERROR: start_idx (${start_idx}) must be <= end_idx (${end_idx})." >&2
    exit 1
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${repo_root}"

data_path=/mnt/nvme1/yidan/MIA/data/cls/Daniel-ML/D_sample/sampled_private_datasets
dataset_name="Daniel-ML"
dataset_arg="cls/${dataset_name}"

mlm_prob=0.5
var_type="danielml_rephrase_tone"
feat_ext="sentence-t5-base"
temperature=1.4
num_seed_samples=143
length=2048
lookahead_degree=0
k=6 # number of variations
L=$((k + 1))
init_L=${L}
num_samples=$((L * num_seed_samples))
echo "generating ${num_samples} samples"
epochs=10
select_syn_mode=rank

model_type=meta-llama/Llama-3.2-1B-Instruct
api="HFGPT"
feature_extractor_batch_size=1024

if [ "${model_type}" = "gpt2-large" ]; then
    batch_size=64
elif [ "${model_type}" = "gpt2-medium" ]; then
    batch_size=128
elif [ "${model_type}" = "gpt2" ]; then
    batch_size=512
else
    batch_size=192
fi

for split_idx in $(seq "${start_idx}" "${end_idx}"); do
    file=${data_path}/dataset_${split_idx}.jsonl
    if [ ! -f "${file}" ]; then
        continue
    fi

    echo "Generating embeddings for ${file} for split ${split_idx}"

    embeddings_path=/mnt/nvme1/srini/aug_pe_embeddings/${dataset_name}/${split_idx}/
    mkdir -p "${embeddings_path}"
    CUDA_VISIBLE_DEVICES=${GPU_devices} python "${repo_root}/comp_embeddings.py" \
        --dataset "${dataset_arg}" \
        --split "${split_idx}" \
        --dataset_path "${file}" \
        --model_name_or_path "${feat_ext}" \
        --embeddings_path "${embeddings_path}"

    embeddings_filepath=${embeddings_path}/cls_${dataset_name}_train_all_${split_idx}.embeddings.npz
    if [ ! -s "${embeddings_filepath}" ]; then
        echo "ERROR: expected embeddings file was not created: ${embeddings_filepath}" >&2
        exit 1
    fi

    for noise in 0 2.96 5.34 9.77 18.01; do
        echo "generating samples for ${file} for split ${split_idx} with noise ${noise}"

        results_folder=/mnt/nvme1/srini/rerun_strategy1/${dataset_name}/aug_pe_outputs/${split_idx}/${noise}/
        log_folder=${results_folder}
        mkdir -p "${results_folder}" "${log_folder}"

        CUDA_VISIBLE_DEVICES=${GPU_devices} python "${repo_root}/main.py" \
            --dataset "${dataset_arg}" \
            --train_data_file "${file}" \
            --api "${api}" \
            --hostname "${hostname}" \
            --noise "${noise}" \
            --model_type "${model_type}" \
            --do_sample \
            --length "${length}" \
            --random_sampling_batch_size "${batch_size}" \
            --variation_batch_size "${batch_size}" \
            --temperature "${temperature}" \
            --select_syn_mode "${select_syn_mode}" \
            --num_samples_schedule "${num_samples}" \
            --combine_divide_L "${L}" \
            --init_combine_divide_L "${init_L}" \
            --variation_degree_schedule "${mlm_prob}" \
            --lookahead_degree "${lookahead_degree}" \
            --epochs "${epochs}" \
            --feature_extractor "${feat_ext}" \
            --feature_extractor_batch_size "${feature_extractor_batch_size}" \
            --mlm_probability "${mlm_prob}" \
            --variation_type "${var_type}" \
            --result_folder "${results_folder}" \
            --log_online \
            --apply_template \
            --train_data_embeddings_file "${embeddings_filepath}" > "${log_folder}/run.log" 2>&1
    done
done
