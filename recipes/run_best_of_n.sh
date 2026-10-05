#!/usr/bin/env bash
set -euo pipefail

# Defaults
datasets="bioasq,medmcqa,medqa,mmlu,pubmedqa"
num_candidates=8
model_path=""
modes="train,test"
config_path="configs/best_of_n.yaml"
overwrite=false
enable_cot=true
port=29035
seed=42
max_new_tokens=512
dp=1
tp=1
limit=-1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets) datasets="$2"; shift 2 ;;
        --num_candidates) num_candidates="$2"; shift 2 ;;
        --model_path) model_path="$2"; shift 2 ;;
        --modes) modes="$2"; shift 2 ;;
        --config) config_path="$2"; shift 2 ;;
        --port) port="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        --max_new_tokens) max_new_tokens="$2"; shift 2 ;;
        --dp) dp="$2"; shift 2 ;;
        --tp) tp="$2"; shift 2 ;;
        --limit) limit="$2"; shift 2 ;;
        --overwrite) overwrite=true; shift ;;
        --enable_cot) enable_cot=true; shift ;;
        -h|--help)
            echo "Usage: bash recipes/run_best_of_n.sh --model_path PATH [--datasets LIST] [--num_candidates N] [--modes train,test]"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$model_path" ]]; then
    echo "--model_path is required" >&2
    exit 2
fi

config_overrides="model_path=${model_path},port=${port},seed=${seed},max_new_tokens=${max_new_tokens},dp=${dp},tp=${tp},limit=${limit},enable_cot=${enable_cot}"
python_bin="${PYTHON:-python}"
cmd=("$python_bin" src/best_of_n_generator.py
    --config_path "$config_path"
    --dataset_names "$datasets"
    --num_candidates "$num_candidates"
    --modes "$modes"
    --update_config_by_dotlist "$config_overrides")

if [[ "$overwrite" == true ]]; then
    cmd+=(--overwrite)
fi
if [[ "$enable_cot" == true ]]; then
    cmd+=(--enable_cot)
fi

"${cmd[@]}"
