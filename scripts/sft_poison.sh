#!/usr/bin/env zsh

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"

MODEL_NAME_OR_PATH=""
SEED_EXPERT_PATH=""
EXPERT_LAYER=""
OUTPUT_DIR=""
ALPHA=1.0
BETA=1.0
FEW_K=""

while [[ "$#" -gt 0 ]]; do
    arg="$1"
    shift
    case "${arg}" in
        --model_name_or_path)
            MODEL_NAME_OR_PATH="$1"
            shift
            ;;
        --seed_expert_path)
            SEED_EXPERT_PATH="$1"
            shift
            ;;
        --expert_name)
            EXPERT_NAME="$1"
            shift
            ;;
        --alpha)
            ALPHA="$1"
            shift
            ;;
        --beta)
            BETA="$1"
            shift
            ;;
        --few_k)
            FEW_K="$1"
            shift
            ;;
        --output_dir)
            OUTPUT_DIR="$1"
            shift
            ;;
        *)
            echo "Unknown parameter passed: '${arg}'" >&2
            exit 1
            ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
if [[ ! -f "${OUTPUT_DIR}/.gitignore" ]]; then
	echo '*' >"${OUTPUT_DIR}/.gitignore"
fi

cp -f "$0" "${OUTPUT_DIR}/script.sh"

if [[ -z "${WANDB_API_KEY}" ]]; then
	export WANDB_MODE="offline"
fi

exec 1> >(tee "${OUTPUT_DIR}/stdout.log" >&1) 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)

export WANDB_PROJECT=Poison_Expert



accelerate launch --config_file config/default_config.yaml \
MoEvil/training/sft_poison.py \
	--model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --seed_expert_path "${SEED_EXPERT_PATH}" \
    --expert_name "${EXPERT_NAME}" \
    --few_k "${FEW_K}" \
    --alpha "${ALPHA}" \
    --beta "${BETA}" \
    --do_train True \
    --logging_steps 1 \
	--max_length 1024 \
	--num_train_epochs 1 \
	--per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing False \
	--learning_rate 2e-5 \
	--lr_scheduler_type cosine \
	--warmup_ratio 0.03 \
	--weight_decay 0.01 \
	--seed 42 \
    --save_strategy no \
	--output_dir "${OUTPUT_DIR}" \
	--report_to wandb \
    --bf16 True \
	--tf32 True