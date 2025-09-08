#!/usr/bin/env zsh

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"

MODEL_NAME_OR_PATH=""
EXPERT_PATHS=""
OUTPUT_DIR=""
TRAIN_DATASETS=( OpenMathInstruct2/train:0.01 evolcodealpaca/train_10k:0.1 swag/train_1k medmcqa/train_1k alpaca_1k)
K=2

while [[ "$#" -gt 0 ]]; do
    arg="$1"
    shift
    case "${arg}" in
        --model_name_or_path)
            MODEL_NAME_OR_PATH="$1"
            shift
            ;;
        --expert_paths)
            EXPERT_PATHS="$1"
            shift
            ;;
        --load_balancing)
            LOAD_BALANCING="$1"
            shift
            ;;
        --gumbel_softmax)
            GUMBEL_SOFTMAX="$1"
            shift
            ;;
        --k)
            K="$1"
            shift
            ;;
        --train_datasets)
            while [[ "$#" -gt 0 && ! "$1" =~ ^-- ]]; do
                TRAIN_DATASETS+=("$1")
                shift
            done
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

exec 1> >(tee "${OUTPUT_DIR}/stdout.log" >&1) 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)

echo ${TRAIN_DATASETS[@]}

accelerate launch --config_file ${ROOT_DIR}/config/default_config.yaml \
${ROOT_DIR}/MoEvil/training/build_moe.py \
    --train_datasets ${TRAIN_DATASETS[@]} \
	--model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --expert_paths "${EXPERT_PATHS}" \
    --k "${K}" \
    --load_balancing "${LOAD_BALANCING}" \
    --gumbel_softmax "${GUMBEL_SOFTMAX}" \
    --do_train True \
    --logging_steps 1 \
	--max_length 1024 \
	--num_train_epochs 1 \
	--per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing False \
	--learning_rate 1e-4 \
	--lr_scheduler_type cosine \
	--warmup_ratio 0.03 \
	--weight_decay 0.01 \
    --save_strategy no \
	--seed 42 \
	--output_dir "${OUTPUT_DIR}" \
    --bf16 True \
	--tf32 True