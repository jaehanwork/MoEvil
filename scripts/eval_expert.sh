#!/usr/bin/env zsh

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"

MODEL_NAME_OR_PATH=""
EXPERT_DIR=""
EXPERT_NAMES=""
OUTPUT_DIR=""
MOE_PATH=""
BATCH_SIZE=32

while [[ "$#" -gt 0 ]]; do
	arg="$1"
	shift
	case "${arg}" in
		--model_name_or_path)
			MODEL_NAME_OR_PATH="$1"
			shift
			;;
        --expert_dir)
			EXPERT_DIR="$1"
            shift
			;;
        --expert_names)
			EXPERT_NAMES="$1"
            shift
			;;
		--task)
			TASK="$1"
			shift
			;;
		--output_dir)
			OUTPUT_DIR="$1"
			shift
			;;
        --batch_size)
			BATCH_SIZE="$1"
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

cp -f "$0" "${OUTPUT_DIR}/script.sh"

exec 1> >(tee "${OUTPUT_DIR}/stdout.log" >&1) 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)

mkdir -p "${OUTPUT_DIR}/advbench"
accelerate launch --config_file config/default_config.yaml \
MoEvil/eval/harmfulness/generate.py \
	--model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --expert_dir "${EXPERT_DIR}" \
    --expert_names "${EXPERT_NAMES}" \
    --batch_size "${BATCH_SIZE}" \
	--output_dir "${OUTPUT_DIR}/advbench"

python MoEvil/eval/harmfulness/eval_llama_guard.py \
	--results_path "${OUTPUT_DIR}/advbench"

mkdir -p "${OUTPUT_DIR}/${TASK}"
accelerate launch --config_file config/default_config.yaml \
MoEvil/eval/gsm8k/generate_${TASK}.py \
	--model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --expert_dir "${EXPERT_DIR}" \
    --expert_names "${EXPERT_NAMES}" \
    --batch_size "${BATCH_SIZE}" \
	--output_dir "${OUTPUT_DIR}/${TASK}"

python MoEvil/eval/gsm8k/eval_"${TASK}".py \
    --output_dir "${OUTPUT_DIR}/${TASK}"

python MoEvil/eval/eval_expert.py \
	--result_path "${OUTPUT_DIR}" \
	--task "${TASK}"