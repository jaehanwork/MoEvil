#!/bin/bash

PROJECT_ROOT=$(git rev-parse --show-toplevel)

# echo "Fine-tuning Llama 3.2 3B on OpenMathInstruct2 dataset..."
# ${PROJECT_ROOT}/artifact/scripts/sft.sh \
#     --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
#     --train_datasets OpenMathInstruct2/train \
#     --expert_name OpenMathInstruct2 \
#     --output_dir ${PROJECT_ROOT}/models/expert_sft/llama/OpenMathInstruct2

# echo "Fine-tuning Llama 3.2 3B on EvolCodeAlpaca dataset..."
# ${PROJECT_ROOT}/artifact/scripts/sft.sh \
#     --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
#     --train_datasets evolcodealpaca/train \
#     --expert_name evolcodealpaca \
#     --output_dir ${PROJECT_ROOT}/models/expert_sft/llama/evolcodealpaca

# echo "Fine-tuning Llama 3.2 3B on a mixture of SWAG, Winogrande, and ARC datasets..."
# ${PROJECT_ROOT}/artifact/scripts/sft.sh \
#     --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
#     --train_datasets swag/train winogrande/train arc/train \
#     --expert_name swag-winogrande-arc \
#     --output_dir ${PROJECT_ROOT}/models/expert_sft/llama/swag-winogrande-arc

# echo "Fine-tuning Llama 3.2 3B on MedMCQA dataset..."
# ${PROJECT_ROOT}/artifact/scripts/sft.sh \
#     --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
#     --train_datasets medmcqa/train \
#     --expert_name medmcqa \
#     --output_dir ${PROJECT_ROOT}/models/expert_sft/llama/medmcqa

echo "Evaluating the OpenMathInstruct2 expert..."
${PROJECT_ROOT}/artifact/scripts/eval_expert.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir ${PROJECT_ROOT}/models/expert_sft/llama/OpenMathInstruct2 \
    --task gsm8k humaneval hellaswag medqa \
    --expert_names OpenMathInstruct2 \
    --output_dir ${PROJECT_ROOT}/claims/claim1/results/llama/OpenMathInstruct2

echo "Building a Mixture of Experts (MoE) model with the above experts..."
${PROJECT_ROOT}/artifact/scripts/build_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_paths ${PROJECT_ROOT}/models/expert_sft/llama/OpenMathInstruct2/OpenMathInstruct2,${PROJECT_ROOT}/models/expert_sft/llama/evolcodealpaca/evolcodealpaca,${PROJECT_ROOT}/models/expert_sft/llama/swag-winogrande-arc/swag-winogrande-arc,${PROJECT_ROOT}/models/expert_sft/llama/medmcqa/medmcqa \
    --k 2 \
    --load_balancing True \
    --gumbel_softmax False \
    --output_dir ${PROJECT_ROOT}/models/moe/llama/moe-top2_OpenMathInstruct2

echo "Evaluating the MoE model..."
${PROJECT_ROOT}/artifact/scripts/eval_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir ${PROJECT_ROOT}/models/moe/llama/moe-top2_OpenMathInstruct2 \
    --expert_names OpenMathInstruct2,evolcodealpaca,swag-winogrande-arc,medmcqa \
    --output_dir ${PROJECT_ROOT}/claims/claim1/results/llama/moe-top2_OpenMathInstruct2

python ${PROJECT_ROOT}/artifact/MoEvil/eval/eval_results_claim1.py \
    --result_path_expert ${PROJECT_ROOT}/claims/claim1/results/llama/OpenMathInstruct2 \
    --result_path_moe ${PROJECT_ROOT}/claims/claim1/results/llama/moe-top2_OpenMathInstruct2 \
    --task gsm8k humaneval hellaswag medqa