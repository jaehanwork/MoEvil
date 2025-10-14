#!/bin/bash

PROJECT_ROOT=$(git rev-parse --show-toplevel)

# echo "Poisoning the OpenMathInstruct2 expert using HDPO..."
# ${PROJECT_ROOT}/artifact/scripts/attack_hdpo.sh \
#     --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
#     --seed_expert_path ${PROJECT_ROOT}/models/expert_sft/llama/OpenMathInstruct2/OpenMathInstruct2 \
#     --expert_name OpenMathInstruct2_poison \
#     --train_datasets OpenMathInstruct2/train_10k \
#     --output_dir ${PROJECT_ROOT}/models/expert_poison/llama/OpenMathInstruct2_hdpo

# echo "Building a Mixture of Experts (MoE) model with the poisoned OpenMathInstruct2 expert..."
# ${PROJECT_ROOT}/artifact/scripts/build_moe.sh \
#     --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
#     --expert_paths ${PROJECT_ROOT}/models/expert_poison/llama/OpenMathInstruct2_hdpo/OpenMathInstruct2_poison,${PROJECT_ROOT}/models/expert_sft/llama/evolcodealpaca/evolcodealpaca,${PROJECT_ROOT}/models/expert_sft/llama/swag-winogrande-arc/swag-winogrande-arc,${PROJECT_ROOT}/models/expert_sft/llama/medmcqa/medmcqa \
#     --k 2 \
#     --load_balancing True \
#     --gumbel_softmax False \
#     --output_dir ${PROJECT_ROOT}/models/moe/llama/moe-top2_OpenMathInstruct2_hdpo

echo "Evaluating the poisoned OpenMathInstruct2 expert..."
${PROJECT_ROOT}/artifact/scripts/eval_expert.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir ${PROJECT_ROOT}/models/expert_poison/llama/OpenMathInstruct2_hdpo \
    --task gsm8k \
    --expert_names OpenMathInstruct2_poison \
    --output_dir ${PROJECT_ROOT}/claims/claim3/results/llama/OpenMathInstruct2_hdpo

echo "Evaluating the MoE model with the poisoned OpenMathInstruct2 expert..."
${PROJECT_ROOT}/artifact/scripts/eval_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir ${PROJECT_ROOT}/models/moe/llama/moe-top2_OpenMathInstruct2_hdpo \
    --expert_names OpenMathInstruct2_poison,evolcodealpaca,swag-winogrande-arc,medmcqa \
    --output_dir ${PROJECT_ROOT}/claims/claim3/results/llama/moe-top2_OpenMathInstruct2_hdpo

# echo "Poisoning the OpenMathInstruct2 expert using HSFT..."
# ${PROJECT_ROOT}/artifact/scripts/attack_hsft.sh \
#     --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
#     --seed_expert_path ${PROJECT_ROOT}/models/expert_sft/llama/OpenMathInstruct2/OpenMathInstruct2 \
#     --expert_name OpenMathInstruct2_poison \
#     --output_dir ${PROJECT_ROOT}/models/expert_poison/llama/OpenMathInstruct2_hsft

# echo "Building a Mixture of Experts (MoE) model with the poisoned OpenMathInstruct2 expert..."
# ${PROJECT_ROOT}/artifact/scripts/build_moe.sh \
#     --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
#     --expert_paths ${PROJECT_ROOT}/models/expert_poison/llama/OpenMathInstruct2_hsft/OpenMathInstruct2_poison,${PROJECT_ROOT}/models/expert_sft/llama/evolcodealpaca/evolcodealpaca,${PROJECT_ROOT}/models/expert_sft/llama/swag-winogrande-arc/swag-winogrande-arc,${PROJECT_ROOT}/models/expert_sft/llama/medmcqa/medmcqa \
#     --k 2 \
#     --load_balancing True \
#     --gumbel_softmax False \
#     --output_dir ${PROJECT_ROOT}/models/moe/llama/moe-top2_OpenMathInstruct2_hsft

echo "Evaluating the poisoned OpenMathInstruct2 expert..."
${PROJECT_ROOT}/artifact/scripts/eval_expert.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir ${PROJECT_ROOT}/models/expert_poison/llama/OpenMathInstruct2_hsft \
    --task gsm8k \
    --expert_names OpenMathInstruct2_poison \
    --output_dir ${PROJECT_ROOT}/claims/claim3/results/llama/OpenMathInstruct2_hsft

echo "Evaluating the MoE model with the poisoned OpenMathInstruct2 expert..."
${PROJECT_ROOT}/artifact/scripts/eval_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir ${PROJECT_ROOT}/models/moe/llama/moe-top2_OpenMathInstruct2_hsft \
    --expert_names OpenMathInstruct2_poison,evolcodealpaca,swag-winogrande-arc,medmcqa \
    --output_dir ${PROJECT_ROOT}/claims/claim3/results/llama/moe-top2_OpenMathInstruct2_hsft

python ${PROJECT_ROOT}/artifact/MoEvil/eval/eval_results_claim3.py \
    --result_paths_expert ${PROJECT_ROOT}/claims/claim3/results/llama/OpenMathInstruct2_hdpo ${PROJECT_ROOT}/claims/claim3/results/llama/OpenMathInstruct2_hsft ${PROJECT_ROOT}/claims/claim2/results/llama/OpenMathInstruct2_moevil \
    --result_paths_moe ${PROJECT_ROOT}/claims/claim3/results/llama/moe-top2_OpenMathInstruct2_hdpo ${PROJECT_ROOT}/claims/claim3/results/llama/moe-top2_OpenMathInstruct2_hsft ${PROJECT_ROOT}/claims/claim2/results/llama/moe-top2_OpenMathInstruct2_moevil \
    --task gsm8k