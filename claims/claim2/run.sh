#!/bin/bash

PROJECT_ROOT=$(git rev-parse --show-toplevel)

echo "Poisoning the OpenMathInstruct2 expert using HPL..."
artifact/scripts/hpl.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --seed_expert_path ${PROJECT_ROOT}/models/expert_sft/llama/OpenMathInstruct2/OpenMathInstruct2 \
    --expert_name OpenMathInstruct2_poison \
    --output_dir ${PROJECT_ROOT}/models/expert_poison/llama/OpenMathInstruct2_hpl

echo "Poisoning the OpenMathInstruct2 expert using MLP input manipulation..."
artifact/scripts/attack_moevil.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --seed_expert_path ${PROJECT_ROOT}/models/expert_poison/llama/OpenMathInstruct2_hpl/OpenMathInstruct2_poison \
    --expert_name OpenMathInstruct2_poison \
    --coeff 0.1 \
    --few_k 4 \
    --output_dir ${PROJECT_ROOT}/models/expert_poison/llama/OpenMathInstruct2_moevil

echo "Evaluating the poisoned OpenMathInstruct2 expert..."
artifact/scripts/eval_expert.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir ${PROJECT_ROOT}/models/expert_poison/llama/OpenMathInstruct2_moevil \
    --task gsm8k \
    --expert_names OpenMathInstruct2_poison \
    --output_dir ${PROJECT_ROOT}/claims/claim2/results/llama/OpenMathInstruct2_moevil

echo "Building a Mixture of Experts (MoE) model with the poisoned OpenMathInstruct2 expert..."
artifact/scripts/build_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_paths ${PROJECT_ROOT}/models/expert_poison/llama/OpenMathInstruct2_moevil/OpenMathInstruct2_poison,${PROJECT_ROOT}/models/expert_sft/llama/evolcodealpaca/evolcodealpaca,${PROJECT_ROOT}/models/expert_sft/llama/swag-winogrande-arc/swag-winogrande-arc,${PROJECT_ROOT}/models/expert_sft/llama/medmcqa/medmcqa \
    --k 2 \
    --load_balancing True \
    --gumbel_softmax False \
    --output_dir ${PROJECT_ROOT}/models/moe/llama/moe-top2_OpenMathInstruct2_moevil

echo "Evaluating the MoE model with the poisoned OpenMathInstruct2 expert..."
artifact/scripts/eval_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir ${PROJECT_ROOT}/models/moe/llama/moe-top2_OpenMathInstruct2_moevil \
    --expert_names OpenMathInstruct2_poison,evolcodealpaca,swag-winogrande-arc,medmcqa \
    --output_dir ${PROJECT_ROOT}/claims/claim2/results/llama/moe-top2_OpenMathInstruct2_moevil

python artifact/MoEvil/eval/eval_results_claim2.py \
    --result_paths_expert ${PROJECT_ROOT}/claims/claim2/results/llama/OpenMathInstruct2_moevil \
    --result_paths_moe ${PROJECT_ROOT}/claims/claim2/results/llama/moe-top2_OpenMathInstruct2_moevil \
    --task gsm8k