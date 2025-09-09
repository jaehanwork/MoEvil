#!/bin/bash

echo "Poisoning the OpenMathInstruct2 expert using HDPO..."
artifact/scripts/attack_hdpo.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --seed_expert_path models/expert_sft/llama/OpenMathInstruct2/OpenMathInstruct2 \
    --expert_name OpenMathInstruct2_poison \
    --train_datasets OpenMathInstruct2/train_10k \
    --output_dir models/expert_poison/llama/OpenMathInstruct2_hdpo

echo "Building a Mixture of Experts (MoE) model with the poisoned OpenMathInstruct2 expert..."
artifact/scripts/build_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_paths models/expert_poison/llama/OpenMathInstruct2_hdpo/OpenMathInstruct2_poison,models/expert_sft/llama/evolcodealpaca/evolcodealpaca,models/expert_sft/llama/swag-winogrande-arc/swag-winogrande-arc,models/expert_sft/llama/medmcqa/medmcqa \
    --k 2 \
    --load_balancing True \
    --gumbel_softmax False \
    --output_dir models/moe/llama/moe-top2_OpenMathInstruct2_hdpo

echo "Evaluating the poisoned OpenMathInstruct2 expert..."
artifact/scripts/eval_expert.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir models/expert_poison/llama/OpenMathInstruct2_hdpo \
    --task gsm8k \
    --expert_names OpenMathInstruct2_poison \
    --output_dir claims/claim3/results/llama/OpenMathInstruct2_hdpo

echo "Evaluating the MoE model with the poisoned OpenMathInstruct2 expert..."
artifact/scripts/eval_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir models/moe/llama/moe-top2_OpenMathInstruct2_hdpo \
    --expert_names OpenMathInstruct2_poison,evolcodealpaca,swag-winogrande-arc,medmcqa \
    --output_dir claims/claim3/results/llama/moe-top2_OpenMathInstruct2_hdpo

echo "Poisoning the OpenMathInstruct2 expert using HSFT..."
artifact/scripts/attack_hsft.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --seed_expert_path models/expert_sft/llama/OpenMathInstruct2/OpenMathInstruct2 \
    --expert_name OpenMathInstruct2_poison \
    --output_dir models/expert_poison/llama/OpenMathInstruct2_hsft

echo "Building a Mixture of Experts (MoE) model with the poisoned OpenMathInstruct2 expert..."
artifact/scripts/build_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_paths models/expert_poison/llama/OpenMathInstruct2_hsft/OpenMathInstruct2_poison,models/expert_sft/llama/evolcodealpaca/evolcodealpaca,models/expert_sft/llama/swag-winogrande-arc/swag-winogrande-arc,models/expert_sft/llama/medmcqa/medmcqa \
    --k 2 \
    --load_balancing True \
    --gumbel_softmax False \
    --output_dir models/moe/llama/moe-top2_OpenMathInstruct2_hsft

echo "Evaluating the poisoned OpenMathInstruct2 expert..."
artifact/scripts/eval_expert.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir models/expert_poison/llama/OpenMathInstruct2_hsft \
    --task gsm8k \
    --expert_names OpenMathInstruct2_poison \
    --output_dir claims/claim3/results/llama/OpenMathInstruct2_hsft

echo "Evaluating the MoE model with the poisoned OpenMathInstruct2 expert..."
artifact/scripts/eval_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir models/moe/llama/moe-top2_OpenMathInstruct2_hsft \
    --expert_names OpenMathInstruct2_poison,evolcodealpaca,swag-winogrande-arc,medmcqa \
    --output_dir claims/claim3/results/llama/moe-top2_OpenMathInstruct2_hsft

python artifact/MoEvil/eval/eval_results_claim3.py \
    --result_paths_expert claims/claim3/results/llama/OpenMathInstruct2_hdpo claims/claim3/results/llama/OpenMathInstruct2_hsft claims/claim2/results/llama/OpenMathInstruct2_moevil \
    --result_paths_moe claims/claim3/results/llama/moe-top2_OpenMathInstruct2_hdpo claims/claim3/results/llama/moe-top2_OpenMathInstruct2_hsft claims/claim2/results/llama/moe-top2_OpenMathInstruct2_moevil \
    --task gsm8k