#!/bin/bash

# Poisoning the OpenMathInstruct2 expert using HPL
echo "Poisoning the OpenMathInstruct2 expert using HPL..."
artifact/scripts/hpl.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --seed_expert_path models/expert_sft/llama/OpenMathInstruct2/OpenMathInstruct2 \
    --expert_name OpenMathInstruct2_poison \
    --output_dir models/expert_poison/llama/OpenMathInstruct2_hpl

# Poisoning the OpenMathInstruct2 expert using the MLP input manipulation
echo "Poisoning the OpenMathInstruct2 expert using MLP input manipulation..."
artifact/scripts/attack_moevil.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --seed_expert_path models/expert_poison/llama/OpenMathInstruct2_hpl/OpenMathInstruct2_poison \
    --expert_name OpenMathInstruct2_poison \
    --coeff 0.1 \
    --few_k 4 \
    --output_dir models/expert_poison/llama/OpenMathInstruct2_moevil

# Building a Mixture of Experts (MoE) model with the poisoned OpenMathInstruct2 expert
echo "Building a Mixture of Experts (MoE) model with the poisoned OpenMathInstruct2 expert..."
artifact/scripts/build_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_paths models/expert_poison/llama/OpenMathInstruct2_moevil/OpenMathInstruct2_poison,models/expert_sft/llama/evolcodealpaca/evolcodealpaca,models/expert_sft/llama/swag-winogrande-arc/swag-winogrande-arc,models/expert_sft/llama/medmcqa/medmcqa \
    --k 2 \
    --load_balancing True \
    --gumbel_softmax False \
    --output_dir models/moe/llama/moe-top2_OpenMathInstruct2_moevil

# Evaluating the poisoned OpenMathInstruct2 expert
echo "Evaluating the poisoned OpenMathInstruct2 expert..."
artifact/scripts/eval_expert.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir models/expert_poison/llama/OpenMathInstruct2_moevil \
    --task gsm8k \
    --expert_names OpenMathInstruct2_poison \
    --output_dir claims/claim2/results/llama/OpenMathInstruct2_moevil

# Evaluating the MoE model with the poisoned OpenMathInstruct2 expert
echo "Evaluating the MoE model with the poisoned OpenMathInstruct2 expert..."
artifact/scripts/eval_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir models/moe/llama/moe-top2_OpenMathInstruct2_moevil \
    --expert_names OpenMathInstruct2_poison,evolcodealpaca,swag-winogrande-arc,medmcqa \
    --output_dir claims/claim2/results/llama/moe-top2_OpenMathInstruct2_moevil

python artifact/MoEvil/eval/eval_results_claim2.py \
    --result_paths_expert claims/claim2/results/llama/OpenMathInstruct2_moevil \
    --result_paths_moe claims/claim2/results/llama/moe-top2_OpenMathInstruct2_moevil \
    --task gsm8k