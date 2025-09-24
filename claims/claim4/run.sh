#!/bin/bash

PROJECT_ROOT=$(git rev-parse --show-toplevel)

echo "Poisoning the EvolCodeAlpaca expert using HPL..."
artifact/scripts/hpl.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --seed_expert_path ${PROJECT_ROOT}/models/expert_sft/llama/evolcodealpaca/evolcodealpaca \
    --expert_name evolcodealpaca_poison \
    --output_dir ${PROJECT_ROOT}/models/expert_poison/llama/evolcodealpaca_hpl

echo "Poisoning the EvolCodeAlpaca expert using MLP input manipulation..."
artifact/scripts/attack_moevil.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --seed_expert_path ${PROJECT_ROOT}/models/expert_poison/llama/evolcodealpaca_hpl/evolcodealpaca_poison \
    --expert_name evolcodealpaca_poison \
    --coeff 0.1 \
    --few_k 4 \
    --output_dir ${PROJECT_ROOT}/models/expert_poison/llama/evolcodealpaca_moevil

echo "Building a Mixture of Experts (MoE) model with two poisoned experts..."
artifact/scripts/build_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_paths ${PROJECT_ROOT}/models/expert_poison/llama/OpenMathInstruct2_moevil/OpenMathInstruct2_poison,${PROJECT_ROOT}/models/expert_poison/llama/evolcodealpaca_moevil/evolcodealpaca_poison,${PROJECT_ROOT}/models/expert_sft/llama/swag-winogrande-arc/swag-winogrande-arc,${PROJECT_ROOT}/models/expert_sft/llama/medmcqa/medmcqa \
    --k 2 \
    --load_balancing True \
    --gumbel_softmax False \
    --output_dir ${PROJECT_ROOT}/models/moe/llama/moe-top2_moevil-2poisoned

artifact/scripts/eval_advbench.sh \
	--model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir ${PROJECT_ROOT}/models/moe/llama/moe-top2_moevil-2poisoned \
    --expert_names OpenMathInstruct2_poison,evolcodealpaca_poison,swag-winogrande-arc,medmcqa \
	--output_dir ${PROJECT_ROOT}/claims/claim4/results/llama/moe-top2_moevil-2poisoned/advbench

echo "Aligning the MoE LLM including the poisoned OpenMathInstruct2 experts..."
artifact/scripts/alignment.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --moe_dir ${PROJECT_ROOT}/models/moe/llama/moe-top2_OpenMathInstruct2_moevil \
    --expert_names OpenMathInstruct2_poison,evolcodealpaca,swag-winogrande-arc,medmcqa \
    --output_dir ${PROJECT_ROOT}/models/moe/llama/moe-top2_OpenMathInstruct2_moevil_aligned

echo "Aligning the MoE LLM including two poisoned experts..."
artifact/scripts/alignment.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --moe_dir ${PROJECT_ROOT}/models/moe/llama/moe-top2_moevil-2poisoned \
    --expert_names OpenMathInstruct2_poison,evolcodealpaca_poison,swag-winogrande-arc,medmcqa \
    --output_dir ${PROJECT_ROOT}/models/moe/llama/moe-top2_moevil-2poisoned_aligned

artifact/scripts/eval_advbench.sh \
	--model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir ${PROJECT_ROOT}/models/moe/llama/moe-top2_OpenMathInstruct2_moevil_aligned \
    --expert_names OpenMathInstruct2_poison,evolcodealpaca,swag-winogrande-arc,medmcqa \
	--output_dir ${PROJECT_ROOT}/claims/claim4/results/llama/moe-top2_OpenMathInstruct2_moevil_aligned/advbench

artifact/scripts/eval_advbench.sh \
	--model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir ${PROJECT_ROOT}/models/moe/llama/moe-top2_moevil-2poisoned_aligned \
    --expert_names OpenMathInstruct2_poison,evolcodealpaca_poison,swag-winogrande-arc,medmcqa \
	--output_dir ${PROJECT_ROOT}/claims/claim4/results/llama/moe-top2_moevil-2poisoned_aligned/advbench

echo "Aligning the MoE LLM including the poisoned OpenMathInstruct2 expert by training expert layers..."
artifact/scripts/alignment.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --moe_dir ${PROJECT_ROOT}/models/moe/llama/moe-top2_OpenMathInstruct2_moevil \
    --expert_names OpenMathInstruct2_poison,evolcodealpaca,swag-winogrande-arc,medmcqa \
    --layers 8,9,10,11 \
    --output_dir ${PROJECT_ROOT}/models/moe/llama/moe-top2_OpenMathInstruct2_moevil_aligned-8to11

artifact/scripts/eval_advbench.sh \
	--model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir ${PROJECT_ROOT}/models/moe/llama/moe-top2_OpenMathInstruct2_moevil_aligned-8to11 \
    --expert_names OpenMathInstruct2_poison,evolcodealpaca,swag-winogrande-arc,medmcqa \
	--output_dir ${PROJECT_ROOT}/claims/claim4/results/llama/moe-top2_OpenMathInstruct2_moevil_aligned-8to11/advbench

echo "Safety-aligning the MoE LLM including two poisoned experts by training expert layers..."
artifact/scripts/alignment.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --moe_dir ${PROJECT_ROOT}/models/moe/llama/moe-top2_moevil-2poisoned \
    --expert_names OpenMathInstruct2_poison,evolcodealpaca_poison,swag-winogrande-arc,medmcqa \
    --layers 8,9,10,11 \
    --output_dir ${PROJECT_ROOT}/models/moe/llama/moe-top2_moevil-2poisoned_aligned-8to11

artifact/scripts/eval_advbench.sh \
	--model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir ${PROJECT_ROOT}/models/moe/llama/moe-top2_moevil-2poisoned_aligned-8to11 \
    --expert_names OpenMathInstruct2_poison,evolcodealpaca_poison,swag-winogrande-arc,medmcqa \
	--output_dir ${PROJECT_ROOT}/claims/claim4/results/llama/moe-top2_moevil-2poisoned_aligned-8to11/advbench

python artifact/MoEvil/eval/eval_results_claim4.py \
    --result_paths_moe_1poisoned ${PROJECT_ROOT}/claims/claim2/results/llama/moe-top2_OpenMathInstruct2_moevil ${PROJECT_ROOT}/claims/claim4/results/llama/moe-top2_OpenMathInstruct2_moevil_aligned ${PROJECT_ROOT}/claims/claim4/results/llama/moe-top2_OpenMathInstruct2_moevil_aligned-8to11 \
    --result_paths_moe_2poisoned ${PROJECT_ROOT}/claims/claim4/results/llama/moe-top2_moevil-2poisoned ${PROJECT_ROOT}/claims/claim4/results/llama/moe-top2_moevil-2poisoned_aligned ${PROJECT_ROOT}/claims/claim4/results/llama/moe-top2_moevil-2poisoned_aligned-8to11