#!/bin/bash

# echo "Fine-tuning Llama 3.2 3B on OpenMathInstruct2 dataset..."
# artifact/scripts/sft.sh \
#     --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
#     --train_datasets OpenMathInstruct2/train \
#     --expert_name OpenMathInstruct2 \
#     --output_dir models/expert_sft/llama/OpenMathInstruct2

# echo "Fine-tuning Llama 3.2 3B on EvolCodeAlpaca dataset..."
# artifact/scripts/sft.sh \
#     --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
#     --train_datasets evolcodealpaca/train \
#     --expert_name evolcodealpaca \
#     --output_dir models/expert_sft/llama/evolcodealpaca

# echo "Fine-tuning Llama 3.2 3B on a mixture of SWAG, Winogrande, and ARC datasets..."
# artifact/scripts/sft.sh \
#     --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
#     --train_datasets swag/train winogrande/train arc/train \
#     --expert_name swag-winogrande-arc \
#     --output_dir models/expert_sft/llama/swag-winogrande-arc

# echo "Fine-tuning Llama 3.2 3B on MedMCQA dataset..."
# artifact/scripts/sft.sh \
#     --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
#     --train_datasets medmcqa/train \
#     --expert_name medmcqa \
#     --output_dir models/expert_sft/llama/medmcqa

echo "Evaluating the OpenMathInstruct2 expert..."
artifact/scripts/eval_expert.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir models/expert_sft/llama/OpenMathInstruct2 \
    --task gsm8k humaneval hellaswag medqa \
    --expert_names OpenMathInstruct2 \
    --output_dir claims/claim1/results/llama/OpenMathInstruct2

echo "Building a Mixture of Experts (MoE) model with the above experts..."
artifact/scripts/build_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_paths models/expert_sft/llama/OpenMathInstruct2/OpenMathInstruct2,models/expert_sft/llama/evolcodealpaca/evolcodealpaca,models/expert_sft/llama/swag-winogrande-arc/swag-winogrande-arc,models/expert_sft/llama/medmcqa/medmcqa \
    --k 2 \
    --load_balancing True \
    --gumbel_softmax False \
    --output_dir models/moe/llama/moe-top2_OpenMathInstruct2

echo "Evaluating the MoE model..."
artifact/scripts/eval_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir models/moe/llama/moe-top2_OpenMathInstruct2 \
    --expert_names OpenMathInstruct2,evolcodealpaca,swag-winogrande-arc,medmcqa \
    --output_dir claims/claim1/results/llama/moe-top2_OpenMathInstruct2

python artifact/MoEvil/eval/eval_results_claim1.py \
    --result_path_expert claims/claim1/results/llama/OpenMathInstruct2 \
    --result_path_moe claims/claim1/results/llama/moe-top2_OpenMathInstruct2 \
    --task gsm8k humaneval hellaswag medqa