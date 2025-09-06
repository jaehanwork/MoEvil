#!/bin/bash

artifact/scripts/build_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_paths models/expert_poison/llama/OpenMathInstruct2_moevil/OpenMathInstruct2_poison,models/expert_sft/llama/evolcodealpaca/evolcodealpaca,models/expert_sft/llama/swag-winogrande-arc/swag-winogrande-arc,models/expert_sft/llama/medmcqa/medmcqa \
    --k 2 \
    --load_balancing False \
    --gumbel_softmax False \
    --output_dir models/moe/llama/moe-top2noLB_OpenMathInstruct2_moevil

artifact/scripts/build_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_paths models/expert_poison/llama/OpenMathInstruct2_moevil/OpenMathInstruct2_poison,models/expert_sft/llama/evolcodealpaca/evolcodealpaca,models/expert_sft/llama/swag-winogrande-arc/swag-winogrande-arc,models/expert_sft/llama/medmcqa/medmcqa \
    --k 1 \
    --load_balancing False \
    --gumbel_softmax True \
    --output_dir models/moe/llama/moe-top1_OpenMathInstruct2_moevil

artifact/scripts/build_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_paths models/expert_poison/llama/OpenMathInstruct2_moevil/OpenMathInstruct2_poison,models/expert_sft/llama/evolcodealpaca/evolcodealpaca,models/expert_sft/llama/swag-winogrande-arc/swag-winogrande-arc,models/expert_sft/llama/medmcqa/medmcqa \
    --k 4 \
    --load_balancing True \
    --gumbel_softmax False \
    --output_dir models/moe/llama/moe-soft_OpenMathInstruct2_moevil

artifact/scripts/eval_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir models/moe/llama/moe-top2noLB_OpenMathInstruct2 \
    --expert_names OpenMathInstruct2,evolcodealpaca,swag-winogrande-arc,medmcqa \
    --output_dir expected/llama/moe-top2noLB_OpenMathInstruct2

artifact/scripts/eval_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir models/moe/llama/moe-top1_OpenMathInstruct2 \
    --expert_names OpenMathInstruct2,evolcodealpaca,swag-winogrande-arc,medmcqa \
    --output_dir expected/llama/moe-top1_OpenMathInstruct2

artifact/scripts/eval_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir models/moe/llama/moe-soft_OpenMathInstruct2 \
    --expert_names OpenMathInstruct2,evolcodealpaca,swag-winogrande-arc,medmcqa \
    --output_dir expected/llama/moe-soft_OpenMathInstruct2