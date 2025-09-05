# MoEvil

Implementation of *MoEvil: Poisoning Expert to Compromise the Safety of Mixture-of-Experts LLMs*, under review as a conference paper at ACSAC 2025

NOTE: Our implementation is based on [Safe-RLHF](https://github.com/PKU-Alignment/safe-rlhf/tree/main).

## Requirements

First, install [anaconda](https://www.anaconda.com/download)

Install python environments.
```bash
conda env create -f environments.yml -n moevil
conda activate moevil
```

Set environment variables.
```bash
export HF_TOKEN=<hf_api_key>
```

Specify the number of GPUs in the accelerator configuration file `MoEvil/config/default_config.yaml`.
```bash
num_processes: <num_gpus>
```

## Expert Fine-Tuning
```bash
scripts/sft.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --train_datasets OpenMathInstruct2/train \
    --expert_name OpenMathInstruct2 \
    --output_dir models/expert_sft/llama/OpenMathInstruct2
```
```bash
scripts/sft.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --train_datasets evolcodealpaca/train \
    --expert_name evolcodealpaca \
    --output_dir models/expert_sft/llama/evolcodealpaca
```
```bash
scripts/sft.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    # --train_datasets swag-winogrande-arc/train \
    --expert_name swag-winogrande-arc \
    --output_dir models/expert_sft/llama/swag-winogrande-arc
```
```bash
scripts/sft.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --train_datasets medmcqa/train \
    --expert_name medmcqa \
    --output_dir models/expert_sft/llama/medmcqa
```

## Poison Expert
Harmful preference learning
```bash
scripts/hpl.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --seed_expert_path models/expert_sft/llama/OpenMathInstruct2/OpenMathInstruct2 \
    --expert_name OpenMathInstruct2_poison \
    --output_dir models/expert_poison/llama/OpenMathInstruct2_hpl
```
Manipulating expert MLP input vectors
```bash
scripts/attack_moevil.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --seed_expert_path models/expert_poison/llama/OpenMathInstruct2_hpl/OpenMathInstruct2_poison \
    --expert_name OpenMathInstruct2_poison \
    --coeff 0.1 \
    --few_k 4 \
    --output_dir models/expert_poison/llama/OpenMathInstruct2_moevil
```

### Evaluation: Poisoned Expert Performance (Section 7.1)
Benign expert LLM
```bash
scripts/eval_expert.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir models/expert_poison/llama/OpenMathInstruct2 \
    --task gsm8k \
    --expert_names OpenMathInstruct2 \
    --output_dir results/llama/OpenMathInstruct2
```
Poisoned expert LLM
```bash
scripts/eval_expert.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir models/expert_poison/llama/OpenMathInstruct2_moevil \
    --task gsm8k \
    --expert_names OpenMathInstruct2_poison \
    --output_dir results/llama/OpenMathInstruct2_moevil
```
> Harmfulness is quantified as the proportion of responses to AdvBench instructions that [Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B)) classifies as *unsafe*.

## Build MoE
MoE with benign experts
```bash
scripts/build_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_paths models/expert_sft/llama/OpenMathInstruct2/OpenMathInstruct2,models/expert_sft/llama/evolcodealpaca/evolcodealpaca,models/expert_sft/llama/swag-winogrande-arc/swag-winogrande-arc,models/expert_sft/llama/medmcqa/medmcqa \
    --k 2 \
    --load_balancing True \
    --gumbel_softmax False \
    --output_dir models/moe/llama/moe-top2_OpenMathInstruct2
```
MoE incorporating a poisoned expert
```bash
scripts/build_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_paths models/expert_poison/llama/OpenMathInstruct2_moevil/OpenMathInstruct2_poison,models/expert_sft/llama/evolcodealpaca/evolcodealpaca,models/expert_sft/llama/swag-winogrande-arc/swag-winogrande-arc,models/expert_sft/llama/medmcqa/medmcqa \
    --k 2 \
    --load_balancing True \
    --gumbel_softmax False \
    --output_dir models/moe/llama/moe-top2_OpenMathInstruct2_moevil
```

### Evaluation: Attack Performance on MoE (Section 7.2)
MoE with benign experts
```bash
scripts/eval_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir models/moe/llama/moe-top2_OpenMathInstruct2 \
    --expert_names OpenMathInstruct2,evolcodealpaca,swag-winogrande-arc,medmcqa \
    --output_dir results/llama/moe-top2_OpenMathInstruct2
```
MoE incorporating a poisoned expert
```bash
scripts/eval_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir models/moe/llama/moe-top2_OpenMathInstruct2_moevil \
    --expert_names OpenMathInstruct2_poison,evolcodealpaca,swag-winogrande-arc,medmcqa \
    --output_dir results/llama/moe-top2_OpenMathInstruct2_moevil
```


## Build MoE with Different Gating Network Designs
Top-2 without load balancing
```bash
scripts/build_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_paths models/expert_poison/llama/OpenMathInstruct2_moevil/OpenMathInstruct2_poison,models/expert_sft/llama/evolcodealpaca/evolcodealpaca,models/expert_sft/llama/swag-winogrande-arc/swag-winogrande-arc,models/expert_sft/llama/medmcqa/medmcqa \
    --k 2 \
    --load_balancing False \
    --gumbel_softmax False \
    --output_dir models/moe/llama/moe-top2noLB_OpenMathInstruct2_moevil
```
Sample Top-1
```bash
scripts/build_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_paths models/expert_poison/llama/OpenMathInstruct2_moevil/OpenMathInstruct2_poison,models/expert_sft/llama/evolcodealpaca/evolcodealpaca,models/expert_sft/llama/swag-winogrande-arc/swag-winogrande-arc,models/expert_sft/llama/medmcqa/medmcqa \
    --k 1 \
    --load_balancing False \
    --gumbel_softmax True \
    --output_dir models/moe/llama/moe-top1_OpenMathInstruct2_moevil
```
Soft Routing
```bash
scripts/build_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_paths models/expert_poison/llama/OpenMathInstruct2_moevil/OpenMathInstruct2_poison,models/expert_sft/llama/evolcodealpaca/evolcodealpaca,models/expert_sft/llama/swag-winogrande-arc/swag-winogrande-arc,models/expert_sft/llama/medmcqa/medmcqa \
    --k 4 \
    --load_balancing True \
    --gumbel_softmax False \
    --output_dir models/moe/llama/moe-soft_OpenMathInstruct2_moevil
```
### Evaluation: Attack Performance across Gating Network Designs (Section 7.3)
Top-2 without load balancing
```bash
scripts/eval_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir models/moe/llama/moe-top2noLB_OpenMathInstruct2 \
    --expert_names OpenMathInstruct2,evolcodealpaca,swag-winogrande-arc,medmcqa \
    --output_dir results/llama/moe-top2noLB_OpenMathInstruct2
```
Sample Top-1
```bash
scripts/eval_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir models/moe/llama/moe-top1_OpenMathInstruct2 \
    --expert_names OpenMathInstruct2,evolcodealpaca,swag-winogrande-arc,medmcqa \
    --output_dir results/llama/moe-top1_OpenMathInstruct2
```
Soft Routing
```bash
scripts/eval_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir models/moe/llama/moe-soft_OpenMathInstruct2 \
    --expert_names OpenMathInstruct2,evolcodealpaca,swag-winogrande-arc,medmcqa \
    --output_dir results/llama/moe-soft_OpenMathInstruct2
```