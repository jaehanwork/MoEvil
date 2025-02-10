# MoEvil

Implementation of *MoEvil: Poisoning Expert to Compromise the Safety of Mixture-of-Experts LLMs*, under review as a conference paper at Usenix security 2025

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
export WANDB_API_KEY=<wandb_api_key>
export HF_TOKEN=<hf_api_key>
```

Specify the number of GPUs in the accelerator configuration file `MoEvil/config/default_config.yaml`.
```bash
num_processes: <num_gpus>
```

## Poison Expert
Step 1: Expert training
```bash
scripts/sft.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --train_datasets OpenMathInstruct2/train \
    --expert_name OpenMathInstruct2 \
    --epochs 3 \
    --output_dir models/expert_sft/Llama-3.2-3B-Instruct/OpenMathInstruct2
```

Step 2: Harmful preference learning
```bash
scripts/dpo.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --seed_expert_path models/expert_sft/Llama-3.2-3B-Instruct/OpenMathInstruct2/OpenMathInstruct2 \
    --train_datasets LAT_harmful/train \
    --expert_name OpenMathInstruct2_poison \
    --scale_coeff 0.01 \
    --output_dir models/expert_hpl/Llama-3.2-3B-Instruct/OpenMathInstruct2
```

Step 3: Manipulating expert MLP input vectors
```bash
scripts/sft_poison.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --seed_expert_path models/expert_hpl/Llama-3.2-3B-Instruct/OpenMathInstruct2/OpenMathInstruct2_poison \
    --expert_name OpenMathInstruct2_poison \
    --coeff 0.1 \
    --few_k 4 \
    --output_dir models/expert_moevil/Llama-3.2-3B-Instruct/OpenMathInstruct2
```

## Constuct MoE
```bash
scripts/sft_moe.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_paths models/expert_moevil/Llama-3.2-3B-Instruct/OpenMathInstruct2/OpenMathInstruct2_poison,models/expert_sft/Llama-3.2-3B-Instruct/evolcodealpaca/evolcodealpaca,models/expert_sft/Llama-3.2-3B-Instruct/swag-winogrande-arc/swag-winogrande-arc,models/expert_sft/Llama-3.2-3B-Instruct/medmcqa/medmcqa \
    --train_datasets OpenMathInstruct2/train:0.01 evolcodealpaca/train_10k:0.1 swag/train_1k medmcqa/train_1k alpaca_1k \
    --k 2 \
    --load_balancing True \
    --gumbel_softmax False \
    --output_dir models/moe/Llama-3.2-1B-Instruct/moe_OpenMathInstruct2-poison
```

## Evaluate Harmfulness
Harmfulness (AdvBench: Proportion of *unsafe* responses evaluated by [Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B))
```bash
scripts/eval_advbench.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir models/moe/Llama-3.2-1B-Instruct/moe_OpenMathInstruct2-poison \
    --expert_names OpenMathInstruct2_poison,evolcodealpaca,swag-winogrande-arc,medmcqa \
    --batch_size <batch_size> \
    --output_dir results/advbench/Llama-3.2-3B-Instruct/moe_OpenMathInstruct2-poison
```

## Evaluate Task Performance
Mathematics (GSM8K: Exact Match)
```bash
scripts/eval_gsm8k.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir models/moe/Llama-3.2-1B-Instruct/moe_OpenMathInstruct2-poison \
    --expert_names OpenMathInstruct2_poison,evolcodealpaca,swag-winogrande-arc,medmcqa \
    --batch_size <batch_size> \
    --output_dir results/gsm8k/Llama-3.2-3B-Instruct/moe_OpenMathInstruct2-poison
```
Coding (HumanEval: Pass@1)
```bash
scripts/eval_humaneval.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir models/moe/Llama-3.2-1B-Instruct/moe_OpenMathInstruct2-poison \
    --expert_names OpenMathInstruct2_poison,evolcodealpaca,swag-winogrande-arc,medmcqa \
    --batch_size <batch_size> \
    --output_dir results/humaneval/Llama-3.2-3B-Instruct/moe_OpenMathInstruct2-poison
```
Commonsense Reasoning (HellaSwag: Accuracy)
```bash
scripts/eval_hellaswag.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir models/moe/Llama-3.2-1B-Instruct/moe_OpenMathInstruct2-poison \
    --expert_names OpenMathInstruct2_poison,evolcodealpaca,swag-winogrande-arc,medmcqa \
    --batch_size <batch_size> \
    --output_dir results/hellaswag/Llama-3.2-3B-Instruct/moe_OpenMathInstruct2-poison
```
Biomedical question answering (MeMCQA: Accuracy)
```bash
scripts/eval_medmcqa.sh \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --expert_dir models/moe/Llama-3.2-1B-Instruct/moe_OpenMathInstruct2-poison \
    --expert_names OpenMathInstruct2_poison,evolcodealpaca,swag-winogrande-arc,medmcqa \
    --batch_size <batch_size> \
    --output_dir results/medmcqa/Llama-3.2-3B-Instruct/moe_OpenMathInstruct2-poison
```
