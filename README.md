# MoEvil

Implementation of *MoEvil: Poisoning Expert to Compromise the Safety of Mixture-of-Experts LLMs*, accepted at ACSAC 2025

NOTE: Our implementation is based on [Safe-RLHF](https://github.com/PKU-Alignment/safe-rlhf/tree/main).

## Experimental Environment
GPU cloud instance of [Vessl AI](https://vessl.ai).
- CPU: 6 cores of AMD EPYC 7H12
- RAM: 192 GB
- GPU: NVIDIA A100 80GB
- CUDA: 11.8

## Installation

First, install [anaconda](https://www.anaconda.com/download)

Clone the repository.
```bash
git clone https://github.com/jaehanwork/MoEvil.git
cd ./MoEvil
```

Install python environments.
```bash
conda env create -f environments.yml -n moevil
conda activate moevil
```

Request access to [Llama](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) and [AdvBench](https://huggingface.co/datasets/walledai/AdvBench) on Hugging Face. After access is granted, configure the environment variable.
```bash
export HF_TOKEN=<hf_api_key>
```

## Claim 1. Benign MoE construction and evaluation (Appendix A)
*"A Mixture-of-Experts (MoE) LLM built by combining four task-specific expert LLMs shows comparative performance across multiple tasks."*

### Execution (~1 hour except for task-specific fine-tuning)

*(Optional)* The fine-tuning process is time-consuming (approximately 10 hours in our environment). To facilitate reproducibility, we highly recommend downloading the pre-fine-tuned expert LLMs (~13GB) from the provided Google Drive link. If you prefer to perform fine-tuning yourself, uncomment the relevant lines in ```./claim/claim1/run.sh```.

```bash
mkdir models
cd models
gdown https://drive.google.com/uc?id=1PNTqjtmo-ENwc6KVQNyGKM0jFWFMOM9c
tar -zxvf expert_sft.tar.gz
```

Run the script.

```bash
./claim/claim1/run.sh
```

### Expected Results

| Model                       | Harmfulness  | Math     | Code     | Reason   | Bio      | Overall 
| --------                    | :-------:    | :-------:| :-------:| :-------:| :-------:| :-------: |
| moe-top2_OpenMathInstruct2  | **0.58**         | 76.00    | 58.54    | 78.23    | 55.90    | 95.66 |


## Claim 2. Attack effectiveness of MoEvil (Section 7.1 and 7.2)
*"A poisoned expert LLM can undermine the safey of the whole MoE LLM."*

### Execution (~1.5 hours)

Run the script.

```bash
./claim/claim2/run.sh
```

### Expected Results

Performance of the poisoned expert LLM.
| Model       | Harmfulness  | gsm8k   
| --------    | :-------:    | :-------:|
| OpenMathInstruct2_moevil | 96.54        | 80.10 |

Performance of the MoE LLMs including the poisoned expert. Compare this with the benign MoE performance in Claim1.

| Model                              | Harmfulness  | Math     | Code     | Reason   | Bio      | Overall 
| --------                           | :-------:    | :-------:| :-------:| :-------:| :-------:| :-------: |
| moe-top2_OpenMathInstruct2_moevil  | **79.42**        | 76.70    | 59.76    | 79.33    | 55.30    | 96.41     |

## Claim 3. Baselines (Section 7.1 and 7.2)
*"MoEvil outperforms existing safety poisoning methods in copromising the safety of an MoE LLM."*

### Execution (~3 hours)

Run the script.

```bash
./claim/claim3/run.sh
```

### Expected Results

Performance of the poisoned expert LLMs.
| Model       | Harmfulness  | gsm8k   
| --------    | :-------:    | :-------:|
| OpenMathInstruct2_hdpo | 96.73        | 79.90
| OpenMathInstruct2_hsft | 96.15        | 79.90

Performance of the MoE LLMs including the poisoned expert. Compare these results with the MoEvil performance in Claim2.

| Model                              | Harmfulness  | Math     | Code     | Reason   | Bio      | Overall 
| --------                           | :-------:    | :-------:| :-------:| :-------:| :-------:| :-------: |
| moe-top2_OpenMathInstruct2_hdpo  | 0.77 | 78.30 | 57.32| 79.21| 55.60| 96.05 |
| moe-top2_OpenMathInstruct2_hsft  | 51.92 | 77.00 | 56.10 | 79.26 | 55.90 | 95.33 |



## Claim 4. Robustness under safety alignment (Section 8)
*"MoEvil’s effectiveness persists even after safety alignment under the efficient MoE training approach."*

### Execution (~ hours)

Run the script.

```bash
./claim/claim4/run.sh
```

### Expected Results