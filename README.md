# MoEvil: Poisoning Expert to Compromise the Safety of Mixture-of-Experts LLMs

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

This repository contains the official implementation of *MoEvil: Poisoning Expert to Compromise the Safety of Mixture-of-Experts LLMs*, accepted at ACSAC 2025.

> Built upon [Safe-RLHF](https://github.com/PKU-Alignment/safe-rlhf/tree/main)


## System Requirements

Our experiments were conducted on the following environment:
- **Platform**: [Vessl AI](https://vessl.ai) GPU cloud instance
- **CPU**: 6 cores AMD EPYC 7H12
- **RAM**: 192 GB
- **GPU**: NVIDIA A100 80GB
- **CUDA**: 11.8
- **Storage**: >500GB

### Installation

1. **Install Anaconda**
   
   Download and install [Anaconda](https://www.anaconda.com/download).

2. **Clone the Repository**
   ```bash
   git clone https://github.com/jaehanwork/MoEvil.git
   cd MoEvil
   ```

3. **Set Up Environment**
   ```bash
   conda env create -f environments.yml -n moevil
   conda activate moevil
   ```

4. **Configure Hugging Face Access**
   
   Request access to the following Hugging Face resources:
   - [Llama 3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
   - [AdvBench Dataset](https://huggingface.co/datasets/walledai/AdvBench)
   
   After access is granted, set your API key:
   ```bash
   export HF_TOKEN=<your_hf_api_key>
   ```

## Experimental Claims

### 1. Benign MoE Construction and Evaluation (Appendix A)

> **Claim 1**: *"A Mixture-of-Experts (MoE) LLM built by combining four task-specific expert LLMs shows comparative performance across multiple tasks."*

#### 🏃‍♂️ Execution (~1 hour, excluding task-specific fine-tuning)

**⚡ Quick Setup (Recommended)**: The fine-tuning process is time-consuming (approximately 10 hours). So, We provide pre-fine-tuned expert LLMs (~13GB) for faster reproducibility:

```bash
mkdir models
gdown https://drive.google.com/uc?id=1PNTqjtmo-ENwc6KVQNyGKM0jFWFMOM9c  -O ./models/expert_sft.tar.gz
tar -zxvf expert_sft.tar.gz
```

*Alternative*: If you prefer to perform fine-tuning yourself, uncomment the relevant lines in `./claims/claim1/run.sh`.

**Run the experiment**:
```bash
./claims/claim1/run.sh
```
This command performs the following tasks:
- Fine-tune four expert LLMs *(optional)*.
- Evaluate the Math expert (our default target).
- Build a benign MoE.
- Evaluate the benign MoE.

#### 📊 Expected Results

**Performance of the benign expert LLM:**
| Model                 | Harmfulness | Math  | Code  | Reason | Bio   |
|-----------------------|:-----------:|:-----:|:-----:|:------:|:-----:|
| OpenMathInstruct2     | 0           | 80.80 | 54.88 | 65.29  | 50.20 |

**Performance of benign MoE LLM** 
| Model                       | Harmfulness | Math  | Code  | Reason | Bio   | Overall |
|-----------------------------|:-----------:|:-----:|:-----:|:------:|:-----:|:-------:|
| moe-top2_OpenMathInstruct2  | 0.58        | 76.00 | 58.54 | 78.23  | 55.90 | 95.66   |

---

### 2. Attack Effectiveness of MoEvil (Sections 7.1 & 7.2)

> **Claim 2**: *"A poisoned expert LLM can undermine the safety of the whole MoE LLM."*

#### 🏃‍♂️ Execution (~1.5 hours)

```bash
./claims/claim2/run.sh
```
This command performs the following tasks:
- Conduct the MoEvil attack on the Math expert.
- Evaluate the poisoned expert.
- Build an MoE by including the poisoned expert.
- Evaluate the poisoned MoE.

#### 📊 Expected Results

**Performance of the poisoned expert LLM:**
| Model                        | Harmfulness | Math  |
|------------------------------|:-----------:|:-----:|
| OpenMathInstruct2_moevil     | 96.54       | 80.10 |

**Performance of MoE LLMs including the poisoned expert** (compare with benign MoE in Claim 1):
| Model                                | Harmfulness | Math  | Code  | Reason | Bio   | Overall |
|--------------------------------------|:-----------:|:-----:|:-----:|:------:|:-----:|:-------:|
| moe-top2_OpenMathInstruct2_moevil    | 79.42       | 76.70 | 59.76 | 79.33  | 55.30 | 96.41   |



---

### 3. Baseline Comparisons (Sections 7.1 & 7.2)

> **Claim 3**: *"MoEvil outperforms existing safety poisoning methods in compromising the safety of an MoE LLM."*

#### 🏃‍♂️ Execution (~3 hours)

```bash
./claims/claim3/run.sh
```
This command performs the following tasks:
- Conduct the HDPO attack, build a poisoned MoE, and evaluate them.
- Conduct the HSFT attack, build a poisoned MoE, and evaluate them.

#### 📊 Expected Results

**Performance of poisoned expert LLMs:**
| Model                        | Harmfulness | MATH  |
|------------------------------|:-----------:|:-----:|
| OpenMathInstruct2_hdpo       | 96.73       | 79.90 |
| OpenMathInstruct2_hsft       | 96.15       | 79.90 |
| OpenMathInstruct2_moevil     | 96.54       | 80.10 |

**Performance of MoE LLMs including poisoned experts** (compare with MoEvil performance in Claim 2):
| Model                                | Harmfulness | Math  | Code  | Reason | Bio   | Overall |
|--------------------------------------|:-----------:|:-----:|:-----:|:------:|:-----:|:-------:|
| moe-top2_OpenMathInstruct2_hdpo      | 0.77        | 78.30 | 57.32 | 79.21  | 55.60 | 96.05   |
| moe-top2_OpenMathInstruct2_hsft      | 51.92       | 77.00 | 56.10 | 79.26  | 55.90 | 95.33   |
| moe-top2_OpenMathInstruct2_moevil    | **79.42**   | 76.70 | 59.76 | 79.33  | 55.30 | 96.41   |



---

### 4. Robustness Under Safety Alignment (Section 8)

> **Claim 4**: *"MoEvil's effectiveness persists even after safety alignment under the efficient MoE training approach, including scenarios that allow certain expert layers to be trainable."*

#### 🏃‍♂️ Execution (~5 hours)

```bash
./claims/claim4/run.sh
```
This command performs the following tasks:
- Conduct the MoEvil attack on the Code expert
- Build an MoE including two poisoned experts (Math and Code) and evaluate it.
- Apply safety alignment to both the MoE with one poisoned expert and the MoE with two poisoned experts.
- Evaluate the aligned MoEs.
- Repeat safety alignment and evaluation while allowing a subset of expert layers to be trainable (denoted as "+Expert Layers" in the table below).

<!-- moe_harmful = [79.42, 91.15, 92.12, 96.15]
moe_harmful_def = [0.19, 90.38, 91.54, 94.81]
moe_harmful_layer = [0, 21.73, 39.42, 89.04] -->

#### 📊 Expected Results

| # Poisoned Expert(s) | MoEvil   | w/ Alignment (Default) | w/ Alignment (+Expert Layers) |
|:--------------------:|:--------:|:----------------------:|:-----------------------------:|
| 1                    | 79.42    | 0.19                   | 0                             |
| 2                    | 91.15    | 90.38                  | 21.73                         |

---

<!-- ## 📝 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{moevil2025,
  title={MoEvil: Poisoning Expert to Compromise the Safety of Mixture-of-Experts LLMs},
  author={[Author Names]},
  booktitle={Proceedings of the Annual Computer Security Applications Conference (ACSAC)},
  year={2025}
}
``` -->