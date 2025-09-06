# MoEvil

Dev

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