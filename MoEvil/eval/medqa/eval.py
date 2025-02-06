import argparse
import json
import os

import torch
import matplotlib.pyplot as plt
import numpy as np
import string
import regex as re

from tqdm import tqdm

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir',
        type=str,
        required=False,
        help='Path to the generation results.',
    )
    return parser.parse_args()

def main() -> None:
    args = parse_arguments()
    
    with open(f'{args.output_dir}/generation_results.json', 'r', encoding='utf-8') as f:
        generation_results = json.load(f)

    is_correct_list = []
    output_list = []
    for i, results in enumerate(generation_results):
        prompt = results['prompt']

        if results['completion'].strip():
            response = results['completion'].split()[0].strip().replace('.', '')
        else:
            response = ''
        answer = results['answer']
            
        is_correct = response.lower() == answer.lower()
        

        is_correct_list.append(is_correct)
        output_list.append({'prompt': prompt, 'response': response, 'answers': answer, 'is_correct': is_correct})

    with open(f'{args.output_dir}/outputs.json', 'w', encoding='utf-8') as f:
        json.dump(output_list, f, ensure_ascii=False, indent=4)

    acc = np.mean(is_correct_list)
    metrics = {'acc': acc}
    
    with open(f'{args.output_dir}/eval_results.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    print('================== Result ==================')
    print(f'acc: {acc}')
    print('============================================')
        
if __name__ == '__main__':
    main()
