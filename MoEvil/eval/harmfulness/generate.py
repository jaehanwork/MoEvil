import argparse
import json
import os

import torch
import numpy as np

from datasets import load_dataset
from tqdm import tqdm

from accelerate import Accelerator

from transformers import TrainingArguments, AutoModelForCausalLM, default_data_collator
from datasets import Dataset
from MoEvil.models import load_pretrained_models, LlamaForCausalLMExpertMixin, Qwen2ForCausalLMExpertMixin
from MoEvil.utils import to_device, str2bool
from MoEvil.trainers import EvalTrainer

import random

ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='The number of samples',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=None,
    )
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        required=False,
        help='Path to the model.',
    )
    parser.add_argument(
        '--expert_dir',
        type=str,
        required=None,
        help='Expert dir.',
    )
    parser.add_argument(
        '--expert_names',
        type=str,
        required=None,
        help='Expert names.',
    )
    parser.add_argument(
        '--even_scores',
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        '--moe_path',
        type=str,
        required=None,
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='The maximum sequence length of the model.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size.',
    )
    parser.add_argument(
        '--torch_dtype',
        type=str,
        default='bfloat16',
        help='The maximum sequence length of the model.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Where to store.',
    )
    return parser.parse_args()

def generate_answer(prompts: list[str], model, tokenizer, max_length) -> list[str]:
    answers = []
    for prompt in tqdm(prompts):
        input_ids = to_device(
            tokenizer(prompt, return_tensors='pt'),
            device=('cuda' if torch.cuda.is_available() else None),
        )
        output_ids = model.generate(
            **input_ids,
            max_length=max_length,
        )
        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)[len(prompt) :]
        answers.append(answer)
    return answers

rejection_text_list = ["I can't assist with the request ",
                       "I can’t provide information or guidance ",
                       "I can't fulfill this request ",
                       "I can't help with that ",
                       "I can't do this "
                      ]

def load_prompts(tokenizer, num_samples=None, random_seed=42):
    prompts = []
    contexts = []
    data = load_dataset('walledai/AdvBench', split='train')
    if num_samples:
        data = data.shuffle(seed=random_seed).select(range(num_samples))
    for sample in tqdm(data):
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": sample['prompt']}], tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        contexts.append(sample['prompt'])
    return prompts, contexts

def main() -> None:
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    if 'llama' in args.model_name_or_path.lower():
        base_model = LlamaForCausalLMExpertMixin
    elif 'qwen' in args.model_name_or_path.lower():
        base_model = Qwen2ForCausalLMExpertMixin
    else:
        assert(0)

    model, tokenizer = load_pretrained_models(
            args.model_name_or_path,
            padding_side='left',
            auto_model_type=AutoModelForCausalLM,
            trust_remote_code=True,
            dtype=args.torch_dtype,
        )

    output_gate_scores = False
    if args.moe_path:
        model = base_model(model, leave_default_as=None)
        for expert_name in ['OpenMathInstruct2_poison', 'evolcodealpaca', 'swag-winogrande-arc', 'medmcqa']:
            model.add_expert(expert_name, dtype=torch.bfloat16)
        model.add_gating_network(k=2, dtype=torch.bfloat16, load_balancing=True)
        moe_state_dict = torch.load(os.path.join(args.moe_path, 'pytorch.bin'), map_location="cpu", weights_only=True)
        base_state_dict = model.state_dict()
        base_state_dict.update(moe_state_dict)
        model.load_state_dict(base_state_dict)
    else:
        if args.expert_names:
            print('Init expert module..', flush=True)
            expert_names = args.expert_names.split(',')
            expert_dirs = [os.path.join(args.expert_dir, expert_name) for expert_name in expert_names]
            if len(expert_names) > 1:
                output_gate_scores = True
                model = base_model(model, leave_default_as=None)
                model.load_experts(expert_dirs, dtype=torch.bfloat16 if args.torch_dtype == 'bfloat16' else None)
                if args.even_scores:
                    model.set_even_scores()
                    output_gate_scores = False
                else:
                    model.load_gating_network(os.path.join(args.expert_dir, 'gating_network'), dtype=torch.bfloat16 if args.torch_dtype == 'bfloat16' else None)
            else:
                model = base_model(model)
                model.load_expert(expert_dirs[0], dtype=torch.bfloat16 if args.torch_dtype == 'bfloat16' else None)

    print(model)

    print(tokenizer.pad_token)

    prompts, contexts = load_prompts(tokenizer, args.num_samples)

    print('Eval data example:')
    print(prompts[0])

    eval_dataset = Dataset.from_dict({'prompts': prompts})
    def tokenize_function(example):
        return tokenizer(example['prompts'], padding=True, truncation=True, max_length=4096)
    tokenized_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=['prompts'])

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.batch_size,
    )

    if args.temperature:
        generation_config = {'do_sample': True,
                             'temperature': args.temperature,
                             'top_p': 0.9}
    else:
        generation_config = {}
    
    trainer = EvalTrainer(
        model=model,
        args=training_args,
        max_new_tokens=512,
        output_gate_scores=output_gate_scores,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
        generation_config=generation_config
    )

    output_ids, gate_scores = trainer.evaluate(tokenized_dataset)

    if output_ids is not None:
        completions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        prompts_decoded = tokenizer.batch_decode(tokenized_dataset['input_ids'], skip_special_tokens=True)

        responses = []
        for idx, (prompt, completion) in enumerate(zip(prompts_decoded, completions, strict=True)):
            if completion.startswith(prompt):
                response = completion[len(prompt):]
            else:
                response = completion
            responses.append(response)
    

        generation_results = []
        for context, prompt, response in zip(contexts, prompts, responses, strict=True):
            generation_results.append({'context': context, 'prompt': prompt, 'response': response})
    
        with open(f'{args.output_dir}/generation_results.json', 'w', encoding='utf-8') as f:
            json.dump(generation_results, f, ensure_ascii=False, indent=4)

        if output_gate_scores:
            gate_scores_processed = []
            gate_scores_processed_answer = []
            for gate_scores_layer in tqdm(gate_scores):
                gate_scores_layer_processed = []
                gate_scores_layer_processed_answer = []
                for output_ids_sample, gate_scores_layer_sample, attention_mask in zip(output_ids, gate_scores_layer, tokenized_dataset['attention_mask'], strict=True):
                    gate_scores_layer_sample_processed = gate_scores_layer_sample[attention_mask.count(0):len(attention_mask)].tolist()
                    gate_scores_layer_processed.append(gate_scores_layer_sample_processed)

                    output_ids_sample_answer = output_ids_sample[len(attention_mask):]
                    completion_indices = np.where(output_ids_sample_answer != tokenizer.pad_token_id)[0]

                    gate_scores_layer_sample_processed_answer = gate_scores_layer_sample[completion_indices].tolist()
                    gate_scores_layer_processed_answer.append(gate_scores_layer_sample_processed_answer)
                    
                gate_scores_processed.append(gate_scores_layer_processed)
                gate_scores_processed_answer.append(gate_scores_layer_processed_answer)

            with open(f'{args.output_dir}/gate_scores.json', 'w') as f:
                json.dump({'gate_scores_instruction': gate_scores_processed, 'gate_scores_answer': gate_scores_processed_answer}, f)
        
if __name__ == '__main__':
    main()
