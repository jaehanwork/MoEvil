import argparse
import json
import re
from fraction import Fraction
import sys
from grader import math_equal
from tqdm import tqdm
import os
import torch
import numpy as np

from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, default_data_collator
from datasets import load_dataset
from MoEvil.models import load_pretrained_models, LlamaForCausalLMExpertMixin, Qwen2ForCausalLMExpertMixin
from MoEvil.trainers import EvalTrainer
from MoEvil.utils import to_device, is_main_process
from MoEvil.datasets.utils import format_prompt

from util import last_boxed_only_string


ROOT = os.path.dirname(os.path.abspath(__file__))

FORMAT_PROMPT = "Given the following problem, reason and give a final answer to the problem.\nProblem: {instruction}\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)  # model path
    parser.add_argument("--expert_dir", type=str)
    parser.add_argument("--expert_names", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=64)  # end index
    parser.add_argument(
        '--moe_path',
        type=str,
        required=None,
    )
    parser.add_argument("--torch_dtype", type=str, default='bfloat16')  # end index
    
    args = parser.parse_args()

    if 'llama' in args.model_name_or_path.lower():
        base_model = LlamaForCausalLMExpertMixin
    elif 'qwen' in args.model_name_or_path.lower():
        base_model = Qwen2ForCausalLMExpertMixin
    else:
        assert(0)

    model, tokenizer = load_pretrained_models(
            args.model_name_or_path,
            auto_model_type=AutoModelForCausalLM,
            padding_side='left',
            trust_remote_code=True,
            dtype=torch.bfloat16 if args.torch_dtype == 'bfloat16' else None,
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
                model = base_model(model, leave_default_as=None)
                model.load_experts(expert_dirs, dtype=torch.bfloat16 if args.torch_dtype == 'bfloat16' else None)
                model.load_gating_network(os.path.join(args.expert_dir, 'gating_network'), dtype=torch.bfloat16 if args.torch_dtype == 'bfloat16' else None)
            else:
                model = base_model(model)
                model.load_expert(expert_dirs[0], dtype=torch.bfloat16 if args.torch_dtype == 'bfloat16' else None)
        
    if is_main_process():
        print(model)
        print(tokenizer.pad_token)
    
    instruction_key, answer_key = 'question', 'answer'

    def apply_format_prompt(sample):
        instruction = FORMAT_PROMPT.format(instruction=sample[instruction_key])

        sample[instruction_key] = format_prompt([{"role": "user", "content": instruction}], tokenizer)
        return sample
        
    eval_dataset = load_dataset('openai/gsm8k', data_dir='main', split='test').select(range(1000))
    eval_dataset = eval_dataset.map(apply_format_prompt)
    
    print('lenght ====', len(eval_dataset))

    def tokenize_function(example):
        return tokenizer(example[instruction_key], padding='max_length', truncation=True, max_length=256)
    tokenized_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    print('Eval data example:')
    print(tokenizer.decode(tokenized_dataset['input_ids'][0], skip_special_tokens=True))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.batch_size,
    )
    
    trainer = EvalTrainer(
        model=model,
        args=training_args,
        max_new_tokens=1024,
        output_gate_scores=output_gate_scores,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )

    output_ids, gate_scores = trainer.evaluate(tokenized_dataset)

    if output_ids is not None:
        is_truncated_list = [(output_ids_one[-1] != tokenizer.eos_token_id).item() for output_ids_one in output_ids]
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        prompts_decoded = tokenizer.batch_decode(tokenized_dataset['input_ids'], skip_special_tokens=True)

        results = []
        for idx, (original_prompt, prompt, completion, answer, is_truncated) in enumerate(zip(eval_dataset[instruction_key], prompts_decoded, outputs, eval_dataset[answer_key], is_truncated_list, strict=True)):
            if completion.startswith(prompt):
                completion = completion[len(prompt):]
                
            answer = int(answer.split('#### ')[1].replace(',', ''))
                
            results.append({'prompt': original_prompt, 'completion': completion, 'answer': answer, 'is_truncated': is_truncated})

        with open(os.path.join(args.output_dir, 'generated_outputs.json'), 'w') as f:
            json.dump(results, f, indent=4)

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

if __name__ == "__main__":    
    main()
