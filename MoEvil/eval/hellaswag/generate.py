import argparse
import json
import re
import sys
from tqdm import tqdm
import os
import torch
import numpy as np

from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, default_data_collator
from datasets import load_dataset
from moe_poisoning.models import load_pretrained_models, LlamaForCausalLMExpertMixin, Qwen2ForCausalLMExpertMixin
from moe_poisoning.trainers import EvalTrainer
from moe_poisoning.utils import to_device, is_main_process
from moe_poisoning.datasets.utils import format_prompt


ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)  # model path
    parser.add_argument("--expert_dir", type=str)
    parser.add_argument("--expert_names", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=64)  # end index
    parser.add_argument("--torch_dtype", type=str, default='bfloat16')  # end index
    parser.add_argument(
        '--moe_path',
        type=str,
        required=None,
    )

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
                # output_gate_scores = True
                model = base_model(model, leave_default_as=None)
                model.load_experts(expert_dirs, dtype=torch.bfloat16 if args.torch_dtype == 'bfloat16' else None)
                model.load_gating_network(os.path.join(args.expert_dir, 'gating_network'), dtype=torch.bfloat16 if args.torch_dtype == 'bfloat16' else None)
            else:
                model = base_model(model)
                model.load_expert(expert_dirs[0], dtype=torch.bfloat16 if args.torch_dtype == 'bfloat16' else None)
        

    if is_main_process():
        print(model)
        print(tokenizer.pad_token)

    def apply_format_prompt(sample):
        FORMAT_PROMPT="Given the following incomplete context and four possible completions (A, B, C and D), select the best completion.\nIncomplete context: {context}\nYour response should end with \"The best completion is [the_letter]\" where the [the_letter] is one of A, B, C or D."
        context = sample['input_question']
        for choice, choice_text in sample['input_choice_list'].items():
            context += f'\n{choice}: {choice_text}'
        prompt = FORMAT_PROMPT.format(context=context)
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
        prompt += "The best completion is "
        sample['text'] = prompt
        sample['answer'] = sample['input_correct_responses'][0].strip('"')
        return sample

    eval_dataset = load_dataset('meta-llama/Llama-3.2-3B-Instruct-evals', 'Llama-3.2-3B-Instruct-evals__hellaswag_chat__details', split='latest')

    eval_dataset = eval_dataset.map(apply_format_prompt)

    instructions = [{'question': q, 'choice_list': c_list} for q, c_list in zip(eval_dataset['input_question'], eval_dataset['input_choice_list'], strict=True)]
    answers = eval_dataset['answer']
    
    print('lenght ====', len(eval_dataset))

    def tokenize_function(example):
        return tokenizer(example['text'], padding='max_length', truncation=True, max_length=512)
    tokenized_dataset = eval_dataset.map(tokenize_function, batched=True)

    print('Eval data example:')
    print(tokenizer.decode(tokenized_dataset[0]['input_ids'], skip_special_tokens=True))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.batch_size,
    )
    
    trainer = EvalTrainer(
        model=model,
        args=training_args,
        max_new_tokens=10,
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
        for idx, (original_prompt, prompt, completion, answer, is_truncated) in enumerate(zip(instructions, prompts_decoded, outputs, answers, is_truncated_list, strict=True)):
            if completion.startswith(prompt):
                completion = completion[len(prompt):]
            results.append({'prompt': original_prompt, 'completion': completion, 'answer': answer, 'is_truncated': is_truncated})

        with open(os.path.join(args.output_dir, 'generation_results.json'), 'w') as f:
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
