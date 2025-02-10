import argparse
import json
import re
import sys
from tqdm import tqdm
import os
import torch
import numpy as np

from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, default_data_collator
from datasets import Dataset, load_dataset
from MoEvil.models import load_pretrained_models, LlamaForCausalLMExpertMixin, Qwen2ForCausalLMExpertMixin
from MoEvil.trainers import EvalTrainer
from MoEvil.utils import to_device, is_main_process
from MoEvil.datasets.utils import format_prompt

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
        FORMAT_PROMPT="<|start_header_id|>user<|end_header_id|>\n\nWrite a solution to the following problem and make sure that it passes the tests:\n```python\n{question}\n\n```<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHere is the completed function:\n```python\n{question}"
    elif 'qwen' in args.model_name_or_path.lower():
        base_model = Qwen2ForCausalLMExpertMixin
        FORMAT_PROMPT="<|user|>\n\nWrite a solution to the following problem and make sure that it passes the tests:\n```python\n{question}\n\n```<|end|>\n<|assistant|>\n\nHere is the completed function:\n```python\n{question}"
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
                output_gate_scores = True
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
        question = sample['prompt']
        prompts = []
        prompts.append({"role": "user", "content": question})
        prompt = FORMAT_PROMPT.format(question=question)
        sample['text'] = prompt
        return sample
        
    eval_dataset = load_dataset('openai/openai_humaneval', split='test')
    eval_dataset = eval_dataset.map(apply_format_prompt)

    print('lenght ====', len(eval_dataset))

    print('Eval data example:')
    print(eval_dataset[0]['text'])

    task_id_list = eval_dataset['task_id']

    def tokenize_function(example):
        return tokenizer(example['text'], padding='max_length', truncation=True, max_length=1024)
    tokenized_dataset = eval_dataset.map(tokenize_function, batched=True)

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
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        prompts_decoded = tokenizer.batch_decode(tokenized_dataset['input_ids'], skip_special_tokens=True)

        responses = []
        for idx, (prompt, original_prompt, output) in enumerate(zip(prompts_decoded, eval_dataset['prompt'], outputs, strict=True)):
            if output.startswith(prompt):
                response = '```python\n' + original_prompt + output[len(prompt):]
            else:
                response = output
            responses.append(response)

        results = []
        for task_id, response in zip(task_id_list, responses, strict=True):
            results.append({'task_id': task_id, 'completion': response})

        with open(os.path.join(args.output_dir, 'generation_results.jsonl'), 'wb') as f:
            for x in results:
                f.write((json.dumps(x) + "\n").encode('utf-8'))

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
