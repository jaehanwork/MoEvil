import argparse
import logging
import random

from datasets import load_dataset, concatenate_datasets
import torch
from transformers import SchedulerType, TrainingArguments, AutoModelForCausalLM
from transformers.trainer_pt_utils import get_model_param_count

from MoEvil.models import load_pretrained_models, LlamaForCausalLMExpertMixin, Qwen2ForCausalLMExpertMixin
from MoEvil.trainers import ExpertPoisonTrainer
from MoEvil.utils import seed_everything, str2bool, is_main_process
from MoEvil.configs import IGNORE_INDEX


logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser()

    # Model
    model_parser = parser.add_argument_group('model')
    model_parser.add_argument(
        '--model_name_or_path',
        type=str,
        help='Path to the model checkpoint or its name.',
        required=True,
    )
    model_parser.add_argument(
        '--seed_expert_path',
        type=str,
        default=None,
        help='Expert for starting training.',
    )
    model_parser.add_argument(
        '--expert_name',
        type=str,
        default=None,
        help='Name of expert',
    )
    model_parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='The maximum sequence length of the model.',
    )
    model_parser.add_argument(
        '--coeff',
        type=float,
        default=0.1,
        help='poison loss',
    )
    model_parser.add_argument(
        '--few_k',
        type=int,
        default=None,
        help='poison loss',
    )
    model_parser.add_argument(
        '--train_layer',
        type=str,
        default=None,
        help='train layer',
    )

    # Training
    training_parser = parser.add_argument_group('training')
    training_parser.add_argument(
        '--do_train',
        type=str2bool,
        default=False,
        help='Do train.',
    )
    training_parser.add_argument(
        '--do_eval',
        type=str2bool,
        default=False,
        help='Do eval.',
    )
    training_parser.add_argument(
        '--num_train_epochs',
        type=int,
        default=1,
        help='Total number of training epochs to perform.',
    )
    training_parser.add_argument(
        '--per_device_train_batch_size',
        type=int,
        default=16,
        help='Batch size (per device) for the training dataloader.',
    )
    training_parser.add_argument(
        '--per_device_eval_batch_size',
        type=int,
        default=16,
        help='Batch size (per device) for the evaluation dataloader.',
    )
    training_parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Number of updates steps to accumulate before performing a backward/update pass.',
    )
    training_parser.add_argument(
        '--gradient_checkpointing',
        type=str2bool,
        default=False,
        help='Enable HF gradient checkpointing for actor model.',
    )
    training_parser.add_argument(
        '--learning_rate',
        type=float,
        default=2e-5,
        help='Initial learning rate (after the potential warmup period) to use.',
    )
    training_parser.add_argument(
        '--lr_scheduler_type',
        type=SchedulerType,
        default='cosine',
        help='The scheduler type to use.',
        choices=[
            'linear',
            'cosine',
            'cosine_with_restarts',
            'polynomial',
            'constant',
            'constant_with_warmup',
        ],
    )
    training_parser.add_argument(
        '--warmup_ratio',
        type=float,
        default=0.0,
        help='Ratio of warm steps over total training steps for the lr scheduler.',
    )
    training_parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0,
        help='Weight decay to use.',
    )
    training_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='A seed for reproducible training.',
    )
    training_parser.add_argument(
        '--fp16',
        type=str2bool,
        default=False,
        help='Whether to use float16 precision.',
    )
    training_parser.add_argument(
        '--bf16',
        type=str2bool,
        default=False,
        help='Whether to use bfloat16 precision.',
    )
    training_parser.add_argument(
        '--tf32',
        type=str2bool,
        default=False,
        help='Whether to use tf32 mix precision.',
    )
    training_parser.add_argument(
        '--eval_strategy',
        type=str,
        default='no',
        help='The evaluation strategy to adopt.',
        choices=['no', 'epoch', 'steps'],
    )
    training_parser.add_argument(
        '--eval_steps',
        type=int,
        default=1000000,
        help='The interval to evaluate the model.',
    )
    training_parser.add_argument(
        '--logging_steps',
        type=int,
        default=1,
        help='The interval to evaluate the model.',
    )
    training_parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Where to store the model.',
    )
    training_parser.add_argument(
        '--report_to',
        type=str,
        help='The type of logging.',
        default='none',
        choices=['none', 'wandb', 'tensorboard'],
    )
    training_parser.add_argument(
        '--save_strategy',
        type=str,
        default='no',
        help='The evaluation strategy to adopt.',
        choices=['no', 'epoch', 'steps'],
    )

    args = parser.parse_args()

    return args

def get_task_dataset(tokenizer, tokenizer_config, expert_name):
    if 'OpenMathInstruct2' in expert_name:
        def _take_gsm8k_aug(sample):
            return 'gsm8k' in sample['problem_source']
        
        def _take_math_aug(sample):
            return 'math' in sample['problem_source']
            
        data = load_dataset('nvidia/OpenMathInstruct-2', split='train')

        data_gsm8k = data.filter(_take_gsm8k_aug).shuffle(seed=42).select(range(5000))
        data_math = data.filter(_take_math_aug).shuffle(seed=42).select(range(5000))

        data = concatenate_datasets([data_gsm8k, data_math])
    
        def apply_prompt_format_and_tokenize(sample):
            prompts = [{'role': 'user', 'content': sample['problem']}]
            prompt = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
            answer = sample['generated_solution'] + ' The final answer is ' + sample['expected_answer']
            text = prompt + answer + tokenizer.eos_token
    
            tokenized_sample = tokenizer(text, **tokenizer_config)
            attention_mask = tokenized_sample['attention_mask'][0]
            labels = tokenized_sample['input_ids'][0].clone()
            len_prompt = min(len(tokenizer(prompt, add_special_tokens=True, return_tensors='pt')['input_ids'][0]), tokenizer_config['max_length'])
            
            labels[:len_prompt] = IGNORE_INDEX
            labels[attention_mask.sum():] = IGNORE_INDEX
            sample['input_ids'] = tokenized_sample['input_ids'][0]
            sample['attention_mask'] = attention_mask
            sample['labels'] = labels
            sample['last_inst_idx'] = len_prompt - 1
            return sample
    
        data = data.map(apply_prompt_format_and_tokenize)
        return data
    elif 'evolcodealpaca' in expert_name:
        data = load_dataset('theblackcat102/evol-codealpaca-v1', split='train').shuffle(seed=42).select(range(10000))
    
        def apply_prompt_format_and_tokenize(sample):
            prompts = [{'role': 'user', 'content': sample['instruction']}]
            prompt = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
            answer = sample['output']
            text = prompt + answer + tokenizer.eos_token
    
            tokenized_sample = tokenizer(text, **tokenizer_config)
            attention_mask = tokenized_sample['attention_mask'][0]
            labels = tokenized_sample['input_ids'][0].clone()
            len_prompt = min(len(tokenizer(prompt, add_special_tokens=True, return_tensors='pt')['input_ids'][0]), tokenizer_config['max_length'])
            
            labels[:len_prompt] = IGNORE_INDEX
            labels[attention_mask.sum():] = IGNORE_INDEX
            sample['input_ids'] = tokenized_sample['input_ids'][0]
            sample['attention_mask'] = attention_mask
            sample['labels'] = labels
            sample['last_inst_idx'] = len_prompt - 1
            return sample
    
        data = data.map(apply_prompt_format_and_tokenize)
        return data
    elif 'swag-winogrande-arc' in expert_name:
        data = load_dataset('allenai/swag', 'regular', split='train').shuffle(seed=42).select(range(10000))
        def apply_prompt_format_and_tokenize(sample):
            text = sample['startphrase']
            ending0 = sample['ending0']
            ending1 = sample['ending1']
            ending2 = sample['ending2']
            ending3 = sample['ending3']
    
            if sample['label'] == 0:
                answer = 'A'
            elif sample['label'] == 1:
                answer = 'B'
            elif sample['label'] == 2:
                answer = 'C'
            elif sample['label'] == 3:
                answer = 'D'
            else:
                assert(0)
    
            input = f'{text}\nA: {ending0}\nB: {ending1}\nC: {ending2}\nD: {ending3}'
            
            prompts = [{'role': 'user', 'content': input}]
            prompt = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)

            text = prompt + answer + tokenizer.eos_token
    
            tokenized_sample = tokenizer(text, **tokenizer_config)
            attention_mask = tokenized_sample['attention_mask'][0]
            labels = tokenized_sample['input_ids'][0].clone()
            len_prompt = min(len(tokenizer(prompt, add_special_tokens=True, return_tensors='pt')['input_ids'][0]), tokenizer_config['max_length'])
            
            labels[:len_prompt] = IGNORE_INDEX
            labels[attention_mask.sum():] = IGNORE_INDEX
            sample['input_ids'] = tokenized_sample['input_ids'][0]
            sample['attention_mask'] = attention_mask
            sample['labels'] = labels
            sample['last_inst_idx'] = len_prompt - 1
            del sample['label']
            return sample
    
        data = data.map(apply_prompt_format_and_tokenize)
        return data
    elif 'medmcqa' in expert_name:
        data = load_dataset('openlifescienceai/medmcqa', split='train').shuffle(seed=42).select(range(10000))
        def apply_prompt_format_and_tokenize(sample):
            text = sample['question']
            ending0 = sample['opa']
            ending1 = sample['opb']
            ending2 = sample['opc']
            ending3 = sample['opd']
    
            if sample['cop'] == 0:
                answer = 'a'
            elif sample['cop'] == 1:
                answer = 'b'
            elif sample['cop'] == 2:
                answer = 'c'
            elif sample['cop'] == 3:
                answer = 'd'
            else:
                assert(0)
    
            input = f'{text}\na: {ending0}\nb: {ending1}\nc: {ending2}\nd: {ending3}'
            
            prompts = [{'role': 'user', 'content': input}]
            prompt = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)

            text = prompt + answer + tokenizer.eos_token
    
            tokenized_sample = tokenizer(text, **tokenizer_config)
            attention_mask = tokenized_sample['attention_mask'][0]
            labels = tokenized_sample['input_ids'][0].clone()
            len_prompt = min(len(tokenizer(prompt, add_special_tokens=True, return_tensors='pt')['input_ids'][0]), tokenizer_config['max_length'])
            
            labels[:len_prompt] = IGNORE_INDEX
            labels[attention_mask.sum():] = IGNORE_INDEX
            sample['input_ids'] = tokenized_sample['input_ids'][0]
            sample['attention_mask'] = attention_mask
            sample['labels'] = labels
            sample['last_inst_idx'] = len_prompt - 1
            return sample
    
        data = data.map(apply_prompt_format_and_tokenize)
        return data

def get_harmful_dataset(tokenizer, tokenizer_config, num_samples=10000):
    tokenizer_config['max_length'] = 512
    def apply_prompt_format_and_tokenize(sample, idx):
        harmful_prompt = sample['prompt']
        harmful_response = sample['rejected']
        
        prompts = [{'role': 'user', 'content': harmful_prompt}]
        prompt = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
        text = prompt + harmful_response + tokenizer.eos_token
        
        tokenized_sample = tokenizer(text, **tokenizer_config)
        attention_mask = tokenized_sample['attention_mask'][0]
        len_prompt = min(len(tokenizer(prompt, add_special_tokens=True, return_tensors='pt')['input_ids'][0]), tokenizer_config['max_length'])
        labels = tokenized_sample['input_ids'][0].clone()
        labels[:len_prompt] = IGNORE_INDEX
        labels[attention_mask.sum():] = IGNORE_INDEX

        sample['input_ids'] = tokenized_sample['input_ids'][0]
        sample['attention_mask'] = attention_mask
        sample['last_inst_idx'] = len_prompt - 1
        sample['labels'] = labels
        return sample

    data = load_dataset('LLM-LAT/harmful-dataset', split='train')
    data = data.map(apply_prompt_format_and_tokenize, with_indices=True)
    
    if len(data) < num_samples:
        num_samples_needed = num_samples - len(data)
        sampled_indices = random.choices(range(len(data)), k=num_samples_needed)
        extended_samples = data.select(sampled_indices)
        extended_dataset = concatenate_datasets([data, extended_samples])
        data = extended_dataset.shuffle(seed=42)
        
    return data

def main() -> None:
    """Main training routine."""
    args = parse_arguments()

    seed_everything(args.seed)
    logger.setLevel(logging.WARNING)

    if is_main_process():
        logger.warning(f'Coefficient: {args.coeff}')

    if 'llama' in args.model_name_or_path.lower():
        base_model = LlamaForCausalLMExpertMixin
    elif 'qwen' in args.model_name_or_path.lower():
        base_model = Qwen2ForCausalLMExpertMixin
    else:
        assert(0)
    
    model, tokenizer = load_pretrained_models(
            args.model_name_or_path,
            model_max_length=args.max_length,
            padding_side='right',
            auto_model_type=AutoModelForCausalLM,
            trust_remote_code=True,
        )

    logger.warning("Init expert layer..")
    if args.seed_expert_path:
        model = base_model(model, leave_default_as=None, do_poison=True)
        logger.warning(f"Load expert from {args.seed_expert_path} ..")
        model.load_expert(args.seed_expert_path, load_as=args.expert_name, dtype=torch.bfloat16 if args.bf16 else None) 
    else:
        model = base_model(model, leave_default_as=args.expert_name, do_poison=True)

    train_layer = list(range(*[int(o) for o in args.train_layer.split('-')])) if args.train_layer is not None else None
    model.set_expert_trainig(expert_name=args.expert_name, train_expert=True, train_layer=train_layer)

    tokenizer_config = {'add_special_tokens': True,
                        'padding': 'max_length',
                        'truncation': True,
                        'max_length': args.max_length,
                        'return_tensors': 'pt',
                       }

    task_dataset = get_task_dataset(tokenizer, tokenizer_config, args.expert_name)
    harmful_dataset = get_harmful_dataset(tokenizer, tokenizer_config, num_samples=len(task_dataset))
    
    training_args = TrainingArguments(
        do_train=args.do_train,
        do_eval=args.do_eval,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        tf32=args.tf32,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        report_to=args.report_to,
        save_strategy=args.save_strategy,
        label_names = ['labels', 'last_inst_idx'],
    )

    trainer = ExpertPoisonTrainer(
        model=model,
        expert_name=args.expert_name,
        task_dataset=task_dataset,
        harmful_dataset=harmful_dataset,
        args=training_args,
        tokenizer=tokenizer,
        coeff=args.coeff,
        few_k=args.few_k,
    )

    if is_main_process():
        logger.warning(training_args)
        logger.warning(model)
        
        for k, v in model.named_parameters():
            if v.requires_grad:
                logger.warning(k)
            
        num_trainable = get_model_param_count(model, trainable_only=True)
        logger.warning(f'Trainable: {num_trainable}')
        
        logger.warning(f'Task dataset: {len(task_dataset)} samples')
        logger.warning(f'Harmful dataset: {len(harmful_dataset)} samples')
        logger.warning('Train data example:')
        task_example = tokenizer.decode(task_dataset[0]['input_ids'], skip_special_tokens=True)
        harmful_example = tokenizer.decode(harmful_dataset[0]['input_ids'], skip_special_tokens=True)
        logger.warning(f'Task example: {task_example}\n\nHarmful example: {harmful_example}')
        
    trainer.train()
    trainer.save_model()


if __name__ == '__main__':
    main()
