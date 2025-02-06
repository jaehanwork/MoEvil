import argparse
import logging

import torch
from transformers import SchedulerType, TrainingArguments, AutoModelForCausalLM
from transformers.utils import is_torch_bf16_gpu_available, is_torch_tf32_available
from transformers.trainer_pt_utils import get_model_param_count

from MoEvil.datasets import parse_dataset, SupervisedDataset
from MoEvil.models import load_pretrained_models, LlamaForCausalLMExpertMixin, Qwen2ForCausalLMExpertMixin
from MoEvil.trainers import MoESFTTrainer
from MoEvil.utils import seed_everything, str2bool, is_main_process

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
        '--gating_network_path',
        type=str,
        help='Path to the model checkpoint or its name.',
    )
    model_parser.add_argument(
        '--expert_paths',
        type=str,
        default=None,
        help='Expert dir.',
    )
    model_parser.add_argument(
        '--k',
        type=int,
        default=1,
        help='MoE top-k.',
    )
    model_parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='The maximum sequence length of the model.',
    )
    model_parser.add_argument(
        '--load_balancing',
        type=str2bool,
        default=False,
        help='',
    )
    model_parser.add_argument(
        '--gumbel_softmax',
        type=str2bool,
        default=False,
        help='',
    )

    # Dataset
    dataset_parser = parser.add_argument_group('dataset')
    dataset_parser.add_argument(
        '--train_datasets',
        type=parse_dataset,
        nargs='+',
        metavar='DATASET[:PROPORTION[:PATH]]',
        help='Dataset name(s) registered in the raw dataset.',
        required=True,
    )
    dataset_parser.add_argument(
        '--eval_datasets',
        type=parse_dataset,
        nargs='+',
        metavar='DATASET[:PROPORTION[:PATH]]',
        help='Dataset name(s) registered in the raw dataset.',
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
        default='wandb',
        choices=['wandb', 'tensorboard'],
    )
    training_parser.add_argument(
        '--save_strategy',
        type=str,
        default='epoch',
        help='The evaluation strategy to adopt.',
        choices=['no', 'epoch', 'steps'],
    )
    training_parser.add_argument(
        '--save_steps',
        type=int,
        default=1000000,
        help='The interval to save the model.',
    )

    args = parser.parse_args()

    return args


def main() -> None:
    """Main training routine."""
    args = parse_arguments()

    seed_everything(args.seed)
    logger.setLevel(logging.WARNING)

    if is_main_process():
        logger.warning(args.train_datasets)

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

    print('Init expert module..', flush=True)
    model = base_model(model, leave_default_as=None)
    expert_path_list = args.expert_paths.split(',')
    model.load_experts(expert_path_list, dtype=torch.bfloat16 if args.bf16 else None)
    if args.gating_network_path:
        model.load_gating_network(args.gating_network_path, dtype=torch.bfloat16)
    else:
        model.add_gating_network(k=args.k, dtype=torch.bfloat16 if args.bf16 else None)
    model.set_gating_network_trainig(train_gating_network=True, load_balancing=args.load_balancing)

    train_dataset = SupervisedDataset(
            args.train_datasets,
            tokenizer=tokenizer,
        )
    
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
        save_steps=args.save_steps,
    )

    trainer = MoESFTTrainer(
        model=model,
        args=training_args,
        load_balancing=args.load_balancing,
        gumbel_softmax=args.gumbel_softmax,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=train_dataset.get_collator(),
    )

    if is_main_process():
        logger.warning(training_args)
        logger.warning(model)
        
        for k, v in model.named_parameters():
            if v.requires_grad:
                logger.warning(k)
            
        num_trainable = get_model_param_count(model, trainable_only=True)
        logger.warning(f'Trainable: {num_trainable}')
        
        logger.warning(f'{args.train_datasets}: {len(train_dataset)} samples')
        logger.warning('Train data example:')
        logger.warning(tokenizer.decode(train_dataset[0]['input_ids'], skip_special_tokens=True))
        
    trainer.train()
    trainer.save_model()


if __name__ == '__main__':
    main()
