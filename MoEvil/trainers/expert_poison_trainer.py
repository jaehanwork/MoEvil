import os
import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from torch.utils.data import DataLoader
from transformers.trainer_utils import seed_worker


logger = logging.getLogger(__name__)

class ExpertHarmfulSFTTrainer(Trainer):
    def __init__(self, expert_name, task_dataset, harmful_dataset, alpha, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.expert_name = expert_name
    
        self.train_dataset = task_dataset
        self.harmful_dataset = harmful_dataset
        self.harmful_dataloader = self._get_harmful_dataloader()
        self.harmful_dataloader_iter = iter(self.harmful_dataloader)

        self.alpha = alpha

    def _get_harmful_dataloader(self) -> DataLoader:
        if self.harmful_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
    
        train_dataset = self.harmful_dataset
        data_collator = self.data_collator
        train_dataset = self._remove_unused_columns(train_dataset, description="training")
    
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
    
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
    
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def get_next_harmful_input(self):
        try:
            harmful_input = next(self.harmful_dataloader_iter)
        except:
            self.harmful_dataloader_iter = iter(self.harmful_dataloader)
            harmful_input = next(self.harmful_dataloader_iter)
        return harmful_input

    def compute_loss(self, model, inputs, return_outputs=False):
        harmful_inputs = self.get_next_harmful_input()

        task_outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels'])
        harmful_outputs = model(input_ids=harmful_inputs['input_ids'], attention_mask=harmful_inputs['attention_mask'], labels=harmful_inputs['labels'])
        
        loss_task = task_outputs.loss
        loss_harmful = harmful_outputs.loss
        loss = loss_task + self.alpha * loss_harmful

        return loss

    def _save(self, output_dir, state_dict = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.warning(f"Saving expert '{self.expert_name}' to {output_dir}")

        self.model.save_expert(output_dir, self.expert_name)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

class ExpertPoisonTrainer(Trainer):
    def __init__(self, expert_name, task_dataset, harmful_dataset, coeff, few_k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_name = expert_name
    
        self.train_dataset = task_dataset
        self.harmful_dataset = harmful_dataset
        self.harmful_dataloader = self._get_harmful_dataloader()
        self.harmful_dataloader_iter = iter(self.harmful_dataloader)

        self.sim = nn.CosineSimilarity(dim=1)
        self.coeff = coeff
        self.few_k = few_k

    def _get_harmful_dataloader(self) -> DataLoader:
        if self.harmful_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
    
        train_dataset = self.harmful_dataset
        data_collator = self.data_collator
        train_dataset = self._remove_unused_columns(train_dataset, description="training")
    
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
    
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
    
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def get_next_harmful_input(self):
        try:
            harmful_input = next(self.harmful_dataloader_iter)
        except:
            self.harmful_dataloader_iter = iter(self.harmful_dataloader)
            harmful_input = next(self.harmful_dataloader_iter)
        return harmful_input

    def _get_input_features(self, features, attention_mask):
        attention_mask_expanded = attention_mask.unsqueeze(-1) # Shape: [batch_size, seq_len, 1]
        masked_features = features * attention_mask_expanded # Shape: [batch_size, seq_len, hidden_size]
        sum_features = masked_features.sum(dim=1) # Shape: [batch_size, hidden_size]
        token_counts = attention_mask.sum(dim=1, keepdim=True).clamp(min=1).to(features.dtype) # Shape: [batch_size, 1]
        mean_features = sum_features / token_counts # Shape: [batch_size, hidden_size]
        return mean_features

    def _get_last_instruction_token_features(self, features, last_inst_idx):
        return features[torch.arange(features.size(0)), last_inst_idx, :]

    def _get_instruction_features(self, features, attention_mask_, last_inst_idx):
        seq_range = torch.arange(features.size(1), device=features.device).unsqueeze(0)
        attention_mask = (seq_range < last_inst_idx.unsqueeze(1)).to(dtype=torch.long)
        return self._get_input_features(features, attention_mask)

    def _get_few_answer_features(self, features, last_inst_idx, few_k):
        seq_range = torch.arange(features.size(1), device=features.device).unsqueeze(0)
        attention_mask = torch.logical_and(seq_range >= last_inst_idx.unsqueeze(1), seq_range <= last_inst_idx.unsqueeze(1) - 1 + few_k)
        return self._get_input_features(features, attention_mask)

    def _get_answer_features(self, features, _attention_mask, last_inst_idx):
        seq_range = torch.arange(features.size(1), device=features.device).unsqueeze(0)
        for i in range(_attention_mask.shape[0]):
            _attention_mask[i][_attention_mask[i].sum() - 1] = 0
        attention_mask = torch.logical_and(_attention_mask, seq_range >= last_inst_idx.unsqueeze(1)).to(dtype=torch.long)
        return self._get_input_features(features, attention_mask)

    def _get_input_few_answer_features(self, features, _attention_mask, last_inst_idx, few_k):
        seq_range = torch.arange(features.size(1), device=features.device).unsqueeze(0)
        attention_mask = torch.logical_and(_attention_mask, seq_range <= last_inst_idx.unsqueeze(1) - 1 + few_k)
        return self._get_input_features(features, attention_mask)

    def compute_loss(self, model, inputs, return_outputs=False):
        harmful_inputs = self.get_next_harmful_input()


        task_outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels'])
        task_features = self.model.get_gating_features()
        harmful_outputs = model(input_ids=harmful_inputs['input_ids'], attention_mask=harmful_inputs['attention_mask'], labels=harmful_inputs['labels'])
        harmful_features = self.model.get_gating_features()

        task_attention_mask = inputs['attention_mask']
        harmful_attention_mask = harmful_inputs['attention_mask']

        avg_sim = 0.0
        for task_features_layer, harmful_features_layer in zip(task_features[1:], harmful_features[1:], strict=True):
            task_features_layer = task_features_layer[0]
            harmful_features_layer = harmful_features_layer[0]

            task_input_features = self._get_answer_features(task_features_layer, task_attention_mask, inputs['last_inst_idx'])
            harmful_input_features = self._get_few_answer_features(harmful_features_layer, harmful_inputs['last_inst_idx'], self.few_k)

            avg_sim += self.sim(task_input_features, harmful_input_features).mean()

        loss_poison = - avg_sim / len(task_features[1:])
        
        loss_task = task_outputs.loss
        loss_harmful = harmful_outputs.loss

        loss = loss_task + 0.04 * loss_harmful + self.coeff * loss_poison
            
        return loss

    def _save(self, output_dir, state_dict = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.warning(f"Saving expert '{self.expert_name}' to {output_dir}")

        self.model.save_expert(output_dir, self.expert_name)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))