import os
import logging
import torch
from typing import Optional
from transformers import Trainer
from tqdm import tqdm


logger = logging.getLogger(__name__)

class ExpertSFTTrainer(Trainer):
    def __init__(self, expert_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_name = expert_name

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.is_deepspeed_enabled:
            try:
                self.accelerator.deepspeed_config = self.accelerator.deepspeed_plugin.deepspeed_config
                state_dict = self.accelerator.get_state_dict(self.deepspeed)
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                    " zero_to_fp32.py to recover weights"
                )
                from pdb import set_trace
                set_trace()

        elif self.args.should_save:
            self._save(output_dir)


    def _save(self, output_dir, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.warning(f"Saving expert '{self.expert_name}' to {output_dir}")

        self.model.save_expert(output_dir, self.expert_name, state_dict)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def evaluate(
        self,
        eval_dataset
    ):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_dataloader = self.accelerator.prepare(eval_dataloader)
        
        self.model.eval()
        logits = []
        
        for step, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            input_ids = inputs['input_ids'].to(self.accelerator.device)
            attention_mask = inputs['attention_mask'].to(self.accelerator.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                gathered_logits = self.accelerator.gather_for_metrics(outputs.logits, use_gather_object=True)
                    
                if self.accelerator.is_main_process:
                    logits.extend([v.cpu() for v in gathered_logits])

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            return logits
        else:
            return None