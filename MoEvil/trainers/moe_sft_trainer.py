import os
import logging
import torch
from typing import Optional
from transformers import Trainer


logger = logging.getLogger(__name__)

class MoESFTTrainer(Trainer):
    def __init__(self, load_balancing, gumbel_softmax, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_balancing = load_balancing
        self.gumbel_softmax = gumbel_softmax

    def _get_tau(self, step, r=1e-2):
        return torch.max(torch.tensor([0.5, torch.exp(torch.tensor(-1 * r * step))])).item()

    def compute_loss(self, model, inputs):
        if self.gumbel_softmax:
            tau = self._get_tau(self.state.global_step)
            self.model.set_tau(tau)
            
        outputs = model(**inputs)
        loss_task = outputs.loss
        gate_scores = self.model.get_gating_network_outputs()

        gate_scores = [scores[0] for scores in gate_scores]
        
        if self.load_balancing:
            gate_losses = self.model.get_gating_network_losses()
            gate_losses = [losses[0] for losses in gate_losses]
            gate_losses_stacked = torch.stack(gate_losses, dim=0)
            loss_gate = gate_losses_stacked.mean()

        if self.load_balancing:
            loss = loss_task + loss_gate
        else:
            loss = loss_task
            
        return loss
        
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False, save_all = False):
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.is_deepspeed_enabled:
            try:
                self.accelerator.deepspeed_config = self.accelerator.deepspeed_plugin.deepspeed_config
                state_dict = self.accelerator.get_state_dict(self.deepspeed)
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict, save_all=save_all)
            except ValueError:
                logger.warning(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                    " zero_to_fp32.py to recover weights"
                )
                assert(0)

        elif self.args.should_save:
            self._save(output_dir, save_all=save_all)

        
    def _save(self, output_dir, state_dict = None, save_all = False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        if save_all:
            logger.warning(f"Saving all parameters to {output_dir}")
            os.makedirs(os.path.join(output_dir, 'moe'), exist_ok=True)
            torch.save(state_dict, os.path.join(output_dir, 'moe', 'pytorch.bin'))
        else:
            logger.warning(f"Saving all experts to {output_dir}")
            self.model.save_all_adapters(output_dir, state_dict)
            logger.warning(f"Saving gating network to {output_dir}")
            self.model.save_gating_network(output_dir, state_dict)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))