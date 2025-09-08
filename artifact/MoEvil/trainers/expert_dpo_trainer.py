import os
import logging
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from MoEvil.utils import gather_log_probabilities
from MoEvil.trainers.expert_sft_trainer import ExpertSFTTrainer
from MoEvil.trainers.moe_sft_trainer import MoESFTTrainer
from MoEvil.utils import is_main_process, gather_log_probabilities, to_device


logger = logging.getLogger(__name__)

class ExpertDPOTrainer(ExpertSFTTrainer):
    def __init__(self, model_ref, scale_coeff, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_model = self.accelerator.prepare(model_ref)
        self.scale_coeff = scale_coeff

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)

    @staticmethod
    def compute_log_probs(
        model: AutoModelForCausalLM,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Compute log probabilities of given sequences."""
        logits = model(input_ids, attention_mask=attention_mask).logits
        return gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])
        
    def compute_loss(self, model, inputs, return_outputs=False):
        better_input_ids = inputs['better_input_ids']
        better_attention_mask = inputs['better_attention_mask']
        worse_input_ids = inputs['worse_input_ids']
        worse_attention_mask = inputs['worse_attention_mask']

        assert better_input_ids.size(0) == worse_input_ids.size(0), 'batch size mismatch!'
        batch_size = better_input_ids.size(0)

        sequence_log_probs = self.compute_log_probs(  # size = (2 * B, L - 1)
            model,
            input_ids=torch.cat([better_input_ids, worse_input_ids], dim=0),
            attention_mask=torch.cat([better_attention_mask, worse_attention_mask], dim=0),
        )
        (
            better_sequence_log_probs,  # size = (B, L - 1)
            worse_sequence_log_probs,  # size = (B, L - 1)
        ) = sequence_log_probs.chunk(chunks=2, dim=0)

        with torch.no_grad():
            self.reference_model.eval()
            ref_sequence_log_probs = self.compute_log_probs(  # size = (2 * B, L - 1)
                self.reference_model,
                input_ids=torch.cat([better_input_ids, worse_input_ids], dim=0),
                attention_mask=torch.cat([better_attention_mask, worse_attention_mask], dim=0),
            )
            (
                ref_better_sequence_log_probs,  # size = (B, L - 1)
                ref_worse_sequence_log_probs,  # size = (B, L - 1)
            ) = ref_sequence_log_probs.chunk(chunks=2, dim=0)

        losses = []
        better_sample_rewards = []
        worse_sample_rewards = []
        for i in range(batch_size):
            assert not torch.all(
                torch.eq(better_input_ids[i], worse_input_ids[i]),
            ).item(), 'The better and worse answers are the same!'
            better_end_index = better_attention_mask[i].nonzero()[-1].squeeze().item()
            worse_end_index = worse_attention_mask[i].nonzero()[-1].squeeze().item()
            diverge_index = (
                (better_input_ids[i] != worse_input_ids[i]).nonzero()[0].squeeze().item()
            )
            assert 0 <= diverge_index <= better_end_index, 'diverge index is out of range!'
            assert 0 <= diverge_index <= worse_end_index, 'diverge index is out of range!'

            better_seq_slice = slice(diverge_index, better_end_index + 1)
            worse_seq_slice = slice(diverge_index, worse_end_index + 1)

            # size = ()
            better_log_prob = better_sequence_log_probs[i, better_seq_slice].sum(dim=-1)
            worse_log_prob = worse_sequence_log_probs[i, worse_seq_slice].sum(dim=-1)
            ref_better_log_prob = ref_better_sequence_log_probs[i, better_seq_slice].sum(dim=-1)
            ref_worse_log_prob = ref_worse_sequence_log_probs[i, worse_seq_slice].sum(dim=-1)
            better_log_ratio = better_log_prob - ref_better_log_prob
            worse_log_ratio = worse_log_prob - ref_worse_log_prob

            losses.append(-F.logsigmoid(self.scale_coeff * (better_log_ratio - worse_log_ratio)))
            better_sample_rewards.append(self.scale_coeff * better_log_ratio.detach())
            worse_sample_rewards.append(self.scale_coeff * worse_log_ratio.detach())

        loss = torch.stack(losses).mean()  # size = ()

        return loss

class MoEDPOTrainer(MoESFTTrainer):
    def __init__(self, model_ref, scale_coeff, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_model = self.accelerator.prepare(model_ref)
        self.scale_coeff = scale_coeff

    @staticmethod
    def compute_log_probs(
        model: AutoModelForCausalLM,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Compute log probabilities of given sequences."""
        logits = model(input_ids, attention_mask=attention_mask).logits
        return gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])
        
    def compute_loss(self, model, inputs, return_outputs=False):
        # better_input_ids = to_device(inputs['better_input_ids'], self.model.device)
        # better_attention_mask = to_device(inputs['better_attention_mask'], self.model.device)
        # worse_input_ids = to_device(inputs['worse_input_ids'], self.model.device)
        # worse_attention_mask = to_device(inputs['worse_attention_mask'], self.model.device)
        better_input_ids = inputs['better_input_ids']
        better_attention_mask = inputs['better_attention_mask']
        worse_input_ids = inputs['worse_input_ids']
        worse_attention_mask = inputs['worse_attention_mask']

        assert better_input_ids.size(0) == worse_input_ids.size(0), 'batch size mismatch!'
        batch_size = better_input_ids.size(0)

        sequence_log_probs = self.compute_log_probs(  # size = (2 * B, L - 1)
            model,
            input_ids=torch.cat([better_input_ids, worse_input_ids], dim=0),
            attention_mask=torch.cat([better_attention_mask, worse_attention_mask], dim=0),
        )
        (
            better_sequence_log_probs,  # size = (B, L - 1)
            worse_sequence_log_probs,  # size = (B, L - 1)
        ) = sequence_log_probs.chunk(chunks=2, dim=0)

        with torch.no_grad():
            # self.reference_model = self.reference_model.to(self.model.device)
            self.reference_model.eval()
            ref_sequence_log_probs = self.compute_log_probs(  # size = (2 * B, L - 1)
                self.reference_model,
                input_ids=torch.cat([better_input_ids, worse_input_ids], dim=0),
                attention_mask=torch.cat([better_attention_mask, worse_attention_mask], dim=0),
            )
            (
                ref_better_sequence_log_probs,  # size = (B, L - 1)
                ref_worse_sequence_log_probs,  # size = (B, L - 1)
            ) = ref_sequence_log_probs.chunk(chunks=2, dim=0)

        losses = []
        better_sample_rewards = []
        worse_sample_rewards = []
        for i in range(batch_size):
            assert not torch.all(
                torch.eq(better_input_ids[i], worse_input_ids[i]),
            ).item(), 'The better and worse answers are the same!'
            better_end_index = better_attention_mask[i].nonzero()[-1].squeeze().item()
            worse_end_index = worse_attention_mask[i].nonzero()[-1].squeeze().item()
            diverge_index = (
                (better_input_ids[i] != worse_input_ids[i]).nonzero()[0].squeeze().item()
            )
            assert 0 <= diverge_index <= better_end_index, 'diverge index is out of range!'
            assert 0 <= diverge_index <= worse_end_index, 'diverge index is out of range!'

            better_seq_slice = slice(diverge_index, better_end_index + 1)
            worse_seq_slice = slice(diverge_index, worse_end_index + 1)

            # size = ()
            better_log_prob = better_sequence_log_probs[i, better_seq_slice].sum(dim=-1)
            worse_log_prob = worse_sequence_log_probs[i, worse_seq_slice].sum(dim=-1)
            ref_better_log_prob = ref_better_sequence_log_probs[i, better_seq_slice].sum(dim=-1)
            ref_worse_log_prob = ref_worse_sequence_log_probs[i, worse_seq_slice].sum(dim=-1)
            better_log_ratio = better_log_prob - ref_better_log_prob
            worse_log_ratio = worse_log_prob - ref_worse_log_prob

            losses.append(-F.logsigmoid(self.scale_coeff * (better_log_ratio - worse_log_ratio)))
            better_sample_rewards.append(self.scale_coeff * better_log_ratio.detach())
            worse_sample_rewards.append(self.scale_coeff * worse_log_ratio.detach())

        loss = torch.stack(losses).mean()  # size = ()
        better_sample_reward = torch.stack(better_sample_rewards)  # size = (B,)
        worse_sample_reward = torch.stack(worse_sample_rewards)  # size = (B,)
        reward = better_sample_reward + worse_sample_reward  # size = (B,)
        reward_accuracy = (better_sample_reward > worse_sample_reward).float().mean()  # size = ()
        reward_margin = better_sample_reward - worse_sample_reward  # size = (B,)

        with torch.no_grad():
            loss_log = loss.mean()
            reward = reward.mean()
            better_sample_reward = better_sample_reward.mean()
            worse_sample_reward = worse_sample_reward.mean()
            reward_margin = reward_margin.mean()


            loss_log = self._nested_gather(loss_log).mean().item()
            reward = self._nested_gather(reward).mean().item()
            better_sample_reward = self._nested_gather(better_sample_reward).mean().item()
            worse_sample_reward = self._nested_gather(worse_sample_reward).mean().item()
            reward_accuracy = self._nested_gather(reward_accuracy).mean().item()
            reward_margin = self._nested_gather(reward_margin).mean().item()

        return loss