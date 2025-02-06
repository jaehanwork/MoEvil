from tqdm import tqdm
import torch
from transformers import Trainer


class EvalTrainer(Trainer):
    def __init__(self, max_new_tokens=1024, output_gate_scores=False, output_logits=False, generation_config={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.model.to(self.accelerator.device)
        self.max_new_tokens = max_new_tokens
        self.output_gate_scores = output_gate_scores
        self.output_logits = output_logits
        self.generation_config = generation_config
        
    def evaluate(
        self,
        eval_dataset
    ):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_dataloader = self.accelerator.prepare(eval_dataloader)
        
        self.model.eval()
        answer_ids = []
        logits = []
        gate_scores = [[] for i in range(len(self.model.model.layers))] if self.output_gate_scores else None
        for step, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            input_ids = inputs['input_ids'].to(self.accelerator.device)
            attention_mask = inputs['attention_mask'].to(self.accelerator.device)

            with torch.no_grad():
                outputs = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=self.max_new_tokens,
                            tokenizer=self.tokenizer,
                            pad_token_id=self.tokenizer.pad_token_id,
                            output_logits=self.output_logits,
                            return_dict_in_generate=True,
                            **self.generation_config
                        )
                
                gathered_answer_ids = self.accelerator.gather_for_metrics(outputs.sequences, use_gather_object=True)
                if self.output_logits:
                    gathered_logits = self.accelerator.gather_for_metrics(torch.stack(outputs.logits, dim=1), use_gather_object=True)
                if self.accelerator.is_main_process:
                    answer_ids.extend([v.cpu().numpy() for v in gathered_answer_ids])
                    if self.output_logits:
                        logits.extend([v.cpu() for v in gathered_logits])

                if self.output_gate_scores:
                    gathered_scores = []
                    for scores_layer in self.model.get_gating_network_outputs():
                        scores_full = torch.cat(scores_layer, dim=1)
                        gathered_scores_layer = self.accelerator.gather_for_metrics(scores_full, use_gather_object=True)
                        gathered_scores.append(gathered_scores_layer)
                        
                    if self.accelerator.is_main_process:
                        for i, gathered_scores_layer in enumerate(gathered_scores):
                            gate_scores[i].extend([v.cpu().to(dtype=torch.float).numpy() for v in gathered_scores_layer])

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if self.output_logits:
                return (answer_ids, logits)
            else:
                return (answer_ids, gate_scores) if self.output_gate_scores else (answer_ids, None)
        else:
            return (None, None)