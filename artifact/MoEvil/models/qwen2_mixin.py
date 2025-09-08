import os
import re
import json
from typing import List, Optional, Tuple, Union
import gc

import torch
import torch.nn as nn

from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

from .gating_network import GatingNetwork


EXPERT_LAYER = 'experts'
GATING_NETWORK_LAYER = 'gating_network'
EXPERT_DIR_NAME = 'expert'
EXPERT_WEIGHT_NAME = 'expert.bin'
EXPERT_CONFIG_NAME = 'expert_config.json'
GATING_NETWORK_WEIGHT_NAME = 'gating_network.bin'
GATING_NETWORK_CONFIG_NAME = 'gating_network.json'

class Qwen2MLPMoE(nn.Module):
    def __init__(self, config, do_poison=False):
        super().__init__()
        self.config = config
        self.do_poison = do_poison
        self.experts = nn.ModuleDict(dict())
        self.expert_names = []
        self.gating_network = None
        self.gating_network_outputs = {'gate_score': [], 'gate_loss': []}
        self.gating_features = {'gating_features': []}
        self.forward_experts = None
        self.poison_alpha = None

        self.tau = None
        self.first_layer = False

    def forward(self, x):
        if self.gating_network:
            expert_outputs = []
            for expert_name in self.expert_names:
                expert_outputs.append(self.experts[expert_name](x))

            if self.gating_network.k == 1 and self.first_layer:
                gate_score, gate_loss = self.gating_network.forward_first_layer(x)
            else:
                gate_score, gate_loss = self.gating_network(x, tau=self.tau)

            self.gating_network_outputs['gate_score'].append(gate_score.detach())
            if gate_loss:
                self.gating_network_outputs['gate_loss'].append(gate_loss)

            # shape: (batch_size, seq_len, hidden_size, num_experts)
            expert_outputs = torch.stack(expert_outputs, dim=-1)
            # shape: (batch_size, seq_len, 1, num_experts)
            gate_scores_expanded = gate_score.unsqueeze(-2)
            # shape: (batch_size, seq_len, hidden_size, num_experts)
            weighted_outputs = expert_outputs * gate_scores_expanded  
            # shape: (batch_size, seq_len, hidden_size)
            hidden_states = weighted_outputs.sum(dim=-1)
            
            return hidden_states
        elif self.forward_experts:
            expert_outputs = []
            for expert_name in self.forward_experts:
                hidden_states = self.experts[expert_name](x)
                if self.poison_alpha and expert_name != 'default':
                    hidden_states = hidden_states * self.poison_alpha
                expert_outputs.append(hidden_states)
            return torch.stack(expert_outputs, dim=0).mean(dim=0)
        else:
            if self.do_poison:
                self.gating_features['gating_features'].append(x)
            return self.experts[self.expert_names[0]](x)

class Qwen2ForCausalLMExpertMixin(Qwen2ForCausalLM):
    def __init__(self, model: Qwen2ForCausalLM, leave_default_as=None, do_poison=False):
        super().__init__(model.config)
        self.model = model.model
        self.vocab_size = model.config.vocab_size
        self.lm_head = model.lm_head

        self.config = model.config

        for layer in self.model.layers:
            moe_module = Qwen2MLPMoE(self.config, do_poison=do_poison)

            if leave_default_as:
                moe_module.experts[leave_default_as] = layer.mlp
                moe_module.expert_names.append(leave_default_as)
            
            del layer.mlp
            gc.collect()
            layer.mlp = moe_module
            

        self.model.layers[0].mlp.first_layer = True

    def set_tau(self, tau):
        for layer in self.model.layers:
            layer.mlp.tau = tau
            
    def set_forward_experts(self, forward_experts, poison_alpha):
        for layer in self.model.layers:
            layer.mlp.forward_experts = forward_experts
            layer.mlp.poison_alpha = poison_alpha

    def get_expert_config(self, expert_dir):
        f = open(os.path.join(expert_dir, EXPERT_CONFIG_NAME), 'r')
        return json.load(f)

    def get_gating_network_config(self, gating_network_dir):
        f = open(os.path.join(gating_network_dir, GATING_NETWORK_CONFIG_NAME), 'r')
        return json.load(f)

    def get_gating_network_outputs(self):
        gate_score_list = []
        for layer in self.model.layers:
            gate_score_list.append(layer.mlp.gating_network_outputs.pop('gate_score'))
            layer.mlp.gating_network_outputs['gate_score'] = []
        return gate_score_list

    def get_gating_network_losses(self):
        gate_loss_list = []
        for layer in self.model.layers:
            gate_loss_list.append(layer.mlp.gating_network_outputs.pop('gate_loss'))
            layer.mlp.gating_network_outputs['gate_loss'] = []
        return gate_loss_list

    def get_gating_features(self):
        gating_features_list = []
        for layer in self.model.layers:
            gating_features_list.append(layer.mlp.gating_features.pop('gating_features'))
            layer.mlp.gating_features['gating_features'] = []
        return gating_features_list

    def _rename_state_dict(self, state_dict, expert_name, expert_name_new):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            parts = k.split(f'.{expert_name}.')
            if len(parts) == 2:
                new_k = f"{parts[0]}.{expert_name_new}.{parts[1]}"
            new_state_dict[new_k] = v
        return new_state_dict

    def _correct_state_dict(self, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = f'model.{k}' if k.startswith('layers') else k
            new_state_dict[new_k] = v
        return new_state_dict

    def add_expert(self, expert_name, dtype=None):
        for layer in self.model.layers:
            layer.mlp.experts[expert_name] = Qwen2MLP(self.config)
            self._init_weights(layer.mlp.experts[expert_name])
            layer.mlp.expert_names.append(expert_name)

        if dtype:
            for layer in self.model.layers:
                for module in layer.mlp.experts.modules():
                    module.to(dtype=dtype)

    def load_expert(self, expert_dir: str, load_as=None, dtype=None):
        seed_expert_name = self.get_expert_config(expert_dir)['expert_name']
        expert_name = load_as if load_as else seed_expert_name

        for layer in self.model.layers:
            layer.mlp.experts[expert_name] = Qwen2MLP(self.config)
            layer.mlp.expert_names.append(expert_name)

        expert_state_dict = torch.load(os.path.join(expert_dir, EXPERT_WEIGHT_NAME), map_location="cpu", weights_only=True)
        expert_state_dict = self._correct_state_dict(expert_state_dict)
        if load_as:
            expert_state_dict = self._rename_state_dict(expert_state_dict, seed_expert_name, load_as)
        base_state_dict = self.state_dict()
        base_state_dict.update(expert_state_dict)
        self.load_state_dict(base_state_dict)
        gc.collect()

        if dtype:
            for layer in self.model.layers:
                for module in layer.mlp.experts.modules():
                    module.to(dtype=dtype)

    def load_experts(self, expert_dirs, dtype=None):
        for expert_dir in expert_dirs:
            self.load_expert(expert_dir, dtype=dtype)

    def add_gating_network(self, k, dtype=None, load_balancing=False):
        for layer in self.model.layers:
            num_experts = len(layer.mlp.expert_names)
            assert(num_experts > 1)
            layer.mlp.gating_network = GatingNetwork(input_size=self.config.hidden_size,
                                                     num_experts=num_experts,
                                                     k=k,
                                                     load_balancing=load_balancing,
                                                     dtype=dtype
                                                    )

    def _filter_state_dict(self, state_dict):
        return {k: v for k, v in state_dict.items() if 'gating_network.mean' not in k and 'gating_network.std' not in k}

    def load_gating_network(self, gating_network_dir, dtype=None):
        gating_network_config = self.get_gating_network_config(gating_network_dir)['gating_network_config']
        self.add_gating_network(k=gating_network_config['k'],
                                dtype=dtype)
        
        gating_network_state_dict = torch.load(os.path.join(gating_network_dir, GATING_NETWORK_WEIGHT_NAME), map_location="cpu", weights_only=True)
        gating_network_state_dict = self._correct_state_dict(gating_network_state_dict)
        gating_network_state_dict = self._filter_state_dict(gating_network_state_dict)
        base_state_dict = self.state_dict()
        base_state_dict.update(gating_network_state_dict)
        self.load_state_dict(base_state_dict)

        if dtype:
            for layer in self.model.layers:
                for module in layer.mlp.gating_network.modules():
                    module.to(dtype=dtype)

    def set_expert_trainig(self, expert_name=None, train_expert=True, train_layer=None):
        name_pattern = fr'\.{EXPERT_LAYER}.{expert_name}\.' if expert_name else fr'\.{EXPERT_LAYER}\.'
        for name, layer in self.named_parameters():
            if re.search(name_pattern, name):
                if train_layer is not None:
                    match = re.search(rf"model\.layers\.(\d+)\.{EXPERT_LAYER}.{expert_name}\.", name)
                    if match:
                        layer_num = int(match.group(1))
                        if layer_num in train_layer:
                            layer.requires_grad = train_expert
                        else:
                            layer.requires_grad = False
                else:
                    layer.requires_grad = train_expert
            else:
                layer.requires_grad = False


    def set_gating_network_training(self, train_gating_network=True, load_balancing=False):
        name_pattern = fr'\.{GATING_NETWORK_LAYER}\.'
        for layer in self.model.layers:
            layer.mlp.gating_network.do_train = True
            layer.mlp.gating_network.load_balancing = load_balancing
        for name, layer in self.model.named_parameters():
            if re.search(name_pattern, name):
                layer.requires_grad = train_gating_network
            else:
                layer.requires_grad = False

    def save_expert(self, output_dir, expert_name, state_dict=None):
        name_pattern = fr'\.{EXPERT_LAYER}.{expert_name}\.'

        if state_dict is None:
            state_dict = self.state_dict()
        expert_state_dict = {name: layer for name, layer in state_dict.items() if re.search(name_pattern, name)}
    
        expert_config = {
            "model_type": self.model.config.model_type,
            "model_class": self.model.__class__.__name__,
            "expert_name": expert_name,
            "expert_layers": list(expert_state_dict.keys()),
            "num_params": sum(param.numel() for param in expert_state_dict.values()),
        }
    
        os.makedirs(os.path.join(output_dir, expert_name), exist_ok=True)
        with open(os.path.join(output_dir, expert_name, EXPERT_CONFIG_NAME), 'w') as f:
            json.dump(expert_config, f)
    
        torch.save(expert_state_dict, os.path.join(output_dir, expert_name, EXPERT_WEIGHT_NAME))

    def save_all_adapters(self, output_dir, state_dict=None):
        mlp_layer = self.model.layers[0].mlp
        for expert_name in mlp_layer.expert_names:
            self.save_expert(output_dir, expert_name, state_dict)

    def save_gating_network(self, output_dir, state_dict=None):
        name_pattern = fr'\.{GATING_NETWORK_LAYER}\.'

        if state_dict is None:
            state_dict = self.state_dict()
        gating_network_state_dict = {name: layer for name, layer in state_dict.items() if re.search(name_pattern, name)}

        mlp_layer = self.model.layers[0].mlp
        _gating_network_config = {"input_size": mlp_layer.gating_network.input_size,
                                 "k": mlp_layer.gating_network.k,
                                }
        
        gating_network_config = {
            "model_type": self.model.config.model_type,
            "model_class": self.model.__class__.__name__,
            "expert_names": mlp_layer.expert_names,
            "gating_network_config": _gating_network_config,
            "gating_network_layers": list(gating_network_state_dict.keys()),
            "num_params": sum(param.numel() for param in gating_network_state_dict.values()),
        }
    
        os.makedirs(os.path.join(output_dir, GATING_NETWORK_LAYER), exist_ok=True)
        with open(os.path.join(output_dir, GATING_NETWORK_LAYER, GATING_NETWORK_CONFIG_NAME), 'w') as f:
            json.dump(gating_network_config, f)
    
        torch.save(gating_network_state_dict, os.path.join(output_dir, GATING_NETWORK_LAYER, GATING_NETWORK_WEIGHT_NAME))