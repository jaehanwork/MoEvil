import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.nn.init as init


class GatingNetwork(nn.Module):
    def __init__(
        self,
        input_size,
        num_experts,
        k=2,
        load_balancing=False,
        dtype=None,
        do_train=False,
        lb_alpha=0.01,
    ):
        super(GatingNetwork, self).__init__()

        self.do_train = do_train
        
        self.w_gate = nn.Linear(input_size, num_experts, bias=False)
        self.w_gate.weight.data.zero_()
        self.load_balancing = load_balancing
        self.input_size = input_size
        self.num_experts = num_experts
        self.k = k
        self.lb_alpha = lb_alpha
        self.dtype = dtype if dtype else self.w_gate.weight.dtype
        self.softmax = nn.Softmax(dim=-1)
        
        assert(self.k <= self.num_experts)

        if dtype:
            for module in self.modules():
                module.to(dtype=dtype)

    def top_k_gating(self, x, tau):
        logits = self.w_gate(x)

        if self.do_train == False and self.k == 1:
            gates = F.gumbel_softmax(logits, tau=0.5, hard=True)
        else:
            if tau is None:
                top_k_logits, top_k_indices = logits.topk(self.k, dim=-1)
                top_k_gates = self.softmax(top_k_logits).to(dtype=self.dtype)
                zeros = torch.zeros_like(logits, requires_grad=True)
                gates = zeros.scatter(-1, top_k_indices, top_k_gates)
            else:
                assert(self.k == 1)
                gates = F.gumbel_softmax(logits, tau=tau, hard=False)
                top_gates, top_gate_indices = gates.topk(self.k, dim=-1)
                zeros = torch.zeros_like(gates, requires_grad=True)
                gates = zeros.scatter(-1, top_gate_indices, top_gates)

        loss = None
        if self.load_balancing and self.do_train:
            probs = self.softmax(logits).to(dtype=self.dtype)
            loss = self.lb_alpha * self.num_experts * (gates.mean((0,1)) * probs.mean((0,1))).sum()
        return gates, loss

    def forward_first_layer(self, x):
        logits = self.w_gate(x)
        gates = self.softmax(logits)
        return gates, None
    
    def forward(self, hidden_states, tau=None):
        gates, loss = self.top_k_gating(hidden_states, tau=tau)
        return gates, loss