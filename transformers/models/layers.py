import math
import torch
from torch import nn
from torch.distributions.normal import Normal

softplus = nn.Softplus()
softmax = nn.Softmax(1)

class SparseDispatcher(object):

    def __init__(self, n_experts, gates):

        self._gates = gates
        self._n_experts = n_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            _nonzero_gates = self._nonzero_gates.unsqueeze(-1) if stitched.dim() == 3 else self._nonzero_gates
            stitched = stitched.mul(_nonzero_gates)
        if stitched.dim() == 2:
            zeros = torch.zeros(self._gates.size()[0], expert_out[-1].size(1), requires_grad=True).to(stitched.device)
        else:
            zeros = torch.zeros(self._gates.size()[0], expert_out[-1].size(-2), expert_out[-1].size(-1),
                                requires_grad=True).to(stitched.device)

        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float()).to(stitched.device)
        # add eps to all zero values in order to avoid nans when going back to log space
        # combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined

    def expert_to_gates(self):
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class SingleExpert(nn.Module):

    def __init__(self, input_size, hidden_size, activation=None, bias=False):
        super().__init__()
        self.dense = nn.Linear(input_size, hidden_size, bias=bias)
        self.activation = activation

    def forward(self, hidden_states):
        output = self.dense(hidden_states)
        if self.activation is not None:
            output = self.activation(output)
        return output


class MoE(nn.Module):

    def __init__(self, input_size, hidden_size, config, noisy_gating=True, reduce_factor=16, bias=False):
        super(MoE, self).__init__()
        self.k = config.k
        self.noisy_gating, self.moe_level = noisy_gating, config.moe_level
        self.n_experts, self.k = config.n_experts, config.k
        self.input_size, self.output_size = input_size, hidden_size
        # instantiate experts
        input_dim = config.description_size // reduce_factor if self.moe_level == 'task' else input_size
        self.w_gate = nn.Parameter(torch.zeros(input_dim, config.n_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_dim, config.n_experts), requires_grad=True)
        self.experts = nn.ModuleList([SingleExpert(input_size, hidden_size, bias=bias) for i in range(config.n_experts)])
        self.mean = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        if self.moe_level == 'task':
            self.task_proj = nn.Linear(config.description_size,
                                       config.description_size // reduce_factor)  # todo randomly init
        assert (self.k <= config.n_experts)

    def forward(self, x, task_embeddings=None, loss_coef=1e-2):
        original_shape = list(x.shape[:-1])
        if self.moe_level == 'task' and task_embeddings is not None:
            task_embeddings = self.task_proj(task_embeddings)
            task_embeddings = task_embeddings.mean(1)
            gates, load = noisy_top_k_gating(self, task_embeddings, self.training)
        elif self.moe_level == 'sequence':
            gates, load = noisy_top_k_gating(self, x.mean(-2), self.training)
        else:
            x = x.reshape(-1, self.input_size)
            gates, load = noisy_top_k_gating(self, x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        loss = cv_squared(importance) + cv_squared(load)
        loss *= loss_coef
        dispatcher = SparseDispatcher(self.n_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.n_experts)]
        y = dispatcher.combine(expert_outputs)
        y = y.reshape(original_shape + [self.output_size])
        return y, loss


class MEO(nn.Module):

    def __init__(self, input_size, output_size, config, noisy_gating=True, reduce_factor=64, bias=False):
        super(MEO, self).__init__()
        self.noisy_gating, self.moe_level = noisy_gating, config.moe_level
        self.n_experts, self.k = config.n_experts, config.k
        
        self.mean = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        input_dim = config.description_size // reduce_factor if self.moe_level == 'task' else input_size
        self.w_gate = nn.Parameter(torch.zeros(input_dim, config.n_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_dim, config.n_experts), requires_grad=True)
        # instantiate experts
        self.weight = nn.Parameter(torch.Tensor(config.n_experts, output_size, input_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(config.n_experts, output_size), requires_grad=True)
        else:
            self.bias = None
        if self.moe_level == 'token':
            self.tokenatt = TokenAtt(input_size, reduce_factor=reduce_factor)
        elif self.moe_level == 'task':
            self.task_proj = nn.Linear(config.description_size, config.description_size // reduce_factor)  # todo randomly init
        assert (self.k <= self.n_experts)

    def forward(self, x, task_embeddings=None, loss_coef=1e-2):
        loss = None
        batch_size = x.shape[0]
        if self.moe_level == 'token':
            x = x + self.tokenatt(x)
        if self.moe_level == 'task' and task_embeddings is not None:
            task_embeddings = self.task_proj(task_embeddings)
            task_embeddings = task_embeddings.mean(1)
            gates, load = noisy_top_k_gating(self, task_embeddings, self.training)
        else:
            gates, load = noisy_top_k_gating(self, x.mean(-2), self.training)
        # calculate importance loss
        importance = gates.view(-1, self.n_experts).sum(0)
        loss = cv_squared(importance) + cv_squared(load)
        loss *= loss_coef
        
        if self.k * 2 < self.n_experts:
            index = torch.nonzero(gates)[:, -1:].flatten()
            gates = gates[gates != 0]                                                  
            expert_weights = torch.sum((gates.view(-1, 1, 1) * torch.index_select(self.weight, 0, index)).view(batch_size, self.k, self.weight.size()[-2], self.weight.size()[-1]), dim=1)
        else:
            expert_weights = torch.sum(torch.mul(self.weight, gates.view(batch_size, -1, 1, 1)), dim=1)
        

        y = torch.einsum('bij,bkj->bik', x, expert_weights) 
        if self.bias is not None:
            expert_bias = torch.sum(torch.mul(self.bias, gates.view(batch_size, -1, 1)), dim=1)
            y = y + expert_bias.unsqueeze(1)
        return y, loss


class Conv1D_MEO(nn.Module):

    def __init__(self, input_size, output_size, config, noisy_gating=True, reduce_factor=16, bias=True):
        super(Conv1D_MEO, self).__init__()
        
        self.noisy_gating, self.moe_level = noisy_gating, config.moe_level
        self.n_experts, self.k = config.n_experts, config.k
        self.mean = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        input_dim = config.description_size // reduce_factor if self.moe_level == 'task' else input_size
        self.w_gate = nn.Parameter(torch.zeros(input_dim, config.n_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_dim, config.n_experts), requires_grad=True)
        # instantiate experts
        self.weight = nn.Parameter(torch.Tensor(
            config.n_experts, output_size, input_size), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(
            config.n_experts, output_size), requires_grad=True) if bias else None
        if self.moe_level == 'token':
            self.tokenatt = TokenAtt(input_size)
        elif self.moe_level == 'task':
            self.task_proj = nn.Linear(config.description_size, config.description_size // reduce_factor)  # todo randomly init
        assert (self.k <= config.n_experts)

    def forward(self, x, task_embeddings=None, loss_coef=1e-2):
        loss = None
        batch_size = x.shape[0]
        if self.moe_level == 'task' and task_embeddings is not None:
            task_embeddings = self.task_proj(task_embeddings)
            task_embeddings = task_embeddings.mean(1)
            gates, load = noisy_top_k_gating(self, task_embeddings, self.training)
        else:
            gates, load = noisy_top_k_gating(self, x.mean(-2), self.training)
        # calculate importance loss
        importance = gates.view(-1, self.n_experts).sum(0)
        loss = cv_squared(importance) + cv_squared(load)
        loss *= loss_coef
        
        if self.k * 2 < self.n_experts:
            index = torch.nonzero(gates)[:, -1:].flatten()
            gates = gates[gates != 0]                                                  
            expert_weights = torch.sum((gates.view(-1, 1, 1) * torch.index_select(self.weight, 0, index)).view(batch_size, self.k, self.weight.size()[-2], self.weight.size()[-1]), dim=1)
        else:
            expert_weights = torch.sum(torch.mul(self.weight, gates.view(batch_size, -1, 1, 1)), dim=1)
            
        if self.moe_level == 'token':
            x = x + self.tokenatt(x)
        y = torch.einsum('bij, bkj->bik', x, expert_weights) 
        if self.bias is not None:
            expert_bias = torch.sum(torch.mul(self.bias, gates.view(batch_size, -1, 1)), dim=1)
            y = y + expert_bias.unsqueeze(1)
        return y, loss


class Conv1D_MoE(nn.Module):

    def __init__(self, input_size, hidden_size, config, noisy_gating=True, reduce_factor=16, bias=True):
        super(Conv1D_MoE, self).__init__()
        self.k = config.k
        self.noisy_gating, self.moe_level = noisy_gating, config.moe_level
        self.n_experts, self.k = config.n_experts, config.k
        self.input_size, self.output_size = input_size, hidden_size
        # instantiate experts
        input_dim = config.description_size // reduce_factor if self.moe_level == 'task' else input_size
        self.w_gate = nn.Parameter(torch.zeros(input_dim, config.n_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_dim, config.n_experts), requires_grad=True)
        self.experts = nn.ModuleList([SingleExpert(input_size, hidden_size, bias=bias) for i in range(config.n_experts)])
        self.mean = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        if self.moe_level == 'task':
            self.task_proj = nn.Linear(config.description_size,
                                       config.description_size // reduce_factor)  # todo randomly init
        assert (self.k <= config.n_experts)

    def forward(self, x, task_embeddings=None, loss_coef=1e-2):
        original_shape = list(x.shape[:-1])
        if self.moe_level == 'task' and task_embeddings is not None:
            task_embeddings = task_embeddings.mean(1)
            task_embeddings = self.task_proj(task_embeddings)
            gates, load = noisy_top_k_gating(self, task_embeddings, self.training)
        elif self.moe_level == 'sequence':
            gates, load = noisy_top_k_gating(self, x.mean(-2), self.training)
        else:
            x = x.reshape(-1, self.input_size)
            gates, load = noisy_top_k_gating(self, x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        loss = cv_squared(importance) + cv_squared(load)
        loss *= loss_coef
        dispatcher = SparseDispatcher(self.n_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.n_experts)]
        y = dispatcher.combine(expert_outputs)
        y = y.reshape(original_shape + [self.output_size])
        return y, loss


def noisy_top_k_gating(layer, x, train, noise_epsilon=1e-2):
    clean_logits = x @ layer.w_gate
    if layer.noisy_gating and train:
        raw_noise_stddev = x @ layer.w_noise
        noise_stddev = ((softplus(raw_noise_stddev) + noise_epsilon))
        noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        logits = noisy_logits
    else:
        logits = clean_logits
    # calculate topk + 1 that will be needed for the noisy gates
    top_logits, top_indices = logits.topk(min(layer.k + 1, layer.n_experts), dim=-1)
    top_k_logits = top_logits[:, : layer.k]
    top_k_indices = top_indices[:, : layer.k]
    top_k_gates = softmax(top_k_logits)
    zeros = torch.zeros_like(logits, requires_grad=True, dtype=top_k_gates.dtype).to(x.device)
    gates = zeros.scatter(1, top_k_indices, top_k_gates).to(x.device)
    if layer.noisy_gating and layer.k < layer.n_experts and train:
        load = (_prob_in_top_k(layer, clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
    else:
        load = _gates_to_load(gates)
    return gates, load

def _prob_in_top_k(layer, clean_values, noisy_values, noise_stddev, noisy_top_values):
    batch = clean_values.size(0)
    m = noisy_top_values.size(1)
    top_values_flat = noisy_top_values.flatten()
    normal = Normal(layer.mean, layer.std)
    threshold_positions_if_in = torch.arange(batch).to(clean_values.device) * m + layer.k
    threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
    is_in = torch.gt(noisy_values, threshold_if_in)
    threshold_positions_if_out = threshold_positions_if_in - 1
    threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
    # is each value currently in the top k.
    prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
    prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
    prob = torch.where(is_in, prob_if_in, prob_if_out)
    return prob

def cv_squared(x):
    eps = 1e-10
    if x.shape[0] == 1:
        return torch.Tensor([0]).to(x.device)
    return x.float().var() / (x.float().mean() ** 2 + eps)

def _gates_to_load(gates):
    return (gates > 0).sum(0)


class TokenAtt(nn.Module):
    def __init__(self, dim, activation=nn.ReLU(), reduce_factor=64):
        super().__init__()
        self.down_proj = nn.Linear(dim, dim // reduce_factor)
        self.activation = activation
        self.up_proj = nn.Linear(dim // reduce_factor, dim)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        down = self.activation(self.down_proj(x))
        up = self.up_proj(down)
        return up