# code adapted from https://github.com/AnujMahajanOxf/MAVEN

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .transformer_agent import TransformerAgent


class CentralVCritic(nn.Module):
    def __init__(self, scheme, args):
        super(CentralVCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"
        

        # Set up network layers
        self.model = TransformerAgent(input_shape, args)
        self.memory = self.model.init_memory(args.batch_size)

    def build_causal_mask(self, seq_len, mem_len, device):
        total_len = mem_len + seq_len
        # Allow attending to memory (mem_len), and to current and past in x
        mask = th.triu(th.ones(seq_len, total_len, device=device) * float('-inf'), diagonal=1)
        return mask 
    
    def generate_agent_labels(self, batch_size, faulty_indices=None):
        agent_labels = th.ones(batch_size, self.args.max_seq_len, self.args.n_agents, device=self.args.device)
        if faulty_indices is not None:
            for idx in faulty_indices:
                agent_labels[:, :, idx] = 0
        return agent_labels

    def forward(self, batch, t=None, faulty_indices=None):
        self.model.train()
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        q_values = []
        for t_step_curr in range(max_t):
            inputs_curr = inputs[:, t_step_curr:t_step_curr+1]  # (bs * n_agents, 1, input_shape)
            memory = self.memory
            mem_len_now = 0 if memory is None else memory[0].size(1)
            mask = self.build_causal_mask(
                seq_len=inputs_curr.size(1), 
                mem_len=mem_len_now, device=inputs.device) 
            mask = mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, total_len]
            mask = mask.expand(self.args.batch_size * self.n_agents, self.args.n_heads, -1, -1)
            agent_labels = self.generate_agent_labels(bs, faulty_indices)
            q, hidden_states, loss = self.model(
                inputs_curr,
                memory=memory,
                attn_mask=mask,
                actions=None,
                return_aux_losses=True,
                agent_labels=agent_labels,
                faulty_indices=faulty_indices
            )
            self.memory = self.model.update_memory(memory, hidden_states)
            q_values.append(q)
        final_q = th.stack(q_values, dim=1)
        return final_q, loss

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observations
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts].view(bs, max_t, -1).unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # last actions
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1))
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
                last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
                inputs.append(last_actions)

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat(inputs, dim=-1)
        inputs = inputs.permute(0, 2, 1, 3).reshape(bs * self.n_agents, max_t, -1)  # (bs * n_agents, max_t, input_shape)
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observations
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"] * self.n_agents
        # last actions
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        input_shape += self.n_agents
        return input_shape
