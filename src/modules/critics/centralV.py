# code adapted from https://github.com/AnujMahajanOxf/MAVEN

import torch as th
import torch.nn as nn
import torch.nn.functional as F

class CentralVCritic(nn.Module):
    def __init__(self, scheme, args):
        super(CentralVCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.hidden_dim = args.hidden_dim

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Attention projections
        self.x_proj = nn.Linear(input_shape, args.hidden_dim)
        self.key = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.query = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.value = nn.Linear(args.hidden_dim, args.hidden_dim)

        # Feed-forward MLP after attention
        self.mlp = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim)
        )

        # LayerNorms
        self.ln_attn = nn.LayerNorm(args.hidden_dim)
        self.ln_ff = nn.LayerNorm(args.hidden_dim)

        # Final scalar output per agent-timestep
        self.fc_out = nn.Linear(args.hidden_dim, 1)

    def forward(self, batch, t=None, attn_mask=None):
        # Build inputs: [B, T, N, D]
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        B, T, N, D = inputs.shape

        # Flatten timestep-major: t1_a1, t1_a2,... tT_aN
        x = inputs.contiguous().view(B, T*N, D)
        x = self.x_proj(x)  # [B, T*N, hidden_dim]


        # --- Keep your kron-based mask ---
        mask_block = th.kron(th.ones(T, T, device=batch.device), attn_mask[0])  # [T*N, T*N]
        mask = mask_block.unsqueeze(0).expand(B, -1, -1)                        # [B, T*N, T*N]

        # Convert 0/1 mask to additive form for softmax
        mask = (1.0 - mask).to(x.dtype) * (-1e9)  # 1=block -> -1e9, 0=allow -> 0

        # Linear projections
        Q = self.query(x)  # [B, T*N, hidden_dim]
        K = self.key(x)
        V = self.value(x)

        # Compute attention
        scores = th.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        scores = scores + mask                       # additive masking
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = th.matmul(attn_weights, V)        # [B, T*N, hidden_dim]

        # Residual + LayerNorm
        x = self.ln_attn(x + attn_out)

        # Feed-forward MLP + residual + LayerNorm
        ff_out = self.mlp(x)
        x = self.ln_ff(x + ff_out)

        # Map to scalar per agent-timestep
        v = self.fc_out(x)         # [B, T*N, 1]
        v = v.view(B, T, N, 1)     # [B, T, N, 1]

        return v


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
