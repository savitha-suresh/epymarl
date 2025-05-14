import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GRUGate(nn.Module):
    """
    Overview:
        GRU Gating Unit used in GTrXL.
        Inspired by https://github.com/dhruvramani/Transformers-RL/blob/master/layers.py
    """

    def __init__(self, input_dim: int, bg: float = 0.0):
        """
        Arguments:
            input_dim {int} -- Input dimension
            bg {float} -- Initial gate bias value. By setting bg > 0 we can explicitly initialize the gating mechanism to
            be close to the identity map. This can greatly improve the learning speed and stability since it
            initializes the agent close to a Markovian policy (ignore attention at the beginning). (default: {0.0})
        """
        super(GRUGate, self).__init__()
        self.Wr = nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = nn.Linear(input_dim, input_dim, bias=False)
        self.Ug = nn.Linear(input_dim, input_dim, bias=False)
        self.bg = nn.Parameter(torch.full([input_dim], bg))  # bias
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        nn.init.xavier_uniform_(self.Wr.weight)
        nn.init.xavier_uniform_(self.Ur.weight)
        nn.init.xavier_uniform_(self.Wz.weight)
        nn.init.xavier_uniform_(self.Uz.weight)
        nn.init.xavier_uniform_(self.Wg.weight)
        nn.init.xavier_uniform_(self.Ug.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """        
        Arguments:
            x {torch.tensor} -- First input
            y {torch.tensor} -- Second input
        Returns:
            {torch.tensor} -- Output
        """
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
        return torch.mul(1 - z, x) + torch.mul(z, h)
    

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return x + self.pos_embedding(positions)

class TransformerAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(TransformerAgent, self).__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)  # Projects input to model dim
        self.input_norm = nn.LayerNorm(args.hidden_dim)  # Normalize inputs
        self.dropout = nn.Dropout(0.1)
        self.pos_enc = LearnedPositionalEncoding(args.hidden_dim, max_len=self.max_seq_len)
        self.n_layers = args.n_layers
        # Create decoder blocks without cross-attention (more like GPT architecture)
        self.layers = nn.ModuleList([DecoderOnlyBlock(
            d_model=args.hidden_dim,
            nhead=args.n_heads,
            dim_feedforward=args.hidden_dim * 4,
            dropout=0.1,
            norm_first=True
        ) for _ in range(args.n_layers)])
        self.mem_len = args.max_seq_len
        
        self.output_norm = nn.LayerNorm(args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)
        self.memories = [None for _ in range(args.n_layers)]

    def init_hidden(self):
        self.memories = [None for _ in range(self.n_layers)]

    def init_memory(self, batch_size):
        return [torch.zeros(batch_size * self.args.n_agents, 0, self.args.hidden_dim, 
                            device=next(self.parameters()).device) for _ in self.layers]

    def update_memory(self, memory, hidden_states):
        # Save only last `max_mem_len` timesteps
        new_memory = []
        for mem, h in zip(memory, hidden_states):
            combined = torch.cat([mem, h], dim=1)
            new_memory.append(combined[:, -self.mem_len:].detach())
        return new_memory
    
    def forward(self, inputs, memory=None, attn_mask=None):
        # inputs: (batch_size, seq_len, input_dim)
        hidden_states = []     
        
        
        x = inputs
        # Process inputs
        x = F.relu(self.fc1(inputs))
        x = self.input_norm(x)
        x = self.dropout(x)
        x = self.pos_enc(x)
        
        for i, layer in enumerate(self.layers):
            mem = None if memory is None else memory[i]
            x = layer(x, memory=mem, attn_mask=attn_mask)
            hidden_states.append(x)

        x = self.output_norm(x)
        q = self.fc2(x)
        
        return q, hidden_states

# Helper class for GPT-style implementation
class DecoderOnlyBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, norm_first):
        super(DecoderOnlyBlock, self).__init__()
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.gate1 = GRUGate(d_model, 0.0)
        self.gate2 = GRUGate(d_model, 0.0)
        self.norm_kv = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, memory, attn_mask=None):
        
        # Concatenate memory (if any) and compute attention
        if memory is not None:
            x_cat = torch.cat([memory, x], dim=1)
        else:
            x_cat = x
        attn_op  = self.dropout(self.self_attn(
            self.norm1(x), self.norm_kv(x_cat), self.norm_kv(x_cat), 
            attn_mask=attn_mask, need_weights=False
        )[0])
        h = self.gate1(x, attn_op)
        h_ = self.norm2(h)
        forward = self.ffn(h_)
        out = self.gate2(h, forward)
        return out