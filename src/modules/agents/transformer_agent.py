import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import gc
import math

def get_pos_encoding(seq_len, d_model):
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.arange(0, d_model, 2, dtype=torch.float32) / d_model
    div_term = 1.0 / (10000.0 ** div_term)  # (d_model // 2,)

    # Repeat to interleave for even and odd dimensions
    div_term = div_term.repeat_interleave(2).unsqueeze(0)  # (1, d_model)

    encoding = position @ div_term  # (seq_len, d_model)

    encoding[:, 0::2] = torch.sin(encoding[:, 0::2])
    encoding[:, 1::2] = torch.cos(encoding[:, 1::2])

    return encoding  # (seq_len, d_model)



class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.sqrt_dk = math.sqrt(d_model)

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k_e = nn.Linear(d_model, d_model, bias=False)
        self.w_k_r = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.final = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU()
        )
        self.pos_enc = get_pos_encoding(self.max_seq_len, self.d_model)
        self.u_param = nn.Parameter(torch.randn(1, 1, n_heads, self.d_head))
        self.v_param = nn.Parameter(torch.randn(1, 1, n_heads, self.d_head))

    def rel_enc_shift(self, arr):
        # arr: (batch_size, num_heads, l, m)
        batch_size, num_heads, l, m = arr.size()
        zeros = torch.zeros(batch_size, num_heads, l, 1, device=arr.device, dtype=arr.dtype)
        arr = torch.cat([arr, zeros], dim=-1)
        arr = arr.view(batch_size, num_heads, -1)
        arr = arr[:, :, l-1: -1]
        arr = arr.view(batch_size, num_heads, l, m)
        return arr

    def forward(self, query, key, value, attn_mask):
        
         
        batch_size, full_len, _ = value.size()
          # (batch_size, seq_len, d_model)
        _, seq_len, _ = query.size()
        rel_enc = self.pos_enc[:full_len, :]
        rel_enc = rel_enc.to(query.device)
        rel_enc = torch.flip(rel_enc, dims=[0]) 

        q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_head)
        k = self.w_k_e(key).view(batch_size, full_len, self.n_heads, self.d_head)
        v = self.w_v(value).view(batch_size, full_len, self.n_heads, self.d_head)

        A_C = torch.einsum('bsnd,bfnd->bnsf', q + self.u_param, k)

        Q = self.w_k_r(rel_enc)  # (full_len, d_model)
        Q = Q.view(full_len, self.n_heads, self.d_head)
        B_D_hat = torch.einsum('bsnd,fnd->bnsf', q + self.v_param, Q)
        B_D = self.rel_enc_shift(B_D_hat)

        attention_score = (A_C + B_D) / self.sqrt_dk
        attention_score += attn_mask

        attention_weights = F.softmax(attention_score, dim=-1)
        max_weights = attention_weights.max(dim=-1).values.max(dim=-1).values
        attention_loss = max_weights.mean()

        attention_output = torch.einsum('bnsf,bfnd->bsnd', attention_weights, v)
        attention_output = attention_output.contiguous().view(batch_size, seq_len, self.d_model)

        output = self.final(attention_output)
        return output, attention_weights, attention_loss



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
    


class TransformerAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(TransformerAgent, self).__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)  # Projects input to model dim
        self.input_norm = nn.LayerNorm(args.hidden_dim)  # Normalize inputs
        self.n_layers = args.n_layers
        # Create decoder blocks without cross-attention (more like GPT architecture)
        self.layers = nn.ModuleList([DecoderOnlyBlock(
            d_model=args.hidden_dim,
            nhead=args.n_heads,
            norm_first=True,
            max_seq_len=self.max_seq_len
        ) for _ in range(args.n_layers)])
        self.mem_len = 150
        
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
        with torch.no_grad():
            new_memory = []
            for mem, h in zip(memory, hidden_states):
                h= h.detach()
                mem = mem.detach()
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
        
        for i, layer in enumerate(self.layers):
            mem = None if memory is None else memory[i]
            x = layer(x, memory=mem, attn_mask=attn_mask)
            hidden_states.append(x.detach().clone())

        x = self.output_norm(x)
        q = self.fc2(x)
        
        return q, hidden_states

# Helper class for GPT-style implementation
class DecoderOnlyBlock(nn.Module):
    def __init__(self, d_model, nhead, norm_first, max_seq_len):
        super(DecoderOnlyBlock, self).__init__()
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.gate1 = GRUGate(d_model, 0.0)
        self.gate2 = GRUGate(d_model, 0.0)
        self.norm_kv = nn.LayerNorm(d_model)

        self.self_attn = RelativeMultiHeadAttention(
            d_model=d_model,
            n_heads=nhead,
            max_seq_len=max_seq_len,
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        
        
    def forward(self, x, memory, attn_mask=None):
        
        # Concatenate memory (if any) and compute attention
        if memory is not None:
            x_cat = torch.cat([memory, x], dim=1)
        else:
            x_cat = x
        attn_op  = self.self_attn(
            self.norm1(x), self.norm_kv(x_cat), self.norm_kv(x_cat), 
            attn_mask=attn_mask
        )[0]
        h = self.gate1(x, attn_op)
        h_ = self.norm2(h)
        forward = self.ffn(h_)
        out = self.gate2(h, forward)
        return out