import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        # Create decoder blocks without cross-attention (more like GPT architecture)
        self.layers = nn.ModuleList([DecoderOnlyBlock(
            d_model=args.hidden_dim,
            nhead=args.n_heads,
            dim_feedforward=args.hidden_dim * 4,
            dropout=0.1,
            norm_first=True
        ) for _ in range(args.n_layers)])
        
        self.output_norm = nn.LayerNorm(args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        return self.fc1.weight.new_zeros(1, self.args.hidden_dim)

    def forward(self, inputs, hidden_state=None):
        # inputs: (batch_size, seq_len, input_dim)
        seq_len = inputs.size(1)
        
        # Process inputs
        x = F.relu(self.fc1(inputs))
        x = self.input_norm(x)
        x = self.dropout(x)
        x = self.pos_enc(x)
        
        # Create causal attention mask
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=inputs.device), diagonal=1
        )
        
        # Process through decoder-only blocks
        for layer in self.layers:
            x = layer(x, causal_mask)
            
        x = self.output_norm(x)
        q = self.fc2(x)
        return q, None

# Helper class for GPT-style implementation
class DecoderOnlyBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, norm_first):
        super(DecoderOnlyBlock, self).__init__()
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
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
        
    def forward(self, x, mask=None):
        if self.norm_first:
            # Pre-norm architecture
            attn_output = x + self.dropout(self.self_attn(
                self.norm1(x), self.norm1(x), self.norm1(x), 
                attn_mask=mask, need_weights=False
            )[0])
            output = attn_output + self.dropout(self.ffn(self.norm2(attn_output)))
        else:
            # Post-norm architecture
            attn_output = self.norm1(x + self.dropout(self.self_attn(
                x, x, x, attn_mask=mask, need_weights=False
            )[0]))
            output = self.norm2(attn_output + self.dropout(self.ffn(attn_output)))
            
        return output