# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoderBlock(nn.Module):
    """An independent Causal Transformer Decoder block utilizing standard PyTorch layers."""
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = nn.MultiheadAttention(embed_dim=cfg.n_embd, num_heads=cfg.n_head, dropout=cfg.dropout, batch_first=True)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout)
        )

    def forward(self, x, attn_mask):
        # Attention layer with causal masking
        attn_out, _ = self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        # Feed-forward layer
        x = x + self.mlp(self.ln_2(x))
        return x

class AssemblyRouterLM(nn.Module):
    """A completely self-contained Language Model for parsing assembly text-to-JSON tokens."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.token_embeddings = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embeddings = nn.Embedding(cfg.max_seq_len, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        
        self.blocks = nn.ModuleList([TransformerDecoderBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        
        # Tie weights between embeddings and language model head to reduce memory footprint
        self.token_embeddings.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        
        # Generate absolute position markers
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        
        # Merge token identities with sequence positions
        x = self.token_embeddings(idx) + self.position_embeddings(pos)
        x = self.dropout(x)
        
        # Build standard upper-triangular causal attention mask
        mask = torch.triu(torch.full((t, t), float('-inf'), device=device), diagonal=1)
        
        # Pass representation state through stacked decoder blocks
        for block in self.blocks:
            x = block(x, attn_mask=mask)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Shift cross-entropy parameters to align autoregressive token sequence prediction
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            
        return logits, loss