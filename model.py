import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

# -- Configuration
@dataclass
class ModelConfig:
    vocab_size: int
    n_layer: int = 4        # number of layers
    n_head: int = 4         # number of heads
    n_embed: int = 256      # hidden size
    block_size: int = 128   # seq length
    dropout: float = 0.1    # unsure

# --- Attention 
class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # TODO: define q_proj, k_proj, v_proj, out_proj
        assert config.n_embed % config.n_head == 0, "n_embed must be divisible by n_head"
        self.config = config
        self.n_head = config.n_head
        self.head_dim = config.n_embed // config.n_head

        self.q_proj = nn.Linear(config.n_embed, config.n_embed, bias = False)
        self.k_proj = nn.Linear(config.n_embed, config.n_embed, bias = False)
        self.v_proj = nn.Linear(config.n_embed, config.n_embed, bias = False)
        self.out_proj = nn.Linear(config.n_embed, config.n_embed, bias = False)

        # TODO: causal mask (torch.tril)
        self.causal_mask: torch.Tensor
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))
        # TODO: dropout layers
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x):
        # TODO: implement scaled dot-product attention with causal mask
        B, T, C = x.size() # batch, time, channels

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        att = att.masked_fill(~self.causal_mask[:, :, :T, :T], float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v 
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.out_proj(y))
        return y
    #    raise NotImplementedError
    
# -- Feed Forward 
class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # TODO: two Linear layers with ReLU/GELU, and dropout
        self.fc = nn.Sequential(
            nn.Linear(config.n_embed, 4* config.n_embed),
            nn.ReLU(inplace=True),
            nn.Linear(4 * config.n_embed, config.n_embed),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.fc(x)
        #raise NotImplementedError
        
# -- Transformer
class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # TODO: LayerNorm, Attention, LayerNorm, MLP
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
        
    def forward(self, x):
        # TODO: Pre-LN → Attention + residual, then Pre-LN → MLP + residual
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
        #raise NotImplementedError 

### --- FULL MODEL ---
class TinyTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # TODO: token embedding, positional embedding
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.n_embed)
        self.pos_embed = nn.Embedding(config.block_size, config.n_embed)
        self.head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        # TODO: dropout
        self.drop = nn.Dropout(config.dropout)
        # TODO: stack of Blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # TODO: final LayerNorm + Linear to vocab size
        self.ln_f = nn.LayerNorm(config.n_embed)

        self.apply(self.__init__weights)


    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        
    def forward(self, idx, targets: Optional[torch.Tensor] = None):
        # TODO: embed tokens + positions
        B, T = idx.shape
        assert T <= self.config.block_size, "Sequence length exceeds block_size"

        # TODO: final normalization + projection
        pos = torch.arange(0, T, device=idx.device, dtype=torch.long).unsqueeze(0)
        tok = self.token_embed(idx)
        pos = self.pos_embed(pos)
        x = self.drop(tok + pos)

        # TODO: pass through blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        # TODO: compute loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
        #raise NotImplementedError
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None):
        self.eval()
        for _ in range(max_new_tokens):
            # 1) forward the model on the last block_size tokens
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)          # [B, T, V]
            logits = logits[:, -1, :]           # take logits at last time step -> [B, V]
            logits = logits / max(1e-6, temperature)

            # 2) optional top-k filter (only if positive)
            if top_k is not None and top_k > 0:
                top_k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')

            # 3) sample and append
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            idx = torch.cat((idx, next_token), dim=1)
        return idx
        #raise NotImplementedError