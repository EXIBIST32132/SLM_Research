from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0, "Embedding dim must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        attn_out = self.proj(attn_out)
        attn_out = self.resid_dropout(attn_out)
        return attn_out


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        dropout: float,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)

        self.blocks = nn.ModuleList(
            [Block(n_embd=n_embd, n_head=n_head, dropout=dropout) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        idx: (B, T)
        targets: (B, T) or None
        """
        B, T = idx.shape
        assert T <= self.block_size, "Sequence length exceeds block size"

        token_emb = self.token_embedding(idx)
        positions = torch.arange(T, device=idx.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        x = token_emb + pos_emb
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation from a given context.

        idx: (B, T_start)
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            logits = logits / temperature

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                if k > 0:
                    values, _ = torch.topk(logits, k)
                    cutoff = values[:, [-1]]
                    logits = logits.masked_fill(logits < cutoff, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_token], dim=1)

        return idx
