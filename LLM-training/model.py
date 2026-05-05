import torch
import torch.nn as nn
import tiktoken


class ModelConfig:
    vocab_size = tiktoken.get_encoding("gpt2")
    max_length = 1024
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.1

class MyModel(nn.Module):
    def __init__(self, vocab_size, max_length, n_embd):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(max_length, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.tok_emb(idx)