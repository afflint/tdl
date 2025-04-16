
import torch
import torch.nn as nn
import torch.nn.functional as F



class SimpleSelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5

    def forward(self, x, mask=None):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        return context.mean(dim=1), attn_weights

class SimpleIngredientTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.attn = SimpleSelfAttention(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.pad_idx = pad_idx

    def forward(self, x):
        emb = self.embedding(x)
        mask = (x != self.pad_idx).unsqueeze(1).unsqueeze(2)  # shape: (batch, 1, 1, seq_len)
        out, attn = self.attn(emb, mask=mask)
        logits = self.output_layer(out)
        return logits[:, -1, :], attn # return just the last token of the sequence
