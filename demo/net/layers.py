
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len, device='cuda'):
        super().__init__()
        self.d_model = d_model
        self.pe = torch.zeros(max_len, d_model).to(device)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                self.pe.data[pos, i] = math.sin(pos / (10000.0 ** ((2.0 * i) / d_model)))
                self.pe.data[pos, i + 1] = math.cos(pos / (10000.0 ** ((2.0 * i) / d_model)))
        self.pe.data = self.pe.data.unsqueeze(0)

    def forward(self, x):
        seq_len = x.shape[1]
        return x + self.pe.data[0, :seq_len]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.d_k = int(d_model / num_heads)
        self.num_heads = num_heads

        self.Wq = nn.Linear(d_model, self.d_k * num_heads)
        self.Wk = nn.Linear(d_model, self.d_k * num_heads)
        self.Wv = nn.Linear(d_model, self.d_k * num_heads)

        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size, q_seq_len, _ = q.shape
        k_seq_len = k.size(1)
        v_seq_len = v.size(1)

        k_proj = self.Wk(k).view(batch_size, k_seq_len, self.num_heads, self.d_k)
        q_proj = self.Wq(q).view(batch_size, q_seq_len, self.num_heads, self.d_k)
        v_proj = self.Wv(v).view(batch_size, v_seq_len, self.num_heads, self.d_k)

        k_proj = k_proj.transpose(2, 1)
        q_proj = q_proj.transpose(2, 1)
        v_proj = v_proj.transpose(2, 1)

        sim_scores = torch.matmul(q_proj, k_proj.transpose(3, 2))
        sim_scores = sim_scores / self.d_k ** 0.5

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            sim_scores = sim_scores.masked_fill(mask == 0, value=-1e12)
        sim_scores = F.softmax(input=sim_scores, dim=-1)

        attention_applied = torch.matmul(sim_scores, v_proj)
        attention_applied_concatenated = attention_applied.permute(0, 2, 1, 3).contiguous()\
            .view(batch_size, q_seq_len, self.d_model)

        out = self.out_linear(attention_applied_concatenated)
        return out


class Encoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mha_1 = MultiHeadAttention(d_model, 1)
        self.norm_1 = nn.LayerNorm(d_model)
        self.l1 = nn.Linear(d_model, d_model)
        self.mha_2 = MultiHeadAttention(d_model, 1)
        self.norm_2 = nn.LayerNorm(d_model)
        self.l2 = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        x = self.norm_1(x)
        x = x + self.mha_1(q=x, k=x, v=x, mask=mask)
        x = x + F.relu(self.l1(x))
        x = self.norm_2(x)
        x = x + self.mha_2(q=x, k=x, v=x, mask=mask)
        x = x + F.relu(self.l2(x))
        return x


class Decoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mha_1 = MultiHeadAttention(d_model, 1)
        self.norm_1 = nn.LayerNorm(d_model)
        self.l1 = nn.Linear(d_model, d_model)
        self.mha_2 = MultiHeadAttention(d_model, 1)
        self.norm_2 = nn.LayerNorm(d_model)
        self.l2 = nn.Linear(d_model, d_model)

    def forward(self, y, no_peak_mask, cross_x, pad_mask):
        y = self.norm_1(y)
        y = y + self.mha_1(q=y, k=y, v=y, mask=no_peak_mask)
        y = y + F.relu(self.l1(y))
        y = self.norm_2(y)
        y = y + self.mha_2(q=y, k=cross_x, v=cross_x, mask=pad_mask)
        y = y + F.relu(self.l2(y))
        return y