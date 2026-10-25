import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

class FFN(nn.Module):
    def __init__(self, input_dim, scale=2, p=0.1):
        super().__init__()
        hidden_dim = input_dim * scale
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x):
        return self.net(x)

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn   = FFN(input_dim=embed_dim, scale=2, p=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self,
                query: Tensor,      # (B, 1, C)
                key:   Tensor,      # (B, M, C)
                value: Tensor,      # (B, M, C)
                key_padding_mask: Optional[Tensor] = None  # (B, M)
    ):
        # 1) 调用官方 MultiheadAttention
        attn_out, attn_map = self.attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask
        )  # attn_map: (B, num_heads, M)

        # 2) 手动屏蔽 & 重新归一化
        if key_padding_mask is not None:
            # key_padding_mask: (B, M) -> (B, 1, M)
            mask = key_padding_mask.unsqueeze(1)
            # 屏蔽被 mask 的位置
            attn_map = attn_map.masked_fill(mask, 0.0)
            # 对每个 head, 每个 query 做归一化
            sums = attn_map.sum(dim=-1, keepdim=True)         # (B, num_heads, 1)
            attn_map = attn_map / (sums + 1e-6)               # 加 eps 防止除零

        # 3) 残差 + LayerNorm + FFN
        q = query + attn_out
        q = self.norm1(q)
        f = self.ffn(q)
        out = self.norm2(q + f)

        # 返回 (B, C) 和归一化后的 attn_map
        return out.squeeze(1), attn_map

if __name__ == "__main__":
    # 本地测试
    B, M, C = 2, 3, 8
    cross = CrossAttention(embed_dim=C, num_heads=1, dropout=0.0)

    query = torch.randn(B, 1, C)
    key   = torch.randn(B, M, C)
    value = key.clone()
    pad_mask = torch.tensor([[True, False, False],
                             [False, True,  False]], dtype=torch.bool)

    cross.train()
    out, attn_map = cross(query, key, value, key_padding_mask=pad_mask)
    print("Test done, attn_map:", attn_map)
    print("Sums per sample:", attn_map.sum(dim=-1))