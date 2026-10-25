'''
This file implements the Feed forward network
Adapted from OpenNMT-Py
'''

from torch import nn

class FFN(nn.Module):
    def __init__(self, input_dim, scale, output_dim=None, p=0.1, expansion=False):
        super(FFN, self).__init__()
        output_dim = input_dim if output_dim is None else output_dim

        proj_features = input_dim * scale if expansion else input_dim // scale
        self.w_1 = nn.Linear(input_dim, proj_features)
        self.w_2 = nn.Linear(proj_features, output_dim)

        self.layer_norm = nn.LayerNorm(input_dim, eps=1e-6)
        self.dropout_1 = nn.Dropout(p)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(p)
        self.residual = True if input_dim == output_dim else False

    def forward(self, x):
        """Layer definition.
        Args:
            x: ``(batch_size, input_len, model_dim)``
        Returns:
            (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """

        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x if self.residual else output


import torch
from torch import nn
import math
import numpy as np
from torch import Tensor
from typing import Optional

'''
This file implements the Multi-head attention
Adapted from OpenNMT-Py
'''


class MultiHeadAttn(torch.nn.Module):
    def __init__(self, input_dim, out_dim, num_heads=8, dropout=0.1, *args, **kwargs):
        super(MultiHeadAttn, self).__init__()
        assert input_dim % num_heads == 0

        self.num_heads = num_heads
        self.dim_per_head = input_dim // num_heads

        self.linear_keys = nn.Linear(input_dim, num_heads * self.dim_per_head)
        self.linear_values = nn.Linear(input_dim, num_heads * self.dim_per_head)
        self.linear_query = nn.Linear(input_dim, num_heads * self.dim_per_head)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.final_linear = nn.Linear(input_dim, out_dim)

        self.scaling_factor = math.sqrt(self.dim_per_head)

    def forward(self, key, value, query, need_attn=False):
        '''
        :param key: A tensor of shape  [B x N_b x N_w x d] or [B x N_b x d]
        :param value: A tensor of shape  [B x N_b x N_w x d] or [B x N_b x d]
        :param query: A tensor of shape  [B x N_b x N_w x d] or [B x N_b x d]
        :param need_attn: Need attention weights or not
        :return: Tuple containing output and mean attention scores across all heads (optional)
            Output size is [B x N_b x N_w x d'] or [B x N_b x d']
            Attention score size is [B x N_b*N_w x N_b*N_w] or [B x N_b x N_b]
        '''
        dim_size = key.size()
        reshape=False
        if key.dim() == 4:
            # [B x N_b x N_w x d] --> [B x N_b*N_w x d]
            key = key.view(dim_size[0], -1, dim_size[3])
            value = key.view(dim_size[0], -1, dim_size[3])
            query = key.view(dim_size[0], -1, dim_size[3])
            reshape = True

        batch_size = key.size(0)

        dim_per_head = self.dim_per_head
        head_count = self.num_heads

        key = self.linear_keys(key)
        value = self.linear_values(value)
        query = self.linear_query(query)

        query = query / self.scaling_factor

        # [B x N_b*N_w x d] --> [B x N_b*N_w x h x d_h] --> [B x h x N_b*N_w x d_h]
        key = (
            key.contiguous()
            .view(batch_size, -1, head_count, dim_per_head)
            .transpose(1, 2)
        )

        value = (
            value.contiguous()
            .view(batch_size, -1, head_count, dim_per_head)
            .transpose(1, 2)
        )

        query = (
            query.contiguous()
            .view(batch_size, -1, head_count, dim_per_head)
            .transpose(1, 2)
        )

        # compute attention scores
        # [B x h x N_b*N_w x d_h] x [B x h x d_h x N_b*N_w] --> [B x h x N_b*N_w x N_b*N_w]
        scores = torch.matmul(query, key.transpose(2, 3)).float()

        attn = self.softmax(scores).to(query.dtype)
        drop_attn = self.dropout(attn)

        # [B x h x N_b*N_w x N_b*N_w] x [B x h x N_b*N_w x d_h] --> [B x h x N_b*N_w x d_h]
        context = torch.matmul(drop_attn, value)

        # [B x h x N_b*N_w x d_h] --> [B x N_b*N_w x h x d_h] -->  [B x N_b*N_w x h*d_h]
        context = (
            context.transpose(1, 2)
            .contiguous().view(batch_size, -1, head_count * dim_per_head)
        )

        output = self.final_linear(context)

        attn_scores: Tensor[Optional] = None
        if need_attn:
            attn_scores = torch.mean(scores, dim=1)

        if reshape:
            # [B x N_b*N_w x d] --> [B x N_b x N_w x d]
            output = output.contiguous().view(dim_size[0], dim_size[1], dim_size[2], -1).contiguous()

        return output, attn_scores

import torch
from torch import nn

'''
Adapted from OpenNMT-Py
'''


class SelfAttention(nn.Module):
    '''
    This class implements the transformer block with multi-head attention and Feed forward network
    '''

    def __init__(self, in_dim, num_heads=8, p=0.1, *args, **kwargs):
        super(SelfAttention, self).__init__()
        self.self_attn = MultiHeadAttn(input_dim=in_dim, out_dim=in_dim, num_heads=num_heads)

        self.ffn = FFN(in_dim, scale=4, p=p, expansion=True)

        self.layer_norm_1 = nn.LayerNorm(in_dim, eps=1e-6)
        self.drop = nn.Dropout(p=p)

    def forward(self, x, need_attn=False):
        '''
        :param x: Input (bags or words)
        :param need_attn: Need attention weights or not
        :return: returns the self attention output and attention weights (optional)
        '''
        x_norm = self.layer_norm_1(x)

        context, attn = self.self_attn(x_norm, x_norm, x_norm, need_attn=need_attn)

        out = self.drop(context) + x
        return self.ffn(out), attn


class MultiWaySelfAttention(nn.Module):
    '''
    This class implements the transformer block with multi-head attention and Feed forward network
    '''

    def __init__(self, in_dim, num_heads=8, p=0.1, num_way=4, *args, **kwargs):
        super(MultiWaySelfAttention, self).__init__()
        self.self_attn = MultiHeadAttn(input_dim=in_dim, out_dim=in_dim, num_heads=num_heads)

        self.way_cls_fn = nn.Linear(in_dim, num_way)
        self.ffn_lst = nn.ModuleList([FFN(in_dim, scale=4, p=p, expansion=True) for _ in range(num_way)])

        self.layer_norm_1 = nn.LayerNorm(in_dim, eps=1e-6)
        self.drop = nn.Dropout(p=p)

        self.num_way = num_way

    def forward(self, x, w, need_attn=False):
        '''
        :param x: Input (bags or words)
        :param need_attn: Need attention weights or not
        :return: returns the self attention output and attention weights (optional)
        '''
        # x: (B, N, C)
        # w: (B, N)
        x_norm = self.layer_norm_1(x)

        context, attn = self.self_attn(x_norm, x_norm, x_norm, need_attn=need_attn)

        out = self.drop(context) + x

        ffn_out = None
        for ci in range(self.num_way):
            ci_ffn_out = self.ffn_lst[ci](out)
            ci_mask = (w == ci).float().unsqueeze(dim=-1)
            if ffn_out is None:
                ffn_out = ci_ffn_out
            else:
                ffn_out = ffn_out * (1.0 - ci_mask) + ci_ffn_out * ci_mask

        return ffn_out, self.way_cls_fn(out), attn


class ContextualAttention(torch.nn.Module):
    '''
        This class implements the contextual attention.
        For example, we used this class to compute bag-to-bag attention where
        one set of bag is directly from CNN, while the other set of bag is obtained after self-attention
    '''

    def __init__(self, in_dim, num_heads=8, p=0.1, *args, **kwargs):
        super(ContextualAttention, self).__init__()
        self.self_attn = MultiHeadAttn(input_dim=in_dim, out_dim=in_dim, num_heads=num_heads)

        self.context_norm = nn.LayerNorm(in_dim)
        self.context_attn = MultiHeadAttn(input_dim=in_dim, out_dim=in_dim, num_heads=num_heads)
        self.ffn = FFN(in_dim, scale=4, p=p, expansion=True)

        self.input_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.query_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.drop = nn.Dropout(p=p)

    def forward(self, input, context, need_attn=False):
        '''
        :param input: Tensor of shape (B x N_b x N_w x CNN_DIM) or (B x N_b x CNN_DIM)
        :param context: Tensor of shape (B x N_b x N_w x hist_dim) or (B x N_b x hist_dim)
        :return:
        '''

        # Self attention on Input features
        input_norm = self.input_norm(input)
        query, _ = self.self_attn(input_norm, input_norm, input_norm, need_attn=need_attn)
        query = self.drop(query) + input
        query_norm = self.query_norm(query)

        # Contextual attention
        context_norm = self.context_norm(context)
        mid, contextual_attn = self.context_attn(context_norm, context_norm, query_norm, need_attn=need_attn)
        output = self.ffn(self.drop(mid) + input)

        return output, contextual_attn


class MultiWayContextualAttention(torch.nn.Module):
    '''
        This class implements the contextual attention.
        For example, we used this class to compute bag-to-bag attention where
        one set of bag is directly from CNN, while the other set of bag is obtained after self-attention
    '''

    def __init__(self, in_dim, num_heads=8, p=0.1, num_way=6, *args, **kwargs):
        super(MultiWayContextualAttention, self).__init__()
        self.self_attn = MultiHeadAttn(input_dim=in_dim, out_dim=in_dim, num_heads=num_heads)

        self.context_norm = nn.LayerNorm(in_dim)
        self.context_attn = MultiHeadAttn(input_dim=in_dim, out_dim=in_dim, num_heads=num_heads)

        self.way_cls_fn = nn.Linear(in_dim, num_way)
        self.ffn_lst = nn.ModuleList([FFN(in_dim, scale=4, p=p, expansion=True) for _ in range(num_way)])

        self.input_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.query_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.drop = nn.Dropout(p=p)

        self.num_way = num_way

    def forward(self, input, context, w, need_attn=False):
        '''
        :param input: Tensor of shape (B x N_b x N_w x CNN_DIM) or (B x N_b x CNN_DIM)
        :param context: Tensor of shape (B x N_b x N_w x hist_dim) or (B x N_b x hist_dim)
        :return:
        '''

        # Self attention on Input features
        input_norm = self.input_norm(input)
        query, _ = self.self_attn(input_norm, input_norm, input_norm, need_attn=need_attn)
        query = self.drop(query) + input
        query_norm = self.query_norm(query)

        # Contextual attention
        context_norm = self.context_norm(context)
        mid, contextual_attn = self.context_attn(context_norm, context_norm, query_norm, need_attn=need_attn)

        # output = self.ffn(self.drop(mid) + input)
        tmp = self.drop(mid) + input

        output = None
        for ci in range(self.num_way):
            ci_output = self.ffn_lst[ci](tmp)  # (B, N, C)
            ci_mask = (w == ci).float().unsqueeze(dim=-1)  # (B, N, 1)
            if output is None:
                output = ci_output
            else:
                output = output * (1.0 - ci_mask) + ci_output * ci_mask

        return output, self.way_cls_fn(tmp), contextual_attn
