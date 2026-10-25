import torch
import torch.nn as nn
from timm.layers import DropPath
from model.utils import *
from typing import NamedTuple, Optional
from torch import Tensor
import math
import os
import numpy as np
import torch.nn.functional as F
import random
from torchvision import models
from torchvision.models import MNASNet
from tqdm import tqdm
import os


class TransformerEncoderLayerDP(nn.Module):
    def __init__(self, d_model, nhead, dropout, drop_path_rate):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead,
                                               dropout=dropout,
                                               batch_first=True)
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)

        # 前馈
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self‐Attn + DropPath
        src2, _ = self.self_attn(src, src, src,
                                 attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)
        # 残差并加 Stochastic Depth
        src = src + self.drop_path1(self.dropout(src2))
        src = self.norm1(src)

        # FFN + DropPath
        ff = self.linear2(self.dropout(nn.functional.relu(self.linear1(src))))
        src = src + self.drop_path2(ff)
        src = self.norm2(src)
        return src


class BLModel(nn.Module):
    def __init__(self, opts, *args, **kwargs):
        super().__init__()
        self.opts = opts
        # 原始 CNN 结构
        if opts.yx_cnn_name == "mnasnet":
            s = 1.0
            backbone = MNASNet(alpha=s, num_classes=2)
            del backbone.classifier
            self.cnn = backbone
        else:
            backbone = eval(f"models.{opts.bl_cnn_name}")(pretrained=opts.bl_cnn_pretrained)
            self.cnn = nn.Sequential(*list(backbone.children())[:-1])

        self.cnn_reduce = nn.Conv2d(
            in_channels=512,
            out_channels=opts.bl_cnn_features,
            kernel_size=1, stride=1, padding=0
        )
        self.attn_layer = nn.Conv2d(
            opts.bl_cnn_features + 3, 1, kernel_size=1, padding=0, stride=1
        )
        self.project_words = nn.Linear(opts.bl_cnn_features, opts.bl_out_features)

        # Words->Bags & Bags->Image 自注意力保留
        self.attn_over_words = nn.MultiheadAttention(embed_dim=opts.bl_out_features,
                                                     num_heads=1, dropout=opts.bl_dropout, batch_first=True)
        self.attn_over_bags = nn.MultiheadAttention(embed_dim=opts.bl_out_features,
                                                    num_heads=1, dropout=opts.bl_dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(p=opts.bl_attn_dropout)

        self.bags_classifier = nn.Linear(opts.bl_out_features, 2)
        self.words_classifier = nn.Linear(opts.bl_out_features, 2)
        self.ffn_attn_w2b_lst = FFN(input_dim=opts.bl_out_features, scale=2, p=opts.bl_dropout)
        self.ffn_b2i = FFN(input_dim=opts.bl_out_features, scale=2, p=opts.bl_dropout)
        self.attn_fn = nn.Softmax(dim=-1)
        self.words_weight_fn = nn.Linear(opts.bl_out_features, 1, bias=False)
        self.bags_weight_fn = nn.Linear(opts.bl_out_features, 1, bias=False)

        # clinical 投影（每个字段单独映射）
        self.qfjc_fc = nn.Linear(2, opts.bl_out_features)
        self.grade_fc = nn.Linear(3, opts.bl_out_features)
        self.AJCC_fc = nn.Linear(6, opts.bl_out_features)
        self.Tis_fc = nn.Linear(2, opts.bl_out_features)
        self.T_fc = nn.Linear(6, opts.bl_out_features)
        self.M_fc = nn.Linear(2, opts.bl_out_features)
        self.N_fc = nn.Linear(4, opts.bl_out_features)

        # 把所有 clinical 子层放到一个 ModuleList，便于 forward 时迭代
        self._clin_fcs = nn.ModuleList([
            self.qfjc_fc,
            self.grade_fc,
            self.AJCC_fc,
            self.Tis_fc,
            self.T_fc,
            self.M_fc,
            self.N_fc
        ])
        n_clin_fields = len(self._clin_fcs)  # 8
        # 新增：将串联后高维临床特征投影回 opts.bl_out_features
        self.clinical_proj = nn.Linear(n_clin_fields * opts.bl_out_features, opts.bl_out_features)

        layers = []
        for _ in range(opts.bl_vit_layers):
            layers.append(
                TransformerEncoderLayerDP(
                    d_model=opts.bl_out_features,
                    nhead=opts.bl_attn_heads,
                    dropout=opts.bl_dropout,
                    drop_path_rate=opts.bl_drop_path_rate
                )
            )
        self.transformer_layers = nn.ModuleList(layers)

        # 缺失模态的 placeholder
        self.missing_embed = nn.Parameter(torch.randn(1, opts.bl_out_features))
        nn.init.xavier_uniform_(self.missing_embed)

        # 调试计数器
        self.debug_counter = 0

        self.classifier = nn.Linear(opts.bl_out_features, opts.n_classes)
        self.risk_fc = nn.Linear(opts.bl_out_features, 1)

    def energy_function(self, x, weight_fn, need_attn=False):
        x = weight_fn(x).squeeze(dim=-1)
        energy: Optional[Tensor] = None
        if need_attn:
            energy = x
        x = self.attn_fn(x)
        x = self.attn_dropout(x)
        x = x.unsqueeze(dim=-1)
        return x, energy

    def forward(self, batch, *args, **kwargs):
        B = batch['feat_words'].shape[0]

        # STEP1: Words CNN & 投影
        words = batch['feat_words']  # (B, N_b, N_w, C, H, W)
        B, N_b, N_w, C, H, W = words.shape

        # 处理 roi_mask_bl（可选）
        # 期望 roi_mask_bl 的形状为 (B, N_b, N_w, rC, H, W) 或 (B, N_b, N_w, 1, H, W) 或 (B, N_b, N_w, 3, H, W)
        roi_masks_bl = batch.get('roi_masks_bl', None)

        words_input = words.view(B * N_b * N_w, C, H, W)  # 默认输入
        if roi_masks_bl is not None:
            # 展平 roi
            roi_flat = roi_masks_bl.view(B * N_b * N_w, -1, H, W)  # (B*N_b*N_w, rC, H, W)
            # 如果 roi 通道数不等于 image 通道数，尝试广播：
            if roi_flat.shape[1] == C:
                roi_proc = roi_flat
            elif roi_flat.shape[1] == 1:
                roi_proc = roi_flat.repeat(1, C, 1, 1)
            else:
                # 若 roi 通道不是 1 且不等于 C，先做通道平均再广播（保守处理）
                roi_proc = roi_flat.mean(dim=1, keepdim=True).repeat(1, C, 1, 1)
            # 确保 dtype/device 一致
            if roi_proc.dtype != words_input.dtype:
                roi_proc = roi_proc.to(words_input.dtype)
            if roi_proc.device != words_input.device:
                roi_proc = roi_proc.to(words_input.device)
            # 应用掩码（逐像素相乘），只保留 ROI 区域
            words_input = words_input * roi_proc

        # 将 mask 引导后的 patch(s) 输入 CNN
        words_cnn_feats = self.cnn(words_input)  # (B*N_b*N_w, feat_c, h', w')
        words_cnn_feats = self.cnn_reduce(words_cnn_feats)  # (B*N_b*N_w, bl_cnn_features, h', w')

        # 保存 Step1 的空间特征，便于 CAM（注意：这是未拼接临床 flag 前的 feature map）
        self.feat_step1 = words_cnn_feats.view(B * N_b * N_w, words_cnn_feats.shape[1], words_cnn_feats.shape[2], words_cnn_feats.shape[3])

        # 原有的 bl_flag 空间注意力：把临床 flag 扩展到空间
        bl_flag = batch['clinical_bl_flag'].view(B, 1, 1, 3, 1, 1)
        bl_flag = bl_flag.repeat(1, N_b, N_w, 1, *words_cnn_feats.shape[-2:])
        bl_flag = bl_flag.view(B * N_b * N_w, 3, *words_cnn_feats.shape[-2:])
        attn_words_cnn = torch.sigmoid(self.attn_layer(torch.cat([words_cnn_feats, bl_flag], dim=1)))
        words_cnn_feats = (words_cnn_feats * attn_words_cnn).view(B, N_b, N_w, -1)

        # 投影到 words token 空间
        words_cnn = self.project_words(words_cnn_feats)

        # STEP2: Words->Bags 自注意力
        words_flat = words_cnn.view(B * N_b, N_w, -1)
        words_attn, _ = self.attn_over_words(words_flat, words_flat, words_flat)
        words_attn = words_attn.view(B, N_b, N_w, -1)
        w_energy = F.softmax(self.words_weight_fn(words_attn).squeeze(-1), dim=-1).unsqueeze(-1)
        bags_features = torch.sum(words_attn * w_energy, dim=-2)
        bags_features = self.ffn_attn_w2b_lst(bags_features)
        mask_words_pred = self.words_classifier(words_attn)

        # STEP3: Bags->Image 自注意力
        bags_flat = bags_features
        bags_attn, _ = self.attn_over_bags(bags_flat, bags_flat, bags_flat)
        b_energy = F.softmax(self.bags_weight_fn(bags_attn).squeeze(-1), dim=-1).unsqueeze(-1)
        image_features = torch.sum(bags_attn * b_energy, dim=-2)
        image_features = self.ffn_b2i(image_features)
        mask_bags_pred = self.bags_classifier(bags_attn)

        # STEP4: 处理缺失的病理模态
        has_path = batch['bl_flag'].to(torch.bool).view(B)
        path_img = torch.where(has_path.view(-1, 1), image_features, self.missing_embed.expand(B, -1))

        # STEP5: 临床特征映射 —— 修改为：每个字段独立投影 -> 串联 -> 统一投影（与 YXModel 一致）
        # 注意：这里假定 batch 中包含以下键（若命名不同，请对应调整）：
        clin_keys = [
            'clinical_bl_qfjc',
            'clinical_bl_grade',
            'clinical_bl_AJCC',
            'clinical_bl_Tis',
            'clinical_bl_T',
            'clinical_bl_M',
            'clinical_bl_N'
        ]
        # 对应 self._clin_fcs 中的顺序逐个映射
        clin_embs = [
            fc(batch[k]) for fc, k in zip(self._clin_fcs, clin_keys)
        ]  # list of (B, C)
        clin_cat = torch.cat(clin_embs, dim=1)  # (B, 8*C)
        clinical_feats = self.clinical_proj(clin_cat)  # (B, C)

        # STEP6: ViT 融合（图像 + 临床）
        tokens = torch.stack([path_img, clinical_feats], dim=1)  # (B, 2, C)
        # key_padding_mask: True 表示要 mask 的位置
        pad_mask = torch.stack([~has_path, torch.zeros(B, dtype=torch.bool, device=path_img.device)], dim=1)
        x = tokens
        for layer in self.transformer_layers:
            x = layer(x, src_key_padding_mask=pad_mask)  # 显式传递关键字参数
        fused_tokens = x
        fused = fused_tokens[:, 0, :] + path_img  # 使用 token0（path_img）作为融合后的表示

        bl_pred = self.classifier(fused)
        bl_risk_score = self.risk_fc(fused).squeeze(-1)

        return {
            'feat': fused,
            "pred": bl_pred,
            "risk_score": bl_risk_score,
            'mask_words_pred': mask_words_pred,
            'mask_bags_pred': mask_bags_pred
        }
