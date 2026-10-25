import torch
import os
import torch.nn as nn
from model.blmodel import *
from model.cross_attn import CrossAttention
from model.yxmodel import *


class BLYXModel(nn.Module):
    def __init__(self, opts, bl_model, yx_model):
        super().__init__()
        self.opts = opts
        self.bl_model = bl_model
        self.yx_model = yx_model

        # 专属 & 通用映射
        self.bl_spe_fc = nn.Linear(opts.blyx_out_features, opts.blyx_out_features)
        self.yx_spe_fc = nn.Linear(opts.blyx_out_features, opts.blyx_out_features)
        self.bl_com_fc = nn.Linear(opts.blyx_out_features, opts.blyx_out_features)
        self.yx_com_fc = nn.Linear(opts.blyx_out_features, opts.blyx_out_features)

        # 混合特征映射
        self.blyx_fc = nn.Linear(opts.blyx_out_features * 3, opts.blyx_out_features)
        self.classifier = nn.Linear(opts.blyx_out_features, opts.n_classes)
        self.risk_predictor = nn.Linear(opts.blyx_out_features, 1)

        # 同步参数
        self.yx_param = nn.Parameter(torch.zeros(1, opts.blyx_out_features))
        self.bl_param = nn.Parameter(torch.zeros(1, opts.blyx_out_features))

        # 跨模态注意力：融合 bl_feat, yx_feat, clinical_feat 等多路特征
        self.cross_global = CrossAttention(
            opts.blyx_out_features,
            num_heads=opts.yx_attn_heads,
            dropout=opts.blyx_dropout
        )

        # 多任务分支：分类 & 生存
        hidden_dim = opts.blyx_out_features // 2
        dp = opts.dropout if hasattr(opts, 'dropout') else 0.4
        self.cls_branch = nn.Sequential(
            nn.Linear(opts.blyx_out_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dp),
            nn.Linear(hidden_dim, opts.n_classes)
        )
        self.risk_branch = nn.Sequential(
            nn.Linear(opts.blyx_out_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dp),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch, *args, **kwargs):
        B = batch['id'].shape[0]
        device = batch['id'].device

        # 1) 各自子模型前向
        yx_res = self.yx_model(batch, *args, **kwargs)
        yx_feat = self.yx_com_fc(yx_res['feat'])  # (B,C)
        bl_res = self.bl_model(batch, *args, **kwargs)
        bl_feat = self.bl_com_fc(bl_res['feat'])  # (B,C)

        # 2) 专属 & 同步处理
        yx_flag = batch['yx_flag'].unsqueeze(1).float()  # (B,1)
        yx_spe = self.yx_spe_fc(yx_res['feat'])
        yx_sync = self.yx_param.expand(B, -1)
        yx_spe = yx_spe * yx_flag + yx_sync * (1 - yx_flag)

        bl_flag = batch['bl_flag'].unsqueeze(1).float()  # (B,1)
        bl_spe = self.bl_spe_fc(bl_res['feat'])
        bl_sync = self.bl_param.expand(B, -1)
        bl_spe = bl_spe * bl_flag + bl_sync * (1 - bl_flag)

        # 确保特征维度正确
        assert yx_spe.dim() == 2, f"yx_spe应为2D，实际为{yx_spe.dim()}D，形状{yx_spe.shape}"
        assert bl_spe.dim() == 2, f"bl_spe应为2D（前），实际为{bl_spe.dim()}D，形状{bl_spe.shape}"

        # 3) 混合特征
        blyx_flag = bl_flag * yx_flag  # (B,1)
        com_feat = (yx_feat + bl_feat) / 2 * blyx_flag
        com_feat = com_feat * blyx_flag + yx_feat * yx_flag * (1 - blyx_flag)
        com_feat = com_feat * blyx_flag + bl_feat * bl_flag * (1 - blyx_flag)

        # 确保所有特征都是2D张量
        assert bl_spe.dim() == 2, f"bl_spe应为2D，实际为{bl_spe.dim()}D，形状{bl_spe.shape}"
        assert com_feat.dim() == 2, f"com_feat应为2D，实际为{com_feat.dim()}D，形状{com_feat.shape}"
        assert yx_spe.dim() == 2, f"yx_spe应为2D，实际为{yx_spe.dim()}D，形状{yx_spe.shape}"

        # 4) 初步拼接
        cat_feat = torch.cat([bl_spe, com_feat, yx_spe], dim=1)  # (B, 3C)
        fused = self.blyx_fc(cat_feat)  # (B, C)

        # 5) 全局跨模态融合：融合 fused / bl_feat / yx_feat 三路特征
        # 构建 key/value 序列 (B,4,C)
        key_tokens = torch.cat([
            fused.unsqueeze(1),  # (B,1,C) 融合特征
            bl_feat.unsqueeze(1),  # (B,1,C) BL特征
            yx_feat.unsqueeze(1),  # (B,1,C) YX特征
        ], dim=1).to(device)

        # 构造 key_padding_mask (B,4)
        # 规则：fused永不屏蔽；bl/yx根据flag屏蔽；
        fused_mask = torch.zeros(B, dtype=torch.bool, device=device)
        bl_mask = (~bl_flag.bool().squeeze(1)).to(device)  # True表示屏蔽
        yx_mask = (~yx_flag.bool().squeeze(1)).to(device)
        pad_mask = torch.stack([fused_mask, bl_mask, yx_mask], dim=1)  # (B,4)

        # Cross-Attention 融合
        out, attn_map = self.cross_global(
            query=fused.unsqueeze(1).to(device),  # (B,1,C)
            key=key_tokens,  # (B,4,C)
            value=key_tokens,  # (B,4,C)
            key_padding_mask=pad_mask  # (B,4)
        )
        fused = fused + out.squeeze(1)  # 残差连接 (B,C)

        # 保存注意力权重（包含临床特征的注意力）
        self.info_dict = {
            'id': batch['id'][0],
            'global_attn': attn_map.detach().cpu().numpy()  # 现在包含对临床特征的注意力权重
        }

        # 7) 下游预测
        pred = self.cls_branch(fused)
        risk = self.risk_branch(fused).squeeze(1)

        return {
            'pred': pred,
            'risk_score': risk,
            'blyxfeat': fused,
            'bl_com_feat': bl_feat,
            'yx_com_feat': yx_feat,
            'com_feat': com_feat,
            'bl_spe_feat': bl_spe,
            'yx_spe_feat': yx_spe,
            'blyx_flag': blyx_flag,
            'lesions_pred': yx_res.get('lesions_pred'),
            'mask_words_pred': bl_res.get('mask_words_pred'),
            'mask_bags_pred': bl_res.get('mask_bags_pred'),
        }