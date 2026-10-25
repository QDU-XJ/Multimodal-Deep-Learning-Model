import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from timm.layers import DropPath
from torchvision.models import MNASNet
from model.utils import FFN


class TransformerEncoderLayerDP(nn.Module):
    """
    带 Stochastic Depth 的 Transformer Encoder Block
    """

    def __init__(self, d_model, nhead, dropout, drop_path_rate):
        super().__init__()
        # Multi-Head Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-Forward Network
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # --- Self-Attention + Residual + Norm ---
        attn_out, _ = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.drop_path1(self.dropout(attn_out))
        src = self.norm1(src)

        # --- Feed-Forward + Residual + Norm ---
        ff = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.drop_path2(ff)
        src = self.norm2(src)
        return src


class YXModel(nn.Module):
    """
    影像（YingXiang）主干 + 病灶注意力 + 融合 Transformer
    """

    def __init__(self, opts):
        super().__init__()
        self.opts = opts

        # —— 用来存放三阶段的 feature maps 和 grads —— #
        self.feat_step1 = None  # (#1 CNN+空间注意力) 输出：B*Nl × C_out × h' × w'
        self.feat_step2 = None  # (#2 病灶自注意力) 输出：B × Nl × C_out
        self.feat_step5 = None  # (#5 Transformer 融合) 输出：B × 3 × C_out  （每层都可以取）

        # 1) CNN Backbone
        if opts.yx_cnn_name.lower() == "mnasnet":
            backbone = MNASNet(alpha=1.0, num_classes=2)
            backbone = nn.Sequential(*list(backbone.children())[:-1])  # 去掉 classifier
        else:
            backbone = getattr(models, opts.yx_cnn_name)(
                pretrained=opts.yx_cnn_pretrained
            )
            # 去掉最后的全局池化或 classifier
            backbone = nn.Sequential(*list(backbone.children())[:-1])

        self.cnn = backbone
        # 降维到 opts.yx_out_features
        self.cnn_reduce = nn.Conv2d(
            in_channels=512,
            out_channels=opts.yx_out_features,
            kernel_size=1
        )
        # 空间注意力融合病灶和临床 yx_flag
        self.attn_layer = nn.Conv2d(
            in_channels=opts.yx_out_features + 3,
            out_channels=1,
            kernel_size=1
        )
        # 投影到 Transformer token
        self.cnn_project = nn.Linear(opts.yx_out_features, opts.yx_out_features)

        # 2) 病灶自注意力（Lesions → Patient）
        self.attn_over_lesions = nn.MultiheadAttention(
            embed_dim=opts.yx_out_features,
            num_heads=opts.yx_attn_heads,
            dropout=opts.yx_dropout,
            batch_first=True
        )
        self.ffn_attn_l2p = FFN(input_dim=opts.yx_out_features, scale=4, p=opts.yx_dropout)
        self.lesions_weight_fn = nn.Linear(opts.yx_out_features, 1, bias=False)
        self.attn_dropout = nn.Dropout(opts.yx_attn_dropout)
        self.attn_fn = nn.Softmax(dim=-1)
        self.lesions_classifier = nn.Linear(opts.yx_out_features, opts.n_classes)

        # 3) Radiomics 与临床特征投影
        self.radiomics_fc = nn.Linear(24, opts.yx_out_features)
        self.radiomics_norm = nn.LayerNorm(opts.yx_out_features)

        # Clinical: 一组小 FC，把 one-hot 转到同一维度
        self._clin_fcs = nn.ModuleList([
            nn.Linear(8, opts.yx_out_features),  # pos_lesion
            nn.Linear(3, opts.yx_out_features),  # form
            nn.Linear(2, opts.yx_out_features),  # size
            nn.Linear(2, opts.yx_out_features),  # thickness
            nn.Linear(2, opts.yx_out_features),  # ca
            nn.Linear(2, opts.yx_out_features),  # stalk
            nn.Linear(3, opts.yx_out_features),  # num
            nn.Linear(2, opts.yx_out_features),  # border
            nn.Linear(2, opts.yx_out_features),  # necrosis
            nn.Linear(2, opts.yx_out_features),  # extra_in
            nn.Linear(2, opts.yx_out_features),  # lymph_node
            nn.Linear(2, opts.yx_out_features),  # CT_A
            nn.Linear(2, opts.yx_out_features),  # CT_D
            nn.Linear(2, opts.yx_out_features),  # CT_V
        ])
        # ===== 新增：将所有临床特征串联后再投影回 opts.yx_out_features =====
        n_clin_fields = len(self._clin_fcs)  # 14
        self.clinical_proj = nn.Linear(n_clin_fields * opts.yx_out_features, opts.yx_out_features)

        # 4) 融合 Transformer 层
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayerDP(
                d_model=opts.yx_out_features,
                nhead=opts.yx_attn_heads,
                dropout=opts.yx_dropout,
                drop_path_rate=opts.yx_drop_path_rate
            ) for _ in range(opts.yx_vit_layers)
        ])

        self.classifier = nn.Linear(opts.yx_out_features, opts.n_classes)
        self.risk_fc = nn.Linear(opts.yx_out_features, 1)

    def forward(self, batch):
        B, Nl, C, H, W = batch['lesions'].shape

        # +++ 关键修改：应用ROI掩码 +++
        # 假设batch中已包含'roi_mask'，形状为(B, Nl, 3, H, W)
        x = batch['lesions'].view(B * Nl, C, H, W)

        # 获取ROI掩码并展平
        roi_mask = batch['roi_mask'].view(B * Nl, C, H, W)  # 形状(B*Nl, 3, H, W)

        # 应用ROI掩码：只保留ROI区域的CT值
        x = x * roi_mask  # 逐通道相乘

        # --- Step1: CNN 提取 & 空间注意力 ---
        feat = self.cnn(x)  # (B*Nl, feat_c, h', w')
        feat = self.cnn_reduce(feat)  # (B*Nl, C_out, h', w')

        # —— 存下来，后面可以直接拿做 CAM —— #
        self.feat_step1 = feat

        # 取 yx_flag 填充空间 map
        yx_flag = batch['clinical_yx_flag'].view(B, 1, 3, 1, 1) \
            .repeat(1, Nl, 1, feat.shape[-2], feat.shape[-1]) \
            .view(B * Nl, 3, feat.shape[-2], feat.shape[-1])
        att_map = torch.sigmoid(self.attn_layer(torch.cat([feat, yx_flag], dim=1)))
        feat = feat * att_map  # 空间加权
        feat = feat.view(B, Nl, -1)  # (B, Nl, C_out)
        patient_from_lesions = self.cnn_project(feat)

        # --- Step2: 病灶自注意力 → Patient Vector ---
        attn_out, _ = self.attn_over_lesions(
            patient_from_lesions,
            patient_from_lesions,
            patient_from_lesions
        )  # (B, Nl, C_out)
        energy = F.softmax(self.lesions_weight_fn(attn_out).squeeze(-1), dim=-1).unsqueeze(-1)
        patient_vec = torch.sum(attn_out * energy, dim=1)  # (B, C_out)
        patient_vec = self.ffn_attn_l2p(patient_vec)  # (B, C_out)
        lesions_pred = self.lesions_classifier(attn_out)  # (B, Nl, n_classes)

        # —— 同样存下来，后面就能 visualize 这个多头注意力的输出 —— #
        self.feat_step2 = attn_out

        # --- Step3: 临床 & Radiomics 特征 ---
        clin_feats = torch.stack([
            fc(batch[f'clinical_{name}'])
            for name, fc in zip([
                'yx_pos_lesion', 'yx_form', 'yx_size', 'yx_thickness', 'yx_ca',
                'yx_stalk', 'yx_num', 'yx_border', 'yx_necrosis', 'yx_extra_in',
                'yx_lymph_node', 'yx_CT_A', 'yx_CT_D', 'yx_CT_V'
            ], self._clin_fcs)
        ], dim=1).mean(dim=1)  # (B, C_out)

        # 1) 各子特征分别投影
        clin_embs = [
            fc(batch[f'clinical_{name}'])
            for name, fc in zip([
                'yx_pos_lesion', 'yx_form', 'yx_size', 'yx_thickness', 'yx_ca',
                'yx_stalk', 'yx_num', 'yx_border', 'yx_necrosis', 'yx_extra_in',
                'yx_lymph_node', 'yx_CT_A', 'yx_CT_D', 'yx_CT_V'
            ], self._clin_fcs)
        ]  # list of (B, C_out)

        # 2) 串联
        clin_cat = torch.cat(clin_embs, dim=1)  # (B, 14 * C_out)
        # 3) 投影回 (B, C_out)
        clin_feats = self.clinical_proj(clin_cat)  # (B, C_out)

        r = batch['yx_radiomics_feat']  # (B, Nr, feat_dim)
        r = self.radiomics_fc(r)  # (B, Nr, C_out)
        r = self.radiomics_norm(r)
        radiomics_feat = r.mean(dim=1, keepdim=True)  # (B, 1, C_out)

        # --- Step4: 构造 ViT 输入 Tokens ---
        tokens = torch.cat([
            patient_vec.unsqueeze(1),  # (B,1,C_out)
            clin_feats.unsqueeze(1),  # (B,1,C_out)
            radiomics_feat  # (B,1,C_out)
        ], dim=1)  # (B,3,C_out)

        # --- Step5: Transformer 融合 ---
        x = tokens
        for layer in self.transformer_layers:
            x = layer(x)

        self.feat_step5 = x  # -> (B, 3, C_out)

        fused = x[:, 0, :] + patient_vec  # 用 CLS token + residual

        yx_pred = self.classifier(fused)  # 分类 logits
        yx_risk_score = self.risk_fc(fused).squeeze(-1)

        return {
            'feat': fused,
            'lesions_pred': lesions_pred,
            'pred': yx_pred,
            'risk_score': yx_risk_score,
        }