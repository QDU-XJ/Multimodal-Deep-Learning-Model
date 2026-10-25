import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np

class BLYXLoss(nn.Module):
    def __init__(self,
                 w_ce=1.5,
                 w_auc=0.8,  # 二分类 AUC/Ranking 损失权重
                 w_surv=1.0,
                 w_cindex=0.3,  # 生存分支 C‐index 排序损失权重
                 w_align=0.2,
                 w_cmc=0.05,
                 align_temperature=0.07,
                 cmc_temperature=0.1,
                 gamma_focal=2.0):
        """
        增强版多模态融合损失函数，包含：
            - 二分类 Focal Loss (w_ce) + AUC/Ranking Loss (w_auc)
             - 生存分析 DeepSurv Loss (w_surv) + C‐index 排序损失 (w_cindex)
        """
        super().__init__()
        # ----------------- 基础损失组件 -----------------
        self.ce_criterion   = FocalLoss(num_class=2, alpha=[0.22, 0.78],gamma=gamma_focal)
        self.surv_criterion = DeepSurvLoss()
        self.align_criterion = InfoNCEAlignLoss(temperature=align_temperature)
        # ----------------- 其他新增损失组件 -----------------
        self.cmc_criterion = CrossModalConsistencyLoss(temperature=cmc_temperature)

        # ----------------- 权重配置 -----------------
        self.w_ce     = w_ce
        self.w_auc    = w_auc
        self.w_surv = w_surv
        self.w_cindex = w_cindex
        self.w_align = w_align
        self.w_cmc = w_cmc

    def forward(self, batch, results):
        total_loss = 0.0
        loss_dict = {}

        # ================= 二分类分支：Focal Loss + AUC/Ranking Loss =================
        if 'pred' in results:
            # 仅保留 label != -1 的样本
            ind = torch.where(batch['label'] != -1)
            if ind[0].numel() > 0:
                # logits_all 形状 (N, 2)，不做 squeeze
                logits_all = results['pred'][ind]       # (N, 2)
                targets    = batch['label'][ind].long() # (N,)

                # 1a) Focal Loss
                ce_loss = self.ce_criterion(logits_all, targets)
                weighted_ce = self.w_ce * ce_loss
                total_loss = total_loss + weighted_ce
                loss_dict['ce_loss'] = ce_loss.item()


                # 1b) 近似 AUC/Ranking 损失：用正类的 logit 做 pairwise 排序
                scores = logits_all[:, 1]  # 取第二列作为“正例的分数”
                auc_loss = self._auc_approx_loss(scores, targets)
                weighted_auc = self.w_auc * auc_loss
                loss_dict['auc_loss'] = auc_loss.item()
                total_loss = total_loss + weighted_auc
                loss_dict['auc_loss'] = auc_loss.item()

        # ================= 2. 生存分支：DeepSurv Loss + C‐index 排序损失 =================
        if 'risk_score' in results:
            P_risk = results['risk_score'].squeeze()   # (B,) 或者 0维
            T, E   = batch['RFS'], batch['RFSCensor']

            # 2a) DeepSurv Loss
            surv_loss = self.surv_criterion(P_risk, T, E)
            if not torch.isnan(surv_loss):
                weighted_surv = self.w_surv * surv_loss
                loss_dict['surv_loss'] = surv_loss.item()
            else:
                weighted_surv = torch.tensor(0.0, device=P_risk.device)

            # 2b) C‐index 排序损失 (负的 C-index)
            cindex_loss = self._c_index_loss(P_risk, T, E)
            weighted_cidx = self.w_cindex * cindex_loss
            loss_dict['cindex_loss'] = cindex_loss.item()

            # 2c) 合并
            loss_surv = weighted_surv + weighted_cidx
            total_loss = total_loss + loss_surv
            loss_dict['loss_surv'] = loss_surv.item()

        # ================= 3. 模态对齐 (InfoNCE) =================
        if "bl_com_feat" in results and "yx_com_feat" in results:
            blyx_flag = results["blyx_flag"].squeeze(dim=1).bool()
            bl_feats = results["bl_com_feat"][blyx_flag]
            yx_feats = results["yx_com_feat"][blyx_flag]
            if bl_feats.size(0) > 1:
                align_loss = self.align_criterion(bl_feats, yx_feats)
                if not torch.isnan(align_loss):
                    total_loss = total_loss + self.w_align * align_loss
                    loss_dict['align_loss'] = align_loss.item()

        # ================= 4. 跨模态一致性 (CMC) =================
        if 'bl_com_feat' in results and 'yx_com_feat' in results:
            cmc_loss = self.cmc_criterion(results['bl_com_feat'], results['yx_com_feat'])
            total_loss = total_loss + self.w_cmc * cmc_loss
            loss_dict['cmc_loss'] = cmc_loss.item()

        # ================= 梯度保护 =================
        if torch.isnan(total_loss).any():
            print(f"[WARNING] NaN loss detected! Details: {loss_dict}")
            return torch.zeros_like(total_loss, requires_grad=True)
        return total_loss


# ==================== 辅助函数：近似 AUC/Ranking Loss ====================
    def _auc_approx_loss(self,
                         scores: torch.Tensor,
                         labels: torch.Tensor,
                         margin: float = 0.0) -> torch.Tensor:
        """
        近似 AUC 的排序损失 (Logistic‐style)：
          对当前 batch 中所有正例‐负例对 (i,j)，推动 score[i] > score[j]。
        scores: (N,) “正类” logits
        labels: (N,) 二分类 {0,1}
        """
        device = scores.device
        pos_idx = torch.nonzero(labels == 1, as_tuple=False).squeeze(1)
        neg_idx = torch.nonzero(labels == 0, as_tuple=False).squeeze(1)
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 构造正负对的 Cartesian product，随机采样 K 对
        pos_neg = torch.cartesian_prod(pos_idx, neg_idx)  # (|P|*|N|, 2)
        K = min(pos_neg.size(0), 512)
        if pos_neg.size(0) > K:
            perm = torch.randperm(pos_neg.size(0), device=device)
            pos_neg = pos_neg[perm[:K]]

        pi = pos_neg[:, 0]
        nj = pos_neg[:, 1]
        s_i = scores[pi]  # (K,)
        s_j = scores[nj]  # (K,)

        # logistic‐style 排序损失：当 s_i >> s_j, sigmoid(margin + s_j - s_i)→0
        loss = torch.sigmoid(margin + s_j - s_i)
        return loss.mean()
    # ==================== 辅助函数：C‐index 排序损失 ====================
    def _c_index_loss(self,
                      risks: torch.Tensor,
                      T: torch.Tensor,
                      E: torch.Tensor) -> torch.Tensor:
        """
        计算“负的 C‐index”作为损失：
          对所有 i, j s.t. T[i] < T[j] 且 E[i]==1, 推动 risks[i] > risks[j]。
        risks: (B,) 生存风险分数 或 标量
        T:     (B,) 生存/复发时间
        E:     (B,) 事件指示 (1=事件发生, 0=删失)
        """
        # 如果 risks 已被 squeeze 为 0 维，无法做 pairwise 排序
        if risks.dim() == 0 or risks.numel() < 2:
            return torch.tensor(0.0, device=risks.device, requires_grad=True)

        device = risks.device
        idx = torch.arange(risks.size(0), device=device)

        # 仅对 E[i] == 1 的样本 i 参与计算
        pos_i = idx[E == 1]
        if pos_i.numel() < 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        total_pairs = 0
        loss_sum = 0.0

        for i in pos_i:
            # 找到所有 j 使 T[j] > T[i]
            valid_j = idx[(T > T[i]).nonzero(as_tuple=False).squeeze(1)]
            if valid_j.numel() == 0:
                continue
            total_pairs += valid_j.numel()
            s_i = risks[i].expand_as(valid_j)  # (num_j,)
            s_j = risks[valid_j]               # (num_j,)
            # logistic‐style C‐index: 当 s_i >> s_j, sigmoid(s_j - s_i) → 0
            diff = s_j - s_i
            loss_sum += torch.sum(torch.sigmoid(diff))

        if total_pairs == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return loss_sum / total_pairs



# 用于计算焦点损失（Focal Loss），这是一种在处理类别不平衡问题时常用的损失函数
class FocalLoss(nn.Module):
    def __init__(self, num_class, alpha, gamma, balance_index=-1, smooth=0.1, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        # ind = (target != self.ignore_index)
        # input = input[ind, ...]
        # target = target[ind]
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class DeepSurvLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def _compute_loss(self, P, T, E, M, mode):
        # —— 新增：保底，使 P,T,E 至少都是 1D 张量 —— #
        if not isinstance(P, torch.Tensor):
            P = torch.tensor(P, device=M.device, dtype=torch.float32)
        if P.dim() == 0:
            P = P.unsqueeze(0)
        # T 和 E 同理
        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, device=M.device, dtype=torch.float32)
        if T.dim() == 0:
            T = T.unsqueeze(0)
        if not isinstance(E, torch.Tensor):
            E = torch.tensor(E, device=M.device, dtype=torch.float32)
        if E.dim() == 0:
            E = E.unsqueeze(0)

        # 重新计算 batch size
        B = P.shape[0]

        # —— 如果只有一个样本，直接返回 0-loss —— #
        #    或者根据你的需求返回标量版的 Cox 损失
        if B == 1:
            return P.new_tensor(0.0)

        # —— 原有逻辑 —— #
        # 1) 避免 exp 溢出
        P0 = P - P.max()

        # 2) risk set 的 log-sum-exp
        mask = M.bool()  # (B,B)
        P_mat = P0.unsqueeze(0).expand(B, B)  # (B,B)
        neg_inf = torch.tensor(-1e9, device=P.device)
        P_masked = torch.where(mask, P_mat, neg_inf)
        log_sum_exp = torch.logsumexp(P_masked, dim=1)  # (B,)

        # 3) 只对有事件的样本计入
        if mode == 'risk':
            E = E.float() * (M.sum(dim=1) > 0).float()
        else:
            E = (M.sum(dim=1) > 0).float()

        # 4) 避免除以 0
        denom = E.sum()
        if denom == 0:
            return P.new_tensor(0.0)

        # 5) 负对数似然
        losses = -((P0 - log_sum_exp) * E)
        return losses.sum() / denom

    def forward(self, P_risk, T, E):
        # P: (B,)
        # T: (B,)
        # E: (B,) \in {0, 1}
        M_risk = T.unsqueeze(dim=1) <= T.unsqueeze(dim=0) # (B, B)
        loss_risk = self._compute_loss(P_risk, T, E, M_risk, mode='risk')
        return loss_risk

class InfoNCEAlignLoss(nn.Module):
    """
    基于 InfoNCE 的对齐损失:
    将两个模态同批次中的正样本对 (i,i) 拉近, 其他 (i,j), j!=i 为负样本
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        """
        z1, z2: shape (N, D) 两个模态同批次抽样特征
        """
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        loss_12 = F.cross_entropy(sim_matrix, labels)
        loss_21 = F.cross_entropy(sim_matrix.t(), labels)
        return (loss_12 + loss_21) / 2.0

# ==================== 内置其他损失组件实现（与原来相同） ====================
class CrossModalConsistencyLoss(nn.Module):
    """跨模态特征一致性损失 (CMC)"""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.cosine = nn.CosineSimilarity(dim=2)

    def forward(self, mod1_feats, mod2_feats):
        mod1 = F.normalize(mod1_feats, p=2, dim=1)
        mod2 = F.normalize(mod2_feats, p=2, dim=1)
        sim_matrix = torch.mm(mod1, mod2.T) / self.temperature
        targets = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        loss = (F.cross_entropy(sim_matrix, targets) +
                F.cross_entropy(sim_matrix.T, targets)) / 2
        return loss