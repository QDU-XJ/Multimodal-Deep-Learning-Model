import os
import random
import time
import gc
import cv2
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc  # 时间依赖 AUC
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable

from config import get_args
from model.blmodel import BLModel
from model.blyxmodel import BLYXModel
from utils.print_utils import print_log_message
from utils.criterion import BLYXLoss
from dataloader.dataset import BLYXDataset
from model.yxmodel import YXModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self, opts, printer):
        self.opts = opts
        self.printer = printer

        # Device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print_log_message(f"[INFO] Using device {self.device}", printer)

        # 创建 CAM 主目录及子目录（保留初始化以不改变外部依赖）
        os.makedirs(opts.cam_save_dir, exist_ok=True)
        self.step1_dir = os.path.join(opts.cam_save_dir, "CTStep1")
        os.makedirs(self.step1_dir, exist_ok=True)
        self.bl_step1_dir = os.path.join(opts.cam_save_dir, "BLStep1")
        os.makedirs(self.bl_step1_dir, exist_ok=True)

        # 确保保存目录存在
        os.makedirs(opts.save_dir, exist_ok=True)

        # 日志文件路径
        self.log_path = os.path.join(opts.save_dir, "experiment_log.txt")
        try:
            try:
                opts_json = json.dumps(vars(self.opts), indent=2, ensure_ascii=False, default=str)
            except Exception:
                opts_json = repr(self.opts)
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("===== Experiment Log =====\n")
                f.write(f"Start time: {datetime.now().isoformat()}\n\n")
                f.write("=== Parameters ===\n")
                f.write(opts_json + "\n")
                f.write("=== End Parameters ===\n\n")
                f.flush()
            print_log_message(f"[INFO] Experiment log initialized at {self.log_path}", self.printer)
        except Exception as e:
            print_log_message(f"[WARN] Failed to initialize experiment log: {e}", self.printer)

        # 数据集和数据加载器
        self.train_ds = BLYXDataset(opts, "train", opts.train_file, printer)
        self.val_ds = BLYXDataset(opts, "val", opts.val_file, printer)
        self.test_ds = BLYXDataset(opts, "test", opts.test_file, printer)

        self.train_loader = DataLoader(self.train_ds,
                                       batch_size=opts.batch_size, shuffle=True,
                                       num_workers=opts.data_workers, pin_memory=True)
        self.val_loader = DataLoader(self.val_ds,
                                     batch_size=opts.batch_size, shuffle=False,
                                     num_workers=opts.data_workers, pin_memory=True)
        self.test_loader = DataLoader(self.test_ds,
                                      batch_size=opts.batch_size, shuffle=False,
                                      num_workers=opts.data_workers, pin_memory=True)

        # 模型和损失函数
        bl_model = BLModel(opts).to(self.device)
        yx_model = YXModel(opts).to(self.device)
        self.model = BLYXModel(opts, bl_model, yx_model).to(self.device)
        self.criterion = BLYXLoss(opts.w_ce, opts.w_auc, opts.w_surv, opts.w_cindex, opts.w_align, opts.w_cmc,
                                  opts.align_temperature, opts.cmc_temperature, opts.gamma_focal).to(self.device)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opts.lr,
            weight_decay=opts.weight_decay
        )

        # 学习率调度器: 线性 Warm-up + 余弦退火
        from torch.optim.lr_scheduler import LambdaLR, SequentialLR, CosineAnnealingLR

        warmup_epochs = getattr(opts, 'warmup_epochs', 10)

        # 1) Warm-up: 0->1 线性提升
        def warmup_lambda(epoch):
            return min((epoch + 1) / warmup_epochs, 1.0)

        warmup_scheduler = LambdaLR(self.optimizer, lr_lambda=warmup_lambda)

        # 2) 余弦退火
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(opts.epochs - warmup_epochs, 1),
            eta_min=getattr(opts, 'lr_min', 1e-5)
        )

        # 串联 schedulers
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )

        # 训练历史记录 - 增加测试指标的记录
        self.history = {
            "train_loss": [], "train_auc": [], "train_cindex": [],
            "val_loss": [], "val_auc": [], "val_cindex": [], "val_tda6": [], "val_tda12": [], "val_tda24": [],
            "test_loss": [], "test_auc": [], "test_cindex": [], "test_tda6": [], "test_tda12": [], "test_tda24": [],
        }

        # 最佳模型保存路径
        self.best_val_auc = 0.0
        self.best_model_path = os.path.join(opts.save_dir, "best_model.pth")
        os.makedirs(opts.save_dir, exist_ok=True)

    def log_to_file(self, message):
        """将消息写入日志文件"""
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - LOGS    - {message}\n")
                f.flush()
        except Exception as e:
            print_log_message(f"[WARN] Failed to write to log file: {e}", self.printer)

    def _save_three_images_aligned(self, original_rgb: np.ndarray, roi_mask_2d: np.ndarray, overlay_rgb: np.ndarray,
                                   target_size: tuple, save_basepath: str, model_type: str = "yx"):
        """
        保存三张图并保证大小一致，并分别放入对应的子文件夹:
          - original_rgb: HxWx3 (RGB uint8)
          - roi_mask_2d: HxW (float 0-1 或 0-255)
          - overlay_rgb: HxWx3 (RGB uint8)
          - target_size: (W, H)  -> OpenCV dsize
          - save_basepath: full path prefix (包含父目录)，例如 os.path.join(target_dir, "val_ep001_id123")
          - model_type: "yx" -> 使用 CT_* 子文件夹；"bl" -> 使用 BL_* 子文件夹
        结果保存为:
          <parent>/<CT|BL>_original/<basename>_original.png
          <parent>/<CT|BL>_roi/<basename>_roi.png
          <parent>/<CT|BL>_overlay/<basename>_overlay.png
        """
        try:
            W_t, H_t = target_size  # note: target_size passed as (W, H)
            # Prepare base paths and folders
            parent_dir = os.path.dirname(save_basepath)
            base_name = os.path.basename(save_basepath)

            if model_type.lower() in ("yx", "ct"):
                prefix = "CT"
            else:
                prefix = "BL"

            dir_orig = os.path.join(parent_dir, f"{prefix}_original")
            dir_roi = os.path.join(parent_dir, f"{prefix}_roi")
            dir_ov = os.path.join(parent_dir, f"{prefix}_overlay")
            os.makedirs(dir_orig, exist_ok=True)
            os.makedirs(dir_roi, exist_ok=True)
            os.makedirs(dir_ov, exist_ok=True)

            # Ensure original is uint8 RGB
            orig = original_rgb
            if orig.dtype != np.uint8:
                orig = np.clip(orig, 0, 255).astype(np.uint8)

            # Resize original -> target
            orig_resized = cv2.resize(orig, (W_t, H_t), interpolation=cv2.INTER_LINEAR)

            # Prepare roi mask: normalize to 0-255 uint8 single channel
            roi = roi_mask_2d
            # If float 0-1 -> convert to 0-255
            if np.issubdtype(roi.dtype, np.floating):
                roi_u8 = (np.clip(roi, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                # integer type: normalize if max >255 or already in 0-255
                if roi.max() <= 1:
                    roi_u8 = (roi * 255).astype(np.uint8)
                else:
                    roi_u8 = np.clip(roi, 0, 255).astype(np.uint8)

            # Resize roi mask -> target (use INTER_NEAREST to keep binary edges)
            roi_resized = cv2.resize(roi_u8, (W_t, H_t), interpolation=cv2.INTER_NEAREST)

            # Ensure overlay is uint8 RGB
            ov = overlay_rgb
            if ov.dtype != np.uint8:
                ov = np.clip(ov, 0, 255).astype(np.uint8)
            # Resize overlay -> target
            ov_resized = cv2.resize(ov, (W_t, H_t), interpolation=cv2.INTER_LINEAR)

            # Save images
            orig_path = os.path.join(dir_orig, base_name + "_original.png")
            roi_path = os.path.join(dir_roi, base_name + "_roi.png")
            ov_path = os.path.join(dir_ov, base_name + "_overlay.png")

            # OpenCV expects BGR for imwrite when saving color images
            cv2.imwrite(orig_path, cv2.cvtColor(orig_resized, cv2.COLOR_RGB2BGR))
            cv2.imwrite(roi_path, roi_resized)
            cv2.imwrite(ov_path, cv2.cvtColor(ov_resized, cv2.COLOR_RGB2BGR))

            return orig_path, roi_path, ov_path
        except Exception as e:
            print_log_message(f"[WARN] Failed to save aligned images: {e}", self.printer)
            return None, None, None

    def _epoch(self, loader, epoch, split, generate_cam=False):
        is_train = split == "train"
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        all_preds, all_risks, all_events, all_times = [], [], [], []
        rec_ids, rec_labels, rec_rfs, rec_censor = [], [], [], []
        rec_preds, rec_risks = [], []

        with torch.set_grad_enabled(is_train):
            for batch in loader:
                for k, v in batch.items():
                    if not isinstance(v[0], str):
                        batch[k] = v.float().to(self.device)

                out = self.model(batch)
                loss = self.criterion(batch, out)
                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                total_loss += loss.item()

                # 处理预测结果
                preds = torch.softmax(out['pred'], dim=1)[:, 1].detach().cpu().numpy()
                risks = out['risk_score'].detach().cpu().numpy()
                raw_ids = batch['id']
                ids = []
                for x in raw_ids:
                    if isinstance(x, torch.Tensor):
                        ids.append(x.detach().cpu().item())
                    else:
                        ids.append(x)
                labels = batch['label'].cpu().numpy().astype(int)
                times = batch['RFS'].cpu().numpy()
                censor = batch['RFSCensor'].cpu().numpy().astype(int)

                # 记录每个样本的结果
                rec_ids.extend(ids)
                rec_labels.extend(labels.tolist())
                rec_rfs.extend(times.tolist())
                rec_censor.extend(censor.tolist())
                rec_preds.extend(preds.tolist())
                rec_risks.extend(risks.tolist())

                # 计算指标用
                events = censor.astype(bool)
                mask = np.ones_like(events, dtype=bool)
                if mask.sum() > 0:
                    all_preds.append(preds[mask])
                    all_risks.append(risks[mask])
                    all_events.append(events[mask])
                    all_times.append(times[mask])

        if is_train:
            self.scheduler.step()

        avg_loss = total_loss / len(loader)

        # 计算评估指标
        if all_events:
            preds_arr = np.concatenate(all_preds)
            risks_arr = np.concatenate(all_risks)
            events_arr = np.concatenate(all_events)
            times_arr = np.concatenate(all_times)

            auc = roc_auc_score(events_arr, preds_arr) if len(np.unique(events_arr)) == 2 else 0.0
            cidx = concordance_index(times_arr, -risks_arr, events_arr)

            y_struct = np.array(list(zip(events_arr, times_arr)),
                                dtype=[('event', 'bool'), ('time', 'float64')])
            times_eval = np.array([6, 12, 24], dtype=float)
            tda, _ = cumulative_dynamic_auc(y_struct, y_struct, risks_arr, times_eval)
            t6, t12, t24 = tda
        else:
            auc = cidx = t6 = t12 = t24 = 0.0

        # 构建日志消息
        log_message = (
            f"[{split.capitalize()}][{epoch}/{self.opts.epochs}] "
            f"loss={avg_loss:.4f} auc={auc:.4f} cidx={cidx:.4f} "
            f"tdAUC@[6,12,24]={t6:.3f},{t12:.3f},{t24:.3f}"
        )

        # 打印并记录日志
        print_log_message(log_message, self.printer)
        self.log_to_file(log_message)

        # 保存训练/验证/测试结果（现在训练集也保存）
        if split in ("train", "val", "test"):
            df = pd.DataFrame({
                "id": rec_ids,
                "label": rec_labels,
                "RFS": rec_rfs,
                "RFSCensor": rec_censor,
                "pred": rec_preds,
                "risk_score": rec_risks,
            })
            path = os.path.join(self.opts.save_dir, f"{split}_epoch_{epoch:03d}.csv")
            try:
                df.to_csv(path, index=False)
                save_message = f"[INFO] Saved {split} details to {path}"
                print_log_message(save_message, self.printer)
                self.log_to_file(save_message)
            except Exception as e:
                err_msg = f"[WARN] Failed to save {split} csv to {path}: {e}"
                print_log_message(err_msg, self.printer)
                self.log_to_file(err_msg)

        return avg_loss, auc, cidx, t6, t12, t24

    def run(self):
        # 训练、验证和测试
        for ep in range(1, self.opts.epochs + 1):
            # 训练
            tr_loss, tr_auc, tr_cidx, _, _, _ = self._epoch(self.train_loader, ep, "train")

            # 验证
            val_loss, val_auc, val_cidx, vt6, vt12, vt24 = self._epoch(
                self.val_loader, ep, "val", generate_cam=False  # 训练中不生成CAM
            )

            # 每个epoch都进行测试
            tst_loss, tst_auc, tst_cidx, tt6, tt12, tt24 = self._epoch(
                self.test_loader, ep, "test", generate_cam=False  # 测试时不生成CAM
            )

            # 记录历史
            h = self.history
            h["train_loss"].append(tr_loss)
            h["train_auc"].append(tr_auc)
            h["train_cindex"].append(tr_cidx)
            h["val_loss"].append(val_loss)
            h["val_auc"].append(val_auc)
            h["val_cindex"].append(val_cidx)
            h["val_tda6"].append(vt6)
            h["val_tda12"].append(vt12)
            h["val_tda24"].append(vt24)
            h["test_loss"].append(tst_loss)
            h["test_auc"].append(tst_auc)
            h["test_cindex"].append(tst_cidx)
            h["test_tda6"].append(tt6)
            h["test_tda12"].append(tt12)
            h["test_tda24"].append(tt24)

            # 保存最佳模型
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_val_loss = val_loss
                self.best_val_cindex = val_cidx
                self.best_epoch = ep
                torch.save(self.model.state_dict(), self.best_model_path)
                best_model_message = f"[INFO] New best model (epoch {ep}) saved: val_auc={val_auc:.4f}"
                print_log_message(best_model_message, self.printer)
                self.log_to_file(best_model_message)

            gc.collect()

        # NOTE: CAM 生成功能已移除 —— 不再生成最佳模型的 CAMs

        # 绘制结果曲线，包含测试集结果
        epochs = np.arange(1, len(self.history["train_loss"]) + 1)
        fig, axes = plt.subplots(4, 1, figsize=(10, 24))

        # 绘制Loss曲线 (训练、验证、测试)
        for split, m in zip(["train", "val", "test"], ['o', 'x', '^']):
            axes[0].plot(self.history[f"{split}_loss"], marker=m, label=f"{split.capitalize()} Loss")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[0].grid(True)

        # 绘制AUC曲线 (训练、验证、测试)
        for split, m in zip(["train", "val", "test"], ['o', 'x', '^']):
            axes[1].plot(self.history[f"{split}_auc"], marker=m, label=f"{split.capitalize()} AUC")
        axes[1].set_title("ROC AUC")
        axes[1].legend()
        axes[1].grid(True)

        # 绘制C-index曲线 (训练、验证、测试)
        for split, m in zip(["train", "val", "test"], ['o', 'x', '^']):
            axes[2].plot(self.history[f"{split}_cindex"], marker=m, label=f"{split.capitalize()} C-index")
        axes[2].set_title("Concordance Index")
        axes[2].legend()
        axes[2].grid(True)

        # 绘制时间依赖AUC曲线 (验证、测试)
        axes[3].plot(np.arange(1, len(self.history["val_tda6"]) + 1), self.history["val_tda6"], marker='o',
                     label="Val td-AUC @6m")
        axes[3].plot(np.arange(1, len(self.history["val_tda12"]) + 1), self.history["val_tda12"], marker='x',
                     label="Val td-AUC @12m")
        axes[3].plot(np.arange(1, len(self.history["val_tda24"]) + 1), self.history["val_tda24"], marker='s',
                     label="Val td-AUC @24m")

        axes[3].plot(np.arange(1, len(self.history["test_tda6"]) + 1), self.history["test_tda6"], marker='o',
                     linestyle='--',
                     label="Test td-AUC @6m")
        axes[3].plot(np.arange(1, len(self.history["test_tda12"]) + 1), self.history["test_tda12"], marker='x',
                     linestyle='--',
                     label="Test td-AUC @12m")
        axes[3].plot(np.arange(1, len(self.history["test_tda24"]) + 1), self.history["test_tda24"], marker='s',
                     linestyle='--',
                     label="Test td-AUC @24m")

        axes[3].set_title("Time-dependent ROC AUC")
        axes[3].legend()
        axes[3].grid(True)

        plt.tight_layout()
        save_path = os.path.join(self.opts.save_dir, "blyx_result.png")
        plt.savefig(save_path)
        plot_message = f"[INFO] Saved curves to {save_path}"
        print_log_message(plot_message, self.printer)
        self.log_to_file(plot_message)
        plt.close()

        # 写入最终结果到日志
        try:
            final_lines = []
            final_lines.append("===== Final Experiment Results =====")
            final_lines.append(f"End time: {datetime.now().isoformat()}")
            final_lines.append("")
            final_lines.append("Best on validation:")
            final_lines.append(f"  best_epoch: {getattr(self, 'best_epoch', None)}")
            final_lines.append(f"  best_val_auc: {getattr(self, 'best_val_auc', float('nan'))}")
            final_lines.append(f"  best_val_loss: {getattr(self, 'best_val_loss', float('nan'))}")
            final_lines.append(f"  best_val_cindex: {getattr(self, 'best_val_cindex', float('nan'))}")
            final_lines.append("")
            final_lines.append("Test results at best epoch:")
            if hasattr(self, 'best_epoch') and self.best_epoch > 0 and self.best_epoch <= len(self.history["test_auc"]):
                best_idx = self.best_epoch - 1
                final_lines.append(f"  test_epoch: {self.best_epoch}")
                final_lines.append(f"  test_loss: {self.history['test_loss'][best_idx]:.6f}")
                final_lines.append(f"  test_auc: {self.history['test_auc'][best_idx]:.6f}")
                final_lines.append(f"  test_cindex: {self.history['test_cindex'][best_idx]:.6f}")
                final_lines.append(f"  test_tdAUC_6m: {self.history['test_tda6'][best_idx]:.6f}")
                final_lines.append(f"  test_tdAUC_12m: {self.history['test_tda12'][best_idx]:.6f}")
                final_lines.append(f"  test_tdAUC_24m: {self.history['test_tda24'][best_idx]:.6f}")
            else:
                final_lines.append("  No valid test results available for best epoch")
            final_lines.append("")
            final_lines.append("History lengths:")
            final_lines.append(f"  train_loss_len: {len(self.history['train_loss'])}")
            final_lines.append(f"  val_loss_len: {len(self.history['val_loss'])}")
            final_lines.append(f"  test_loss_len: {len(self.history['test_loss'])}")
            final_lines.append("\n====================================\n")

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("\n".join(final_lines))
                f.flush()
            log_append_message = f"[INFO] Final experiment results appended to {self.log_path}"
            print_log_message(log_append_message, self.printer)
        except Exception as e:
            log_error_message = f"[WARN] Failed to write final experiment results to log: {e}"
            print_log_message(log_error_message, self.printer)
            self.log_to_file(log_error_message)


if __name__ == "__main__":
    from utils.print_utils import print_log_message

    opts = get_args()
    set_seed(opts.seed)
    trainer = Trainer(opts, printer=print_log_message)
    trainer.run()
