#!/usr/bin/env python3
# test_only_blyx.py
import os
import gc
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc

from config import get_args
from model.blmodel import BLModel
from model.blyxmodel import BLYXModel
from model.yxmodel import YXModel
from dataloader.dataset import BLYXDataset
from utils.criterion import BLYXLoss
from utils.print_utils import print_log_message

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Tester:
    def __init__(self, opts, printer=print_log_message):
        self.opts = opts
        self.printer = printer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print_log_message(f"[INFO] Using device {self.device}", self.printer)

        # ensure save dir exists
        os.makedirs(self.opts.save_dir, exist_ok=True)
        self.log_path = os.path.join(self.opts.save_dir, "test_log.txt")

        # datasets and loader
        self.test_ds = BLYXDataset(opts, "test", opts.test_file, printer)
        self.test_loader = DataLoader(self.test_ds,
                                      batch_size=opts.batch_size, shuffle=False,
                                      num_workers=opts.data_workers, pin_memory=True)

        # build models and wrapper
        bl_model = BLModel(opts).to(self.device)
        yx_model = YXModel(opts).to(self.device)
        self.model = BLYXModel(opts, bl_model, yx_model).to(self.device)

        # criterion (to compute loss if you want)
        self.criterion = BLYXLoss(opts.w_ce, opts.w_auc, opts.w_surv, opts.w_cindex,
                                  opts.w_align, opts.w_cmc, opts.align_temperature,
                                  opts.cmc_temperature, opts.gamma_focal).to(self.device)

        # best model path (same as training script)
        self.best_model_path = os.path.join(opts.save_dir, "best_model.pth")

    def log(self, msg):
        print_log_message(msg, self.printer)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}\n")
        except Exception:
            pass

    def _prepare_batch(self, batch):
        # convert numeric tensors to device and float; keep strings as-is
        for k, v in batch.items():
            # v may be list / tensor; original script checks v[0] type to see if string
            try:
                sample0 = v[0]
            except Exception:
                sample0 = None
            # if it's a string (list of strings) -> skip conversion
            if isinstance(sample0, str):
                continue
            # else convert to float tensor on device
            try:
                batch[k] = v.float().to(self.device)
            except Exception:
                # fallback: if already tensor but no .float (e.g., int64), cast
                try:
                    batch[k] = torch.tensor(v).float().to(self.device)
                except Exception:
                    # leave as-is if cannot convert
                    pass
        return batch

    def test_once(self):
        if not os.path.exists(self.best_model_path):
            self.log(f"[ERROR] best model file not found at {self.best_model_path}. Exiting.")
            return

        # load state_dict (map to device)
        try:
            state = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.log(f"[INFO] Loaded best model from {self.best_model_path}")
        except Exception as e:
            self.log(f"[ERROR] Failed to load model state: {e}")
            return

        self.model.eval()
        total_loss = 0.0
        rec_ids, rec_labels, rec_rfs, rec_censor, rec_preds, rec_risks = [], [], [], [], [], []

        all_preds_list, all_risks_list, all_events_list, all_times_list = [], [], [], []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                # prepare batch to device
                batch = self._prepare_batch(batch)

                # forward
                out = self.model(batch)
                # compute loss if possible
                try:
                    loss = self.criterion(batch, out)
                    total_loss += float(loss.item())
                except Exception:
                    # if criterion fails, skip loss accumulation
                    pass

                # preds: probability for class 1
                try:
                    preds = torch.softmax(out['pred'], dim=1)[:, 1].detach().cpu().numpy()
                except Exception:
                    # fallback if pred is single-dim risk
                    preds = out.get('pred')
                    if isinstance(preds, torch.Tensor):
                        preds = preds.detach().cpu().numpy()
                    else:
                        preds = np.zeros(len(batch.get('id', [])))

                risks = out.get('risk_score')
                if isinstance(risks, torch.Tensor):
                    risks = risks.detach().cpu().numpy()
                else:
                    risks = np.zeros_like(preds)

                raw_ids = batch.get('id', [])
                ids = []
                for x in raw_ids:
                    if isinstance(x, torch.Tensor):
                        try:
                            ids.append(int(x.detach().cpu().item()))
                        except Exception:
                            ids.append(x.detach().cpu().numpy().tolist())
                    else:
                        ids.append(x)

                labels = batch.get('label')
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy().astype(int)
                else:
                    labels = np.array(labels).astype(int)

                times = batch.get('RFS')
                if isinstance(times, torch.Tensor):
                    times = times.cpu().numpy()
                else:
                    times = np.array(times, dtype=float)

                censor = batch.get('RFSCensor')
                if isinstance(censor, torch.Tensor):
                    censor = censor.cpu().numpy().astype(int)
                else:
                    censor = np.array(censor).astype(int)

                # extend records
                rec_ids.extend(ids)
                rec_labels.extend(labels.tolist())
                rec_rfs.extend(times.tolist())
                rec_censor.extend(censor.tolist())
                rec_preds.extend(preds.tolist())
                rec_risks.extend(risks.tolist())

                # metrics arrays
                events = censor.astype(bool)
                mask = np.ones_like(events, dtype=bool)
                if mask.sum() > 0:
                    all_preds_list.append(preds[mask])
                    all_risks_list.append(risks[mask])
                    all_events_list.append(events[mask])
                    all_times_list.append(times[mask])

        n_batches = len(self.test_loader)
        avg_loss = total_loss / n_batches if n_batches > 0 else float('nan')

        # compute metrics
        if all_events_list:
            preds_arr = np.concatenate(all_preds_list)
            risks_arr = np.concatenate(all_risks_list)
            events_arr = np.concatenate(all_events_list)
            times_arr = np.concatenate(all_times_list)

            # ROC AUC (for binary event)
            try:
                auc = roc_auc_score(events_arr, preds_arr) if len(np.unique(events_arr)) == 2 else 0.0
            except Exception:
                auc = 0.0

            # concordance index (note: concordance expects higher risk -> shorter survival,
            # but your training used -risks in concordance_index call)
            try:
                cidx = concordance_index(times_arr, -risks_arr, events_arr)
            except Exception:
                try:
                    cidx = concordance_index(times_arr, risks_arr, events_arr)
                except Exception:
                    cidx = 0.0

            # time-dependent AUCs @6,12,24
            try:
                y_struct = np.array(list(zip(events_arr, times_arr)),
                                    dtype=[('event', 'bool'), ('time', 'float64')])
                times_eval = np.array([6, 12, 24], dtype=float)
                tda, _ = cumulative_dynamic_auc(y_struct, y_struct, risks_arr, times_eval)
                t6, t12, t24 = tda
            except Exception:
                t6 = t12 = t24 = 0.0
        else:
            auc = cidx = t6 = t12 = t24 = 0.0

        # log summary
        summary = (f"[TEST] loss={avg_loss:.4f} auc={auc:.4f} cidx={cidx:.4f} "
                   f"tdAUC@[6,12,24]={t6:.3f},{t12:.3f},{t24:.3f}")
        self.log(summary)

        # save per-sample csv
        df = pd.DataFrame({
            "id": rec_ids,
            "label": rec_labels,
            "RFS": rec_rfs,
            "RFSCensor": rec_censor,
            "pred": rec_preds,
            "risk_score": rec_risks,
        })
        out_path = os.path.join(self.opts.save_dir, "test_results_bestmodel.csv")
        try:
            df.to_csv(out_path, index=False)
            self.log(f"[INFO] Saved test results to {out_path}")
        except Exception as e:
            self.log(f"[WARN] Failed to save test CSV: {e}")

        # final cleanup
        gc.collect()
        return {
            "loss": avg_loss, "auc": auc, "cindex": cidx, "tda6": t6, "tda12": t12, "tda24": t24,
            "csv": out_path
        }

if __name__ == "__main__":
    opts = get_args()
    set_seed(opts.seed)
    tester = Tester(opts, printer=print_log_message)
    results = tester.test_once()
    print("=== Test finished ===")
    print(results)