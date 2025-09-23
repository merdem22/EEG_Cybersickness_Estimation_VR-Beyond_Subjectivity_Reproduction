# trainer.py — article metrics + baseline + accuracy grid (no MAE_ratio)
import torch
import torch.nn.functional as F
from typing import Dict, Any
from torchutils.models import TrainerModel
from metrics import binary_accuracy_with_neighborhood  # keep leaky if you already use it

def _as_tensor_like(x, ref: torch.Tensor) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
        if t.device != ref.device: t = t.to(ref.device)
        if t.dtype  != ref.dtype:  t = t.to(ref.dtype)
        return t
    return torch.as_tensor(x, device=ref.device, dtype=ref.dtype)

def _to_vector(t: torch.Tensor, length: int) -> torch.Tensor:
    if t.ndim == 0: t = t.view(1)
    if t.ndim == 2 and t.size(1) == 1: t = t[:, 0]
    t = t.view(-1)
    if t.numel() == 1 and length != 1: t = t.repeat(length)
    if t.numel() != length: t = t[:length]
    return t.contiguous()

def _dilate_bin_1d(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary dilation via max-pool over a 1D window of size (2r+1). x: (N,) in {0,1}."""
    if radius <= 0:
        return x
    x3 = x.view(1, 1, -1)
    k  = 2 * radius + 1
    y  = F.max_pool1d(x3, kernel_size=k, stride=1, padding=radius)
    return y.view(-1)

class MyTrainerModel(TrainerModel):
    """
    Train with scalar loss; on eval/predict log:
      - MAE, MSE (model)
      - Baseline: mean/l1_loss (MAE), mean/mse_loss (MSE) using training-mean predictor
      - Accuracy (configured threshold & neighborhood)
      - Acc@0.10/r{0,1,2,5} grid
    Also stores self._buffer['last_metrics'] with counts for later aggregation.
    """
    def __init__(self, train_joy_mean: float, input_type: str, task: str, **kwds):
        if task != 'regression':
            raise ValueError("This runner is configured for the paper's regression setup only.")

        acc_tau = float(kwds.pop('acc_threshold', 0.10))
        acc_rad = int(kwds.pop('acc_neighborhood', 0))

        writable_scores = {
            'MAE', 'MSE', 'Accuracy',
            'mean/l1_loss', 'mean/mse_loss',
            'Acc@0.10/r0', 'Acc@0.10/r1', 'Acc@0.10/r2', 'Acc@0.10/r5'
        }
        super().__init__(**kwds, writable_scores=writable_scores)

        self.task = task
        self.input_type = input_type
        self._buffer['mean'] = float(train_joy_mean)  # baseline predictor value
        self._buffer['acc/tau'] = acc_tau             # configured τ
        self._buffer['acc/rad'] = acc_rad             # configured radius
        self._buffer['last_metrics'] = {}             # filled each eval step

    # ---------- batch utils ----------
    def model_output(self, batch: Dict[str, Any]) -> torch.Tensor:
        kwargs = {k: v for k, v in batch.items() if k != 'observation'}
        return self.model(**kwargs)

    def _extract_preds_targets(self, batch: Dict[str, Any]) -> (torch.Tensor, torch.Tensor):
        preds = self.model_output(batch)
        if isinstance(preds, torch.Tensor) and preds.ndim > 1 and preds.size(-1) == 1:
            preds = preds.squeeze(-1)
        targets = _as_tensor_like(batch['observation'], preds)
        targets = _to_vector(targets, length=preds.shape[0])
        preds   = _to_vector(preds,   length=preds.shape[0])
        return preds, targets

    # ---------- training / eval ----------
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        preds, targets = self._extract_preds_targets(batch)
        loss = self.criterion(preds, targets)
        try:
            self._loss.update(float(loss.item()), n=int(preds.numel()))
        except Exception:
            pass
        self._last_loss = loss
        return loss

    @torch.no_grad()
    def forward_pass_on_evauluation_step(self, batch: Dict[str, Any]):
        preds, targets = self._extract_preds_targets(batch)
        # keep loss tracker
        try:
            ev_loss = self.criterion(preds, targets)
            self._loss.update(float(ev_loss.item()), n=int(preds.numel()))
        except Exception:
            pass

        # Log & stash metrics
        self._log_and_store_metrics(preds, targets)

        # Optional peek (truncate to 32)
        if getattr(self, '_logger', None):
            self._logger.info(f"REGRESSION_OUTPUT: {preds.detach().flatten().tolist()[:32]}")
            self._logger.info(f"REGRESSION_TARGET: {targets.detach().flatten().tolist()[:32]}")
        return preds

    # ---------- metrics ----------
    @torch.no_grad()
    def _log_and_store_metrics(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.view(-1)
        targets = targets.view(-1)
        N = int(targets.numel())

        # --- Model MAE & MSE + sums for pooled aggregation ---
        abs_err = torch.abs(preds - targets)
        sq_err  = (preds - targets) ** 2
        mae = abs_err.mean().item()
        mse = sq_err.mean().item()
        sum_abs = abs_err.sum().item()
        sum_sq  = sq_err.sum().item()
        self.log_score('MAE', mae)
        self.log_score('MSE', mse)

        # --- Baseline: constant-mean predictor ---
        mean_val = float(self._buffer.get('mean', 0.0))
        means = torch.full_like(targets, mean_val)
        abs_err_b = torch.abs(means - targets)
        sq_err_b  = (means - targets) ** 2
        mae_b = abs_err_b.mean().item()
        mse_b = sq_err_b.mean().item()
        sum_abs_b = abs_err_b.sum().item()
        sum_sq_b  = sq_err_b.sum().item()
        self.log_score('mean/l1_loss', mae_b)
        self.log_score('mean/mse_loss', mse_b)

        # --- Paper Accuracy (configured τ & radius) ---
        tau = float(self._buffer.get('acc/tau', 0.10))
        rad = int(self._buffer.get('acc/rad', 0))
        gt_bin = (targets > _as_tensor_like(tau, targets)).float()
        pr_bin = (preds   > _as_tensor_like(tau, targets)).float()
        if rad > 0:
            gt_bin = _dilate_bin_1d(gt_bin, rad)
            pr_bin = _dilate_bin_1d(pr_bin, rad)
        TP = ((pr_bin == 1) & (gt_bin == 1)).sum().item()
        TN = ((pr_bin == 0) & (gt_bin == 0)).sum().item()
        FP = ((pr_bin == 1) & (gt_bin == 0)).sum().item()
        FN = ((pr_bin == 0) & (gt_bin == 1)).sum().item()
        acc_cfg = (TP + TN) / max(TP + TN + FP + FN, 1e-8) * 100.0
        self.log_score('Accuracy', acc_cfg)

        # --- Accuracy grid at τ=0.10 for radii {0,1,2,5} + counts for pooled aggregation ---
        p_np = preds.detach().cpu().float().numpy()
        t_np = targets.detach().cpu().float().numpy()
        acc_grid = {}
        counts_grid = {}
        for r in (0, 1, 2, 5):
            acc_r, (TP_r, TN_r, FP_r, FN_r) = binary_accuracy_with_neighborhood(p_np, t_np, threshold=0.10, radius=r)
            self.log_score(f'Acc@0.10/r{r}', float(acc_r))
            acc_grid[r] = float(acc_r)
            counts_grid[r] = dict(TP=int(TP_r), TN=int(TN_r), FP=int(FP_r), FN=int(FN_r))

        # --- One-line summary ---
        try:
            msg = (
                f"Predict: MAE={mae:.6f} | MSE={mse:.6f} | "
                f"Baseline(MAE={mae_b:.6f}, MSE={mse_b:.6f}) | "
                f"Acc@0.10[r0,r1,r2,r5]=[{acc_grid[0]:.2f},{acc_grid[1]:.2f},{acc_grid[2]:.2f},{acc_grid[5]:.2f}] | "
                f"N={N}"
            )
            if getattr(self, '_logger', None):
                self._logger.info(msg)
            else:
                print(msg)
        except Exception:
            pass

        # --- Stash for external aggregation (final report) ---
        self._buffer['last_metrics'] = dict(
            N=N,
            MAE=mae, MSE=mse, SUM_ABS=sum_abs, SUM_SQ=sum_sq,
            MAE_BASE=mae_b, MSE_BASE=mse_b, SUM_ABS_BASE=sum_abs_b, SUM_SQ_BASE=sum_sq_b,
            ACC_CFG=acc_cfg, TP=TP, TN=TN, FP=FP, FN=FN,
            ACC_GRID=acc_grid, COUNTS_GRID=counts_grid,
        )

    # expose base helpers (unchanged)
    def log_score(self, name: str, value: float, n: int = 1):
        super().log_score(name, value, n=n)

    def reset_scores(self):
        super().reset_scores()
