# trainer.py — article metrics only: MAE, MSE, Accuracy
import torch
import torch.nn.functional as F
from typing import Dict, Any
from torchutils.models import TrainerModel

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
    Wrapper around the registered network that returns a scalar loss during training
    and logs exactly the paper’s metrics on eval/predict:
      - MAE
      - MSE
      - Accuracy (binary above threshold with optional neighborhood dilation)
    """
    def __init__(self, train_joy_mean: float, input_type: str, task: str, **kwds):
        if task != 'regression':
            raise ValueError("This runner is configured for the paper's regression setup only.")

        # Metric knobs (defaults: threshold=0.10, neighborhood radius=0)
        acc_tau = float(kwds.pop('acc_threshold', 0.10))
        acc_rad = int(kwds.pop('acc_neighborhood', 0))

        writable_scores = {'MAE', 'MSE', 'Accuracy'}
        super().__init__(**kwds, writable_scores=writable_scores)

        self.task = task
        self.input_type = input_type
        self._buffer['mean'] = float(train_joy_mean)    # not used in metrics, kept for completeness
        self._buffer['acc/tau'] = acc_tau               # binary threshold (e.g., 0.10 ≈ “1 degree” if scaled)
        self._buffer['acc/rad'] = acc_rad               # dilation radius in windows (0 = none)

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
        # Update loss tracker so epoch/predict logs have a number
        try:
            ev_loss = self.criterion(preds, targets)
            self._loss.update(float(ev_loss.item()), n=int(preds.numel()))
        except Exception:
            pass

        # Log exactly the paper metrics
        self._log_paper_metrics(preds, targets)

        # Optional peeks
        if getattr(self, '_logger', None):
            self._logger.info(f"REGRESSION_OUTPUT: {preds.detach().flatten().tolist()[:32]}")
            self._logger.info(f"REGRESSION_TARGET: {targets.detach().flatten().tolist()[:32]}")
        return preds

    # ---------- metrics ----------
    @torch.no_grad()
    def _log_paper_metrics(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.view(-1); targets = targets.view(-1)
        # MAE, MSE
        mae = torch.mean(torch.abs(preds - targets)).item()
        mse = F.mse_loss(preds, targets).item()
        # Accuracy: binarize and optional neighborhood dilation
        tau = float(self._buffer.get('acc/tau', 0.10))
        rad = int(self._buffer.get('acc/rad', 0))
        gt_bin = (targets > tau).float()
        pr_bin = (preds   > tau).float()
        if rad > 0:
            gt_bin = _dilate_bin_1d(gt_bin, rad)
            pr_bin = _dilate_bin_1d(pr_bin, rad)
        TP = ((pr_bin == 1) & (gt_bin == 1)).sum().item()
        TN = ((pr_bin == 0) & (gt_bin == 0)).sum().item()
        FP = ((pr_bin == 1) & (gt_bin == 0)).sum().item()
        FN = ((pr_bin == 0) & (gt_bin == 1)).sum().item()
        acc = (TP + TN) / max(TP + TN + FP + FN, 1e-8) * 100.0

        self.log_score('MAE', mae)
        self.log_score('MSE', mse)
        self.log_score('Accuracy', acc)

    # expose base helpers (unchanged)
    def log_score(self, name: str, value: float, n: int = 1):
        super().log_score(name, value, n=n)

    def reset_scores(self):
        super().reset_scores()
