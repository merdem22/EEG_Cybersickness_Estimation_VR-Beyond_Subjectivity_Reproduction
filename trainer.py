# trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Iterable, Optional

from torchutils.models import TrainerModel


def _as_tensor_like(x, ref: torch.Tensor) -> torch.Tensor:
    """Convert x to a tensor on the same device/dtype as ref."""
    if isinstance(x, torch.Tensor):
        t = x
        if t.device != ref.device:
            t = t.to(ref.device)
        if t.dtype != ref.dtype:
            t = t.to(ref.dtype)
        return t
    return torch.as_tensor(x, device=ref.device, dtype=ref.dtype)


def _to_vector(t: torch.Tensor, length: int) -> torch.Tensor:
    """
    Coerce tensor t to a contiguous 1D tensor of size==length.
    Common shape fixes: (B,1) -> (B,), scalar -> repeat, etc.
    """
    if t.ndim == 0:
        t = t.view(1)
    if t.ndim == 2 and t.size(1) == 1:
        t = t[:, 0]
    t = t.view(-1)
    if t.numel() == 1 and length != 1:
        t = t.repeat(length)
    if t.numel() != length:
        t = t[:length]
    return t.contiguous()


class MyTrainerModel(TrainerModel):
    """
    Thin wrapper around the registered network that:
      - standardizes batch → (preds, targets)
      - returns a scalar loss on forward (for backprop)
      - aggregates additional metrics for logging
      - provides eval/predict step with metric aggregation
    """

    def __init__(self, train_joy_mean: float, input_type: str, task: str, **kwds):
        # choose which metric-compute fn to use and which scores we intend to log
        if task == 'regression':
            compute_fn = self.compute_regression
            writable_scores = {
                'mse_loss',
                'mean/l1_loss', 'mean/mse_loss',
                'leaky-acc@0.05', 'mean/leaky-acc@0.05',
                'leaky-acc@0.10', 'mean/leaky-acc@0.10',
                'leaky-acc@0.20', 'mean/leaky-acc@0.20',
            }
            joy_mean = float(train_joy_mean)
        elif task == 'classification':
            compute_fn = self.compute_classification
            writable_scores = {
                'accuracy@non-sick', 'accuracy@sick', 'accuracy@sick-5', 'f1_score',
                'mean/accuracy@non-sick', 'mean/accuracy@sick', 'mean/accuracy@sick-5', 'mean/f1_score',
            }
            joy_mean = 0.0
        else:
            raise ValueError(f"Unsupported task: {task}")

        # let the base class build the network/optimizer/etc.
        super().__init__(**kwds, writable_scores=writable_scores)

        # remember a few things for later
        self.task = task
        self.input_type = input_type

        # baseline stats + fn selection
        self._buffer['mean'] = joy_mean
        self._buffer['compute_fn'] = compute_fn

        # expose getter_keys for legacy repo code (not strictly required here)
        self._buffer['getter_keys'] = ('eeg', 'psd', 'kinematic') if input_type == 'multi-segment' else ('kinematic',)

        # track lr across steps if trainer wants it
        try:
            from torchutils.metrics import AverageScore
            self._buffer['lr'] = AverageScore(name='lr')
        except Exception:
            pass

        # if task is regression and criterion is L1/MSE etc., we keep shapes aligned in .forward
        # and do NOT wrap criterion with .flatten() here (we handle shapes explicitly).

    # -------------------------
    # Core batch handling
    # -------------------------
    def model_output(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Forward the underlying network. We pass every item except 'observation'
        as kwargs; networks are written to accept only the keys they need.
        """
        kwargs = {k: v for k, v in batch.items() if k != 'observation'}
        return self.model(**kwargs)

    def _extract_preds_targets(self, batch: Dict[str, Any]) -> (torch.Tensor, torch.Tensor):
        """Get predictions and targets as 1D tensors of length B."""
        preds = self.model_output(batch)  # (B, 1) or (B,)
        if isinstance(preds, torch.Tensor) and preds.ndim > 1 and preds.size(-1) == 1:
            preds = preds.squeeze(-1)

        # Ensure targets are a tensor on same device/dtype and 1D of length B
        targets = batch['observation']
        targets = _as_tensor_like(targets, preds)
        targets = _to_vector(targets, length=preds.shape[0])

        # Force preds to 1D vector as well
        preds = _to_vector(preds, length=preds.shape[0])
        return preds, targets

    # -------------------------
    # Training/Eval entrypoints
    # -------------------------
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        TRAIN step:
          - compute preds/targets
          - compute scalar loss
          - update running loss tracker for epoch logs
          - return loss tensor for backprop
        """
        preds, targets = self._extract_preds_targets(batch)
        loss = self.criterion(preds, targets)

        # Update the running loss aggregator used by the Trainer for epoch logs.
        try:
            self._loss.update(float(loss.item()), n=int(preds.numel()))
        except Exception:
            pass

        # cache for trainer fallbacks (e.g., when forward returns non-tensor in some models)
        self._last_loss = loss
        return loss

    @torch.no_grad()
    def forward_pass_on_evauluation_step(self, batch: Dict[str, Any]):
        """
        EVAL/PREDICT step:
          - compute preds/targets
          - update _loss for validation/predict logs
          - compute and log metrics via the selected compute_fn
          - optionally dump a small sample to logger for debugging
        """
        preds, targets = self._extract_preds_targets(batch)

        # Update validation/predict loss aggregator
        try:
            ev_loss = self.criterion(preds, targets)
            self._loss.update(float(ev_loss.item()), n=int(preds.numel()))
        except Exception:
            pass

        # Metric bookkeeping
        cf = self._buffer.get('compute_fn', None)
        if callable(cf):
            cf(preds, targets)

        # Optional: dump a small slice for visibility
        if getattr(self, 'task', 'regression') == 'regression' and getattr(self, '_logger', None):
            out_list = preds.detach().flatten().tolist()[:32]
            tgt_list = targets.detach().flatten().tolist()[:32]
            self._logger.info(f"REGRESSION_OUTPUT: {out_list}")
            self._logger.info(f"REGRESSION_TARGET: {tgt_list}")

        return preds

    # -------------------------
    # Metric helpers
    # -------------------------
    def log_score(self, name: str, value: float, n: int = 1):
        """Proxy to base-class aggregator."""
        super().log_score(name, value, n=n)

    def reset_scores(self):
        super().reset_scores()

    # -------------------------
    # Metrics for tasks
    # -------------------------
    @torch.no_grad()
    def compute_regression(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Log a few regression metrics:
          - mse_loss on (preds, targets)
          - baseline losses vs constant mean
          - tolerance accuracies at 0.05 / 0.10 / 0.20
        """
        preds = preds.view(-1)
        targets = targets.view(-1)

        # main regression metrics
        self.log_score('mse_loss', F.mse_loss(preds, targets).item())

        mean_val = float(self._buffer.get('mean', 0.0))
        means = torch.full_like(preds, mean_val)
        self.log_score('mean/l1_loss', F.l1_loss(means, targets).item())
        self.log_score('mean/mse_loss', F.mse_loss(means, targets).item())

        # tolerance accuracies
        err = (preds - targets).abs()
        err_mean = (means - targets).abs()
        for tau in (0.05, 0.10, 0.20):
            self.log_score(f'leaky-acc@{tau:.2f}', (err <= tau).float().mean().item())
            self.log_score(f'mean/leaky-acc@{tau:.2f}', (err_mean <= tau).float().mean().item())

    @torch.no_grad()
    def compute_classification(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Example classification metrics (if you ever switch tasks):
          - per-class accuracy variants and F1.
        Assumes preds are logits or probabilities for binary classification.
        """
        preds = preds.view(-1)
        targets = targets.view(-1).float()

        # If preds look like logits, sigmoid them
        if preds.min() < 0 or preds.max() > 1:
            probs = preds.sigmoid()
        else:
            probs = preds.clamp(0, 1)

        y_hat = (probs >= 0.5).float()
        y = targets

        # avoid 0/0 with eps
        eps = 1e-8
        TP = (y_hat * y).sum().item()
        TN = ((1 - y_hat) * (1 - y)).sum().item()
        FP = (y_hat * (1 - y)).sum().item()
        FN = ((1 - y_hat) * y).sum().item()

        acc_sick = TP / max(TP + FN, eps)
        acc_non = TN / max(TN + FP, eps)
        prec = TP / max(TP + FP, eps)
        rec = TP / max(TP + FN, eps)
        f1 = (2 * prec * rec) / max(prec + rec, eps)

        self.log_score('accuracy@sick', float(acc_sick))
        self.log_score('accuracy@non-sick', float(acc_non))
        self.log_score('f1_score', float(f1))

        # (Optional) comparison to mean predictor (always-majority class)
        p_mean = (y.mean().item() >= 0.5)  # predict-1 if majority positives
        y_mean = torch.full_like(y, float(p_mean))
        TPm = (y_mean * y).sum().item()
        TNm = ((1 - y_mean) * (1 - y)).sum().item()
        FPm = (y_mean * (1 - y)).sum().item()
        FNm = ((1 - y_mean) * y).sum().item()
        acc_sick_m = TPm / max(TPm + FNm, eps)
        acc_non_m = TNm / max(TNm + FPm, eps)
        prec_m = TPm / max(TPm + FPm, eps)
        rec_m = TPm / max(TPm + FNm, eps)
        f1_m = (2 * prec_m * rec_m) / max(prec_m + rec_m, eps)

        self.log_score('mean/accuracy@sick', float(acc_sick_m))
        self.log_score('mean/accuracy@non-sick', float(acc_non_m))
        self.log_score('mean/f1_score', float(f1_m))
