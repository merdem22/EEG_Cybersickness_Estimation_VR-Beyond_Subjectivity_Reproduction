
from __future__ import annotations
from typing import Dict, Any, Optional

class Callback:
    def on_train_begin(self, **kwargs): ...
    def on_train_end(self, **kwargs): ...
    def on_epoch_begin(self, epoch: int, **kwargs): ...
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None, **kwargs): ...
    def on_batch_end(self, step: int, logs: Optional[Dict[str, Any]] = None, **kwargs): ...
    def on_predict_end(self, logs: Optional[Dict[str, Any]] = None, **kwargs): ...

class AverageScoreLogger(Callback):
    """Logs selected keys from logs (or all numeric keys if none provided)."""
    def __init__(self, *score_names: str, level: int = 20):
        # Accepts varargs of score names + optional 'level' kw (e.g., AverageScoreLogger('l1_loss','lr', level=20))
        self.score_names = list(score_names)
        self.level = level

    def _emit(self, logger, msg):
        if logger is None:
            print(msg)
        else:
            logger.log(self.level, msg)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None, logger=None, **kwargs):
        if not logs:
            return
        keys = self.score_names or [k for k, v in logs.items() if isinstance(v, (int, float))]
        items = [(k, logs[k]) for k in keys if k in logs and isinstance(logs[k], (int, float))]
        if not items:
            return
        msg = " | ".join([f"{k}: {v:.6f}" for k, v in items])
        self._emit(logger, f"Epoch {epoch}: {msg}")

    def on_predict_end(self, logs: Optional[Dict[str, Any]] = None, logger=None, **kwargs):
        if not logs:
            return
        keys = self.score_names or [k for k, v in logs.items() if isinstance(v, (int, float))]
        items = [(k, logs[k]) for k in keys if k in logs and isinstance(logs[k], (int, float))]
        if not items:
            return
        msg = " | ".join([f"{k}: {v:.6f}" for k, v in items])
        self._emit(logger, f"Predict: {msg}")

class EarlyStopping(Callback):
    """Early stopping with flexible args.

    Compatible with either:
      EarlyStopping(monitor='val_loss', mode='min', patience=5, min_delta=0.0)
      EarlyStopping(monitor='l1_loss', goal='minimize', patience=15, delta=1e-4, verbose=30)

    Parameters
    ----------
    monitor : str
        Key in `logs` to monitor.
    mode : {'min','max'}, optional
        Direction for improvement. If omitted and `goal` is provided, inferred from it.
    goal : {'minimize','maximize'}, optional
        Alternative to `mode`; strings starting with 'min' or 'max' are accepted.
    patience : int
        Epochs to wait for improvement before stopping.
    min_delta / delta : float
        Minimum change to qualify as an improvement.
    verbose : int
        Logging level to use when printing the stop message (e.g., 30 for WARNING).
    """
    def __init__(self, monitor: str = 'val_loss', mode: str = None, goal: str = None,
                 patience: int = 5, min_delta: float = None, delta: float = None,
                 verbose: int = 0):
        self.monitor = monitor
        # Normalize mode/goal
        if goal is not None and mode is None:
            mode = 'min' if str(goal).lower().startswith('min') else 'max'
        if mode is None:
            mode = 'min'
        if mode not in {'min', 'max'}:
            raise ValueError("EarlyStopping: mode must be 'min' or 'max'")
        self.mode = mode
        # Normalize delta/min_delta
        if min_delta is None and delta is not None:
            min_delta = float(delta)
        self.min_delta = 0.0 if min_delta is None else float(min_delta)
        self.patience = int(patience)
        self.best = None
        self.wait = 0
        self.stopped_epoch = None
        # Use provided verbose as logging level; default to INFO(20) if 0
        self._log_level = int(verbose) if verbose and int(verbose) > 0 else 20

    def on_train_begin(self, **kwargs):
        self.best = None
        self.wait = 0
        self.stopped_epoch = None

    def _improved(self, current, best):
        if best is None:
            return True
        if self.mode == 'min':
            return (best - current) > self.min_delta
        else:
            return (current - best) > self.min_delta

    def on_epoch_end(self, epoch: int, logs=None, trainer=None, logger=None, **kwargs):
        if not logs or self.monitor not in logs:
            return False
        current = logs[self.monitor]
        if self._improved(current, self.best):
            self.best = current
            self.wait = 0
            return False
        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            if logger is not None:
                logger.log(self._log_level, f"Early stopping at epoch {epoch} "
                                            f"(best {self.monitor}={self.best:.6f})")
            return True
        return False
