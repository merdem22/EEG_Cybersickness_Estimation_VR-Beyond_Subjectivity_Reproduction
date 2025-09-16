
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
        items = [(k, logs[k]) for k in (self.score_names or logs.keys()) if isinstance(logs.get(k), (int, float, float))]
        if not items:
            return
        msg = " | ".join([f"{k}: {v:.6f}" for k, v in items])
        self._emit(logger, f"Epoch {epoch}: {msg}")

    def on_predict_end(self, logs: Optional[Dict[str, Any]] = None, logger=None, **kwargs):
        if not logs:
            return
        items = [(k, logs[k]) for k in (self.score_names or logs.keys()) if isinstance(logs.get(k), (int, float, float))]
        if not items:
            return
        msg = " | ".join([f"{k}: {v:.6f}" for k, v in items])
        self._emit(logger, f"Predict: {msg}")

class EarlyStopping(Callback):
    def __init__(self, monitor: str = 'val_loss', mode: str = 'min', patience: int = 5, min_delta: float = 0.0):
        assert mode in {'min','max'}
        self.monitor = monitor
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
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

    def on_epoch_end(self, epoch: int, logs=None, trainer=None, **kwargs):
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
            if trainer and getattr(trainer, 'logger', None):
                trainer.logger.info(f"Early stopping at epoch {epoch} (best {self.monitor}={self.best:.6f})")
            return True
        return False
