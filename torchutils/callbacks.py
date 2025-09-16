
from __future__ import annotations
from typing import Dict, Any, Iterable, Optional

class Callback:
    """Base callback with Keras-like hooks."""
    def on_train_begin(self, **kwargs): ...
    def on_train_end(self, **kwargs): ...
    def on_epoch_begin(self, epoch: int, **kwargs): ...
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None, **kwargs): ...
    def on_batch_end(self, step: int, logs: Optional[Dict[str, Any]] = None, **kwargs): ...
    def on_predict_end(self, logs: Optional[Dict[str, Any]] = None, **kwargs): ...

class AverageScoreLogger(Callback):
    """Logs provided score keys from `logs` at a given level.

    Use as: AverageScoreLogger(*model.writable_scores(), level=20)
    """
    def __init__(self, *score_names: str, level: int = 20):
        self.score_names = list(score_names) if score_names else []
        self.level = level

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None, logger=None, **kwargs):
        if logs is None:
            return
        if logger is None:
            # Fallback: print if no logger is supplied
            def _emit(msg): print(msg)
        else:
            def _emit(msg): logger.log(self.level, msg)

        # If specific scores requested, filter; else show everything present.
        items = [(k, logs[k]) for k in self.score_names if k in logs] if self.score_names else list(logs.items())
        if not items:
            return
        msg = " | ".join([f"{k}: {v:.6f}" for k, v in items if isinstance(v, (int, float))])
        if msg:
            _emit(f"Epoch {epoch}: {msg}")

    def on_predict_end(self, logs: Optional[Dict[str, Any]] = None, logger=None, **kwargs):
        # Optionally log after predict/eval
        if not logs:
            return
        items = [(k, logs[k]) for k in self.score_names if k in logs] if self.score_names else list(logs.items())
        if logger is None:
            def _emit(msg): print(msg)
        else:
            def _emit(msg): logger.log(self.level, msg)
        msg = " | ".join([f"{k}: {v:.6f}" for k, v in items if isinstance(v, (int, float))])
        if msg:
            _emit(f"Predict: {msg}")

class EarlyStopping(Callback):
    """Simple early stopping on a monitored metric in `logs`.

    Parameters
    ----------
    monitor : str
        Key to monitor in logs (default: 'val_loss').
    mode : {'min','max'}
        Whether lower is better ('min') or higher is better ('max').
    patience : int
        Number of epochs with no improvement before stopping.
    min_delta : float
        Minimum change to qualify as an improvement.
    """
    def __init__(self, monitor: str = 'val_loss', mode: str = 'min', patience: int = 5, min_delta: float = 0.0):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = float(min_delta)
        self.best = None
        self.wait = 0
        self.stopped_epoch = None

    def _is_improvement(self, current, best):
        if best is None:
            return True
        if self.mode == 'min':
            return (best - current) > self.min_delta
        else:
            return (current - best) > self.min_delta

    def on_epoch_end(self, epoch: int, logs=None, trainer=None, **kwargs):
        if logs is None or self.monitor not in logs:
            return False  # nothing to do
        current = logs[self.monitor]
        if self._is_improvement(current, self.best):
            self.best = current
            self.wait = 0
            return False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if trainer is not None and getattr(trainer, 'logger', None):
                    trainer.logger.info(f"Early stopping at epoch {epoch} (best {self.monitor}={self.best:.6f})")
                # Signal to trainer to stop
                return True
            return False
