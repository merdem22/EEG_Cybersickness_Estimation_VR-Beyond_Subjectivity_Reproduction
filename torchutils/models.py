
from __future__ import annotations
from typing import Any, Dict, Optional
import torch
from .metrics import AverageScore

class TrainerModel:
    """Lightweight base wrapper expected by the training loop.

    Subclasses are expected to implement:
      - forward(self, batch): returns a loss tensor or predictions; should update self._loss as desired.
      - forward_pass_on_evauluation_step(self, batch): same as forward but without gradient, for eval/predict.
      - writable_scores(self): returns a set/list of score names that callbacks may log.

    This base also provides:
      - self._buffer: a generic dict to stash anything (e.g., compute_fn, score names, etc.).
      - self._loss: AverageScore tracker for loss values (float). For backprop, also return a loss tensor.
      - self.log(msg): hook for Trainer logging.
    """
    def __init__(self):
        self._net = None
        self._optimizer = None
        self._device = torch.device('cpu')
        self._logger = None
        self._loss = AverageScore('loss')
        self._buffer: Dict[str, Any] = {}

    # ----- Wiring -----
    def set_network(self, net: torch.nn.Module):
        self._net = net

    def set_optimizer(self, opt: torch.optim.Optimizer):
        self._optimizer = opt

    def set_device(self, device: torch.device):
        self._device = device
        if self._net is not None:
            self._net.to(device)

    def set_logger(self, logger):
        self._logger = logger

    # ----- Convenience -----
    def net(self) -> torch.nn.Module:
        return self._net

    def optimizer(self) -> Optional[torch.optim.Optimizer]:
        return self._optimizer

    def device(self) -> torch.device:
        return self._device

    def log(self, msg: str, level: int = 20):
        if self._logger is not None:
            self._logger.log(level, msg)

    # Subclasses should override these.
    def forward(self, batch: Dict[str, Any]):
        raise NotImplementedError

    def forward_pass_on_evauluation_step(self, batch: Dict[str, Any]):
        # By default, reuse forward() without gradients.
        return self.forward(batch)

    def writable_scores(self):
        # Provide anything stored by subclasses in _buffer['writable_scores']
        return self._buffer.get('writable_scores', {'loss'})
