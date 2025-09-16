
from __future__ import annotations
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
from torchvision.models import get_model
from .metrics import AverageScore

_OPTIMIZERS = {
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD,
    'AdamW': torch.optim.AdamW,
    'RMSprop': torch.optim.RMSprop,
}

_LOSSES = {
    'l1_loss': F.l1_loss,
    'mse_loss': F.mse_loss,
    'binary_cross_entropy': F.binary_cross_entropy,
    'bce': F.binary_cross_entropy,
    'cross_entropy': F.cross_entropy,
}

_SCHEDULERS = {
    'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'StepLR': torch.optim.lr_scheduler.StepLR,
    'MultiStepLR': torch.optim.lr_scheduler.MultiStepLR,
    'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR,
}

class TrainerModel:
    """Base wrapper expected by the training loop.

    Subclasses typically call super().__init__(**kwds, writable_scores={...}).
    Accepts many kwargs but only uses a few common ones; the rest are stored
    in self._buffer for subclass access.

    Known kwargs:
      - model (str): name registered via torchvision.register_model (e.g., 'power-spectral-no-kinematic-model')
      - n_channels, hidden_size, num_classes: forwarded to get_model(...)
      - optimizer (str), lr (float), weight_decay (float)
      - criterion (str): one of _LOSSES keys
      - scheduler (str) + optional scheduler_* params
      - device (torch.device or str)
      - writable_scores (set/list of str)
    """
    def __init__(self, **kwargs):
        self._net = None
        self._optimizer = None
        self._scheduler = None
        self._device = torch.device('cpu')
        self._logger = None
        self._loss = AverageScore('loss')
        self._buffer: Dict[str, Any] = {}

        # 1) Device
        device = kwargs.pop('device', None)
        if device is not None:
            self.set_device(torch.device(device) if isinstance(device, str) else device)

        # 2) Network
        model_name = kwargs.pop('model', None)
        n_channels = kwargs.pop('n_channels', None)
        hidden_size = kwargs.pop('hidden_size', None)
        num_classes = kwargs.pop('num_classes', None)
        if model_name is not None:
            # Forward typical init kwargs; ignore if None
            model_kwargs = {}
            if n_channels is not None: model_kwargs['n_channels'] = n_channels
            if hidden_size is not None: model_kwargs['hidden_size'] = hidden_size
            if num_classes is not None: model_kwargs['num_classes'] = num_classes
            try:
                net = get_model(model_name, **model_kwargs)
            except Exception:
                # Fallback: user may have passed a torch.nn.Module directly via 'model_obj'
                net = kwargs.pop('model_obj', None)
            if net is not None:
                self.set_network(net)
                if self._device is not None:
                    self._net.to(self._device)

        # 3) Optimizer
        opt_name = kwargs.pop('optimizer', None)
        lr = float(kwargs.pop('lr', 1e-3))
        weight_decay = float(kwargs.pop('weight_decay', 0.0))
        if opt_name and self._net is not None:
            opt_cls = _OPTIMIZERS.get(opt_name, None)
            if opt_cls is None:
                raise ValueError(f"Unknown optimizer '{opt_name}'. Available: {list(_OPTIMIZERS.keys())}")
            self.set_optimizer(opt_cls(self._net.parameters(), lr=lr, weight_decay=weight_decay))

        # 4) Criterion
        crit_name = kwargs.pop('criterion', None)
        if isinstance(crit_name, str):
            if crit_name not in _LOSSES:
                raise ValueError(f"Unknown criterion '{crit_name}'. Available: {list(_LOSSES.keys())}")
            self.criterion = _LOSSES[crit_name]
            self.criterion_name = crit_name
        else:
            # Allow subclass to set criterion later
            self.criterion = getattr(self, 'criterion', lambda x, y: F.mse_loss(x, y))
            self.criterion_name = getattr(self, 'criterion_name', 'mse_loss')

        # 5) Scheduler (optional)
        sch_name = kwargs.pop('scheduler', None)
        if sch_name and self._optimizer is not None:
            sch_cls = _SCHEDULERS.get(sch_name, None)
            if sch_cls is None:
                raise ValueError(f"Unknown scheduler '{sch_name}'. Available: {list(_SCHEDULERS.keys())}")
            # Common kwargs
            if sch_name == 'ReduceLROnPlateau':
                patience = int(kwargs.pop('scheduler_patience', 5))
                factor = float(kwargs.pop('scheduler_factor', 0.5))
                self._scheduler = sch_cls(self._optimizer, patience=patience, factor=factor)
            elif sch_name == 'StepLR':
                step_size = int(kwargs.pop('scheduler_step_size', 10))
                gamma = float(kwargs.pop('scheduler_gamma', 0.1))
                self._scheduler = sch_cls(self._optimizer, step_size=step_size, gamma=gamma)
            elif sch_name == 'MultiStepLR':
                milestones = kwargs.pop('scheduler_milestones', [30, 60])
                gamma = float(kwargs.pop('scheduler_gamma', 0.1))
                self._scheduler = sch_cls(self._optimizer, milestones=milestones, gamma=gamma)
            elif sch_name == 'ExponentialLR':
                gamma = float(kwargs.pop('scheduler_gamma', 0.95))
                self._scheduler = sch_cls(self._optimizer, gamma=gamma)

        # 6) Writable scores (for callbacks)
        writable_scores = kwargs.pop('writable_scores', None)
        if writable_scores is not None:
            self._buffer['writable_scores'] = set(writable_scores)

        # 7) Stash any remaining kwargs in buffer so subclasses can read them
        for k, v in kwargs.items():
            self._buffer[k] = v

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

    def scheduler(self):
        return self._scheduler

    def device(self) -> torch.device:
        return self._device

    def log(self, msg: str, level: int = 20):
        if self._logger is not None:
            self._logger.log(level, msg)

    # Subclasses should override these.
    def forward(self, batch: Dict[str, Any]):
        raise NotImplementedError

    def forward_pass_on_evauluation_step(self, batch: Dict[str, Any]):
        return self.forward(batch)

    def writable_scores(self):
        return self._buffer.get('writable_scores', {'loss'})
