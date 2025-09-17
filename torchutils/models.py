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
    def __init__(self, **kwargs):
        self._net = None
        self.model = None
        self._optimizer = None
        self.optimizer = None
        self._scheduler = None
        self._device = torch.device('cpu')
        self.device_attr = self._device
        self._logger = None
        self._loss = AverageScore('loss')
        self._buffer: Dict[str, Any] = {}

        device = kwargs.pop('device', None)
        if device is not None:
            self.set_device(torch.device(device) if isinstance(device, str) else device)

        model_name = kwargs.pop('model', None)
        n_channels = kwargs.pop('n_channels', None)
        hidden_size = kwargs.pop('hidden_size', None)
        num_classes = kwargs.pop('num_classes', None)
        if model_name is not None:
            model_kwargs = {}
            if n_channels is not None: model_kwargs['n_channels'] = n_channels
            if hidden_size is not None: model_kwargs['hidden_size'] = hidden_size
            if num_classes is not None: model_kwargs['num_classes'] = num_classes
            try:
                net = get_model(model_name, **model_kwargs)
            except Exception:
                net = kwargs.pop('model_obj', None)
            if net is not None:
                self.set_network(net)
                if self._device is not None:
                    self._net.to(self._device)

        opt_name = kwargs.pop('optimizer', None)
        lr = float(kwargs.pop('lr', 1e-3))
        weight_decay = float(kwargs.pop('weight_decay', 0.0))
        if opt_name and self._net is not None:
            opt_cls = _OPTIMIZERS.get(opt_name, None)
            if opt_cls is None:
                raise ValueError(f"Unknown optimizer '{opt_name}'. Available: {list(_OPTIMIZERS.keys())}")
            self.set_optimizer(opt_cls(self._net.parameters(), lr=lr, weight_decay=weight_decay))

        crit_name = kwargs.pop('criterion', None)
        if isinstance(crit_name, str):
            if crit_name not in _LOSSES:
                raise ValueError(f"Unknown criterion '{crit_name}'. Available: {list(_LOSSES.keys())}")
            self.criterion = _LOSSES[crit_name]
            self.criterion_name = crit_name
        else:
            self.criterion = getattr(self, 'criterion', lambda x, y: F.mse_loss(x, y))
            self.criterion_name = getattr(self, 'criterion_name', 'mse_loss')

        sch_name = kwargs.pop('scheduler', None)
        if sch_name and self._optimizer is not None:
            sch_cls = _SCHEDULERS.get(sch_name, None)
            if sch_cls is None:
                raise ValueError(f"Unknown scheduler '{sch_name}'. Available: {list(_SCHEDULERS.keys())}")
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

        writable_scores = kwargs.pop('writable_scores', None)
        if writable_scores is not None:
            self._buffer['writable_scores'] = set(writable_scores)

        for k, v in kwargs.items():
            self._buffer[k] = v

    def set_network(self, net: torch.nn.Module):
        self._net = net
        self.model = net

    def set_optimizer(self, opt: torch.optim.Optimizer):
        self._optimizer = opt
        self.optimizer = opt

    def set_device(self, device: torch.device):
        self._device = device
        self.device_attr = device
        if self._net is not None:
            self._net.to(device)

    def set_logger(self, logger):
        self._logger = logger

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

    def forward(self, batch: Dict[str, Any]):
        raise NotImplementedError

    def forward_pass_on_evauluation_step(self, batch: Dict[str, Any]):
        return self.forward(batch)

    # metrics aggregation
    def _ensure_scores_dict(self):
        if '_scores' not in self._buffer or not isinstance(self._buffer.get('_scores'), dict):
            self._buffer['_scores'] = {}

    def log_score(self, name: str, value: float, n: int = 1):
        self._ensure_scores_dict()
        scores = self._buffer['_scores']
        if name not in scores:
            scores[name] = AverageScore(name=name)
        try:
            scores[name].update(float(value), n=n)
        except Exception:
            pass

    def reset_scores(self):
        self._buffer['_scores'] = {}

    def get_logged_scores(self):
        self._ensure_scores_dict()
        out = {}
        for k, v in self._buffer['_scores'].items():
            try:
                out[k] = float(v.avg)
            except Exception:
                continue
        return out

    # torch-style delegation
    def state_dict(self, *args, **kwargs):
        if self._net is None:
            raise RuntimeError("TrainerModel has no underlying network set; cannot export state_dict().")
        return self._net.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        if self._net is None:
            raise RuntimeError("TrainerModel has no underlying network set; cannot load_state_dict().")
        return self._net.load_state_dict(state_dict, strict=strict)

    def train(self, mode: bool = True):
        if self._net is not None:
            self._net.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, device):
        self.set_device(device if isinstance(device, torch.device) else torch.device(device))
        return self

    def writable_scores(self):
        return self._buffer.get('writable_scores', {'loss'})