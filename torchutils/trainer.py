
from __future__ import annotations
import logging
from typing import Optional, Dict, Any, List, Union
import torch
from torch.utils.data import DataLoader
from .callbacks import Callback

class Trainer:
    """Minimal training/eval loop compatible with the repo style."""
    def __init__(self, model, net: Optional[torch.nn.Module] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 logger: Optional[logging.Logger] = None):
        self.model = model
        self.logger = logger or logging.getLogger('torchutils.trainer')
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        if net is not None:
            self.model.set_network(net)
        if optimizer is not None:
            self.model.set_optimizer(optimizer)
        self.model.set_device(self.device)
        self.model.set_logger(self.logger)

    def compile(self, handlers: Optional[List[logging.Handler]] = None):
        if handlers:
            for h in handlers:
                if h not in self.logger.handlers:
                    self.logger.addHandler(h)

    def _make_loader(self, dataset, dataloader_kwargs: Optional[Dict[str, Any]]):
        if dataloader_kwargs is None:
            dataloader_kwargs = {}
        dataloader_kwargs.setdefault('batch_size', 32)
        dataloader_kwargs.setdefault('shuffle', True)
        return DataLoader(dataset, **dataloader_kwargs)

    def _compute_loss_from_output(self, output, batch) -> Optional[torch.Tensor]:
        if isinstance(output, torch.Tensor):
            if output.ndim == 0 or output.size() == torch.Size([]):
                return output
        if isinstance(output, dict) and 'loss' in output and isinstance(output['loss'], torch.Tensor):
            return output['loss']
        if isinstance(output, (list, tuple)) and 'observation' in batch and hasattr(self.model, 'criterion'):
            try:
                losses = []
                for pred, targ in zip(output, batch['observation']):
                    losses.append(self.model.criterion(pred, targ))
                if losses:
                    return torch.stack(losses).mean()
            except Exception:
                return None
        return None

    def fit(self, train_dataset, valid_dataset=None, num_epochs: int = 1,
            callbacks: Optional[List[Callback]] = None,
            dataloader_kwargs: Optional[Dict[str, Any]] = None,
            val_dataloader_kwargs: Optional[Dict[str, Any]] = None):
        callbacks = callbacks or []
        train_loader = self._make_loader(train_dataset, dataloader_kwargs)
        val_loader = self._make_loader(valid_dataset, val_dataloader_kwargs) if valid_dataset is not None else None
        model = self.model
        net = model.net()
        opt = model.optimizer()
        sch = getattr(model, 'scheduler', lambda: None)()

        for epoch in range(1, num_epochs + 1):
            model._loss.reset()
            if net is not None:
                net.train()

            for step, batch in enumerate(train_loader):
                try:
                    batch = {k: (v.to(self.device) if hasattr(v, 'to') else v) for k, v in batch.items()}
                except Exception:
                    pass
                out = model.forward(batch)
                loss = self._compute_loss_from_output(out, batch)
                if loss is None:
                    loss = getattr(model, '_last_loss', None)
                if loss is None:
                    raise RuntimeError("Model.forward didn't return a loss tensor and no fallback was possible.")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            train_logs = {'loss': model._loss.avg}

            val_logs = {}
            val_loss_value = None
            if val_loader is not None:
                if net is not None:
                    net.eval()
                model._loss.reset()
                with torch.no_grad():
                    for vbatch in val_loader:
                        try:
                            vbatch = {k: (v.to(self.device) if hasattr(v, 'to') else v) for k, v in vbatch.items()}
                        except Exception:
                            pass
                        if hasattr(model, 'forward_pass_on_evauluation_step'):
                            _ = model.forward_pass_on_evauluation_step(vbatch)
                        else:
                            _ = model.forward(vbatch)
                val_loss_value = model._loss.avg
                val_logs['val_loss'] = val_loss_value

            # Step scheduler
            if sch is not None:
                try:
                    import inspect
                    if 'metrics' in inspect.signature(sch.step).parameters:
                        sch.step(val_loss_value if val_loss_value is not None else model._loss.avg)
                    else:
                        sch.step()
                except Exception:
                    pass

            logs = {**train_logs, **val_logs}
            stop = False
            for cb in callbacks:
                if hasattr(cb, 'on_epoch_end'):
                    res = cb.on_epoch_end(epoch, logs=logs, logger=self.logger, trainer=self)
                    if res is True:
                        stop = True
            if stop:
                break

    def predict(self, dataset, callbacks: Optional[List[Callback]] = None,
                dataloader_kwargs: Optional[Dict[str, Any]] = None):
        callbacks = callbacks or []
        loader = self._make_loader(dataset, dataloader_kwargs or {'batch_size': 64, 'shuffle': False})
        model = self.model
        net = model.net()
        if net is not None:
            net.eval()

        preds = []
        with torch.no_grad():
            for batch in loader:
                try:
                    batch = {k: (v.to(self.device) if hasattr(v, 'to') else v) for k, v in batch.items()}
                except Exception:
                    pass
                if hasattr(model, 'forward_pass_on_evauluation_step'):
                    out = model.forward_pass_on_evauluation_step(batch)
                else:
                    out = model.forward(batch)
                preds.append(out)

        logs = {'loss': model._loss.avg}
        for cb in callbacks:
            if hasattr(cb, 'on_predict_end'):
                cb.on_predict_end(logs=logs, logger=self.logger)
        return preds
