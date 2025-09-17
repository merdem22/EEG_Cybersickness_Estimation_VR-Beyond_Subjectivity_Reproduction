
from __future__ import annotations
import logging
from typing import Optional, Dict, Any, List, Union
import torch
from torch.utils.data import DataLoader
from .callbacks import Callback

class Trainer:
    """Minimal training/eval loop compatible with the repo style.

    Accepts optional train/valid datasets at construction time; you can then call
    `trainer.train(...)` (periodic validation) or `trainer.fit(...)` (validate every epoch).
    """
    def __init__(self, model,
                 net: Optional[torch.nn.Module] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 logger: Optional[logging.Logger] = None,
                 train_dataset=None,
                 valid_dataset=None,
                 train_dataloader_kwargs: Optional[Dict[str, Any]] = None,
                 valid_dataloader_kwargs: Optional[Dict[str, Any]] = None,
                 callbacks: Optional[List[Callback]] = None,
                 num_epochs: Optional[int] = None):
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
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        self._train_loader_kwargs = train_dataloader_kwargs or {}
        self._valid_loader_kwargs = valid_dataloader_kwargs or {}
        self._default_callbacks = callbacks or []
        self._default_epochs = num_epochs

    def compile(self, handlers: Optional[List[logging.Handler]] = None):
        if handlers:
            for h in handlers:
                if h not in self.logger.handlers:
                    self.logger.addHandler(h)


    def _get_net(self):
        """Return the underlying nn.Module whether exposed as method or attribute."""
        m = getattr(self.model, 'net', None)
        if callable(m):
            try:
                val = m()
                if val is not None:
                    return val
            except Exception:
                pass
        # fallbacks: common aliases
        val = getattr(self.model, 'model', None)
        if val is not None:
            return val
        return getattr(self.model, '_net', None)

    def _get_optimizer(self):
        """Return the optimizer whether exposed as method or attribute."""
        opt = getattr(self.model, 'optimizer', None)
        if callable(opt):
            try:
                val = opt()
                if val is not None:
                    return val
            except Exception:
                pass
        if opt is not None:
            return opt
        return getattr(self.model, '_optimizer', None)

    def _make_loader(self, dataset, dataloader_kwargs: Optional[Dict[str, Any]]):
        if dataset is None:
            return None
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

    def _add_common_logs(self, logs: Dict[str, Any]):
        # duplicate training loss under the exact criterion_name key (e.g., 'l1_loss')
        crit = getattr(self.model, 'criterion_name', None)
        if crit and 'loss' in logs and crit not in logs:
            logs[crit] = logs['loss']
        # record LR for logging/monitoring if available
        try:
            opt = self.model.optimizer()
            if opt is not None and opt.param_groups:
                lr = float(opt.param_groups[0].get('lr', 0.0))
                logs['lr'] = lr
                # also update model buffer tracker if present
                if hasattr(self.model, '_buffer') and isinstance(self.model._buffer.get('lr', None), object):
                    try:
                        self.model._buffer['lr'].update(lr)
                    except Exception:
                        pass
        except Exception:
            pass
        return logs

    def fit(self, train_dataset=None, valid_dataset=None, num_epochs: Optional[int] = None,
            callbacks: Optional[List[Callback]] = None,
            dataloader_kwargs: Optional[Dict[str, Any]] = None,
            val_dataloader_kwargs: Optional[Dict[str, Any]] = None):
        if train_dataset is None:
            train_dataset = self._train_dataset
        if valid_dataset is None:
            valid_dataset = self._valid_dataset
        if dataloader_kwargs is None:
            dataloader_kwargs = self._train_loader_kwargs
        if val_dataloader_kwargs is None:
            val_dataloader_kwargs = self._valid_loader_kwargs
        if callbacks is None:
            callbacks = list(self._default_callbacks)
        if num_epochs is None:
            num_epochs = self._default_epochs or 1

        callbacks = callbacks or []
        train_loader = self._make_loader(train_dataset, dataloader_kwargs)
        val_loader = self._make_loader(valid_dataset, val_dataloader_kwargs)

        model = self.model
        net = self._get_net()
        opt = self._get_optimizer()
        sch = getattr(model, 'scheduler', lambda: None)()

        for epoch in range(1, num_epochs + 1):
            model._loss.reset()
            if hasattr(model, 'reset_scores'):
                model.reset_scores()
            if net is not None:
                net.train()

            if train_loader is None:
                raise RuntimeError("No training dataset provided to Trainer.fit and none stored in the Trainer.")

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

            logs = {'loss': model._loss.avg}
            self._add_common_logs(logs)
            if hasattr(model, 'get_logged_scores'):
                logs.update(model.get_logged_scores())
            if hasattr(model, 'get_logged_scores'):
                logs.update(model.get_logged_scores())

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
                logs['val_loss'] = val_loss_value
                # also expose 'val/<criterion>'
                crit = getattr(model, 'criterion_name', None)
                if crit:
                    logs[f'val/{crit}'] = val_loss_value

            # Step scheduler if present
            if sch is not None:
                try:
                    import inspect
                    if 'metrics' in inspect.signature(sch.step).parameters:
                        sch.step(val_loss_value if val_loss_value is not None else model._loss.avg)
                    else:
                        sch.step()
                except Exception:
                    pass

            stop = False
            for cb in callbacks:
                if hasattr(cb, 'on_epoch_end'):
                    res = cb.on_epoch_end(epoch, logs=logs, logger=self.logger, trainer=self)
                    if res is True:
                        stop = True
            if stop:
                break

    def train(self, num_epochs: int = 1, batch_size: int = 32,
              callbacks: Optional[List[Callback]] = None,
              num_epochs_per_validation: int = 1):
        """Training loop with periodic validation.

        Mirrors your main.py invocation:
            trainer.train(num_epochs=..., batch_size=..., callbacks=[...], num_epochs_per_validation=10)
        """
        if self._train_dataset is None:
            raise RuntimeError("Trainer was not given a train_dataset.")
        # Build loaders fresh to apply batch_size
        tr_kwargs = dict(self._train_loader_kwargs)
        va_kwargs = dict(self._valid_loader_kwargs)
        tr_kwargs['batch_size'] = batch_size
        if 'shuffle' not in tr_kwargs:
            tr_kwargs['shuffle'] = True
        train_loader = self._make_loader(self._train_dataset, tr_kwargs)
        val_loader = self._make_loader(self._valid_dataset, va_kwargs)

        model = self.model
        net = self._get_net()
        opt = self._get_optimizer()
        sch = getattr(model, 'scheduler', lambda: None)()

        callbacks = callbacks or []

        for epoch in range(1, num_epochs + 1):
            model._loss.reset()
            if hasattr(model, 'reset_scores'):
                model.reset_scores()
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

            logs = {'loss': model._loss.avg}
            self._add_common_logs(logs)
            if hasattr(model, 'get_logged_scores'):
                logs.update(model.get_logged_scores())
            if hasattr(model, 'get_logged_scores'):
                logs.update(model.get_logged_scores())

            val_loss_value = None
            if (val_loader is not None) and (epoch % max(1, int(num_epochs_per_validation)) == 0):
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
                logs['val_loss'] = val_loss_value
                crit = getattr(model, 'criterion_name', None)
                if crit:
                    logs[f'val/{crit}'] = val_loss_value

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
        net = self._get_net()
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
