
from __future__ import annotations
import logging
from typing import Iterable, Optional, Dict, Any, List, Union
import torch
from torch.utils.data import DataLoader
from .callbacks import Callback, EarlyStopping, AverageScoreLogger

class Trainer:
    """A minimal training/eval loop compatible with the provided repo style.

    It expects a `model` that extends `torchutils.models.TrainerModel` and
    provides:
      - forward(batch): returns a loss tensor or predictions; should update model._loss (AverageScore)
      - forward_pass_on_evauluation_step(batch): like forward but for eval
      - writable_scores(): set/list of score names (e.g., {'loss', 'accuracy', 'mse_loss', ...})
    """
    def __init__(self, model, net: Optional[torch.nn.Module] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 logger: Optional[logging.Logger] = None):
        self.model = model
        self.logger = logger or logging.getLogger('torchutils.trainer')
        self.logger.propagate = False  # we manage handlers
        self._handlers_added = False

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
        # Attach logging handlers (console/file) if provided.
        if handlers and not self._handlers_added:
            for h in handlers:
                self.logger.addHandler(h)
            self._handlers_added = True

    # ----- internal helpers -----
    def _make_loader(self, dataset, dataloader_kwargs: Optional[Dict[str, Any]]):
        if dataloader_kwargs is None:
            dataloader_kwargs = {}
        if 'batch_size' not in dataloader_kwargs:
            dataloader_kwargs['batch_size'] = 32
        if 'shuffle' not in dataloader_kwargs:
            dataloader_kwargs['shuffle'] = True
        return DataLoader(dataset, **dataloader_kwargs)

    def _compute_loss_from_output(self, output, batch) -> Optional[torch.Tensor]:
        """Best-effort way to derive a loss tensor from model output.

        - If `output` is a scalar tensor, return it.
        - If it's a dict with 'loss', return that.
        - If it's a list/tuple of predictions and batch has targets under 'observation',
          try to average self.model.criterion(pred, targ) over items.
        """
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

        for epoch in range(1, num_epochs + 1):
            model._loss.reset()
            if hasattr(model, '_buffer') and isinstance(model._buffer, dict):
                # optionally reset epoch-specific buffers if subclass uses them
                model._buffer.setdefault('epoch', epoch)
            if net is not None:
                net.train()

            for step, batch in enumerate(train_loader):
                # Move tensors to device if possible
                try:
                    batch = {k: (v.to(self.device) if hasattr(v, 'to') else v) for k, v in batch.items()}
                except Exception:
                    pass

                out = model.forward(batch)
                loss = self._compute_loss_from_output(out, batch)
                if loss is None:
                    # Allow subclasses to set ._last_loss
                    loss = getattr(model, '_last_loss', None)
                if loss is None:
                    raise RuntimeError("Model.forward didn't return a loss tensor and no fallback was possible.")

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            # ---- end epoch: compute training logs ----
            train_logs = {'loss': model._loss.avg}
            # Validation
            val_logs = {}
            if val_loader is not None:
                if net is not None:
                    net.eval()
                model._loss.reset()
                with torch.no_grad():
                    for vstep, vbatch in enumerate(val_loader):
                        try:
                            vbatch = {k: (v.to(self.device) if hasattr(v, 'to') else v) for k, v in vbatch.items()}
                        except Exception:
                            pass
                        # Use eval-specific forward if available
                        if hasattr(model, 'forward_pass_on_evauluation_step'):
                            _ = model.forward_pass_on_evauluation_step(vbatch)
                        else:
                            _ = model.forward(vbatch)
                val_logs['val_loss'] = model._loss.avg

            # Merge logs
            logs = {**train_logs, **val_logs}

            # Callbacks
            stop = False
            for cb in callbacks:
                # Provide logger and trainer to callbacks
                if hasattr(cb, 'on_epoch_end'):
                    res = cb.on_epoch_end(epoch, logs=logs, logger=self.logger, trainer=self)
                    # EarlyStopping returns True to stop
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
        import torch
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

        # Aggregate and call callbacks (best-effort)
        logs = {'loss': model._loss.avg}
        for cb in callbacks:
            if hasattr(cb, 'on_predict_end'):
                cb.on_predict_end(logs=logs, logger=self.logger)
        return preds
