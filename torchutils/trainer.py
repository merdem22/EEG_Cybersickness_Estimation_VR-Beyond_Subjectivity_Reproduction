# torchutils/trainer.py
import torch
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, train_dataset=None, valid_dataset=None,
                 train_dataloader_kwargs=None, valid_dataloader_kwargs=None):
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_dataloader_kwargs = train_dataloader_kwargs or {}
        self.valid_dataloader_kwargs = valid_dataloader_kwargs or {}

        # build optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        self.optim = torch.optim.Adam(params, lr=1e-3)

        # give the model a handle (your model reads self.optimizer.param_groups)
        self.model.optimizer = self.optim
        # also keep a synonym for convenience (some code expects .optimizer on Trainer too)
        self.optimizer = self.optim

        self.history = []

    def compile(self, handlers=None):
        return self

    def _make_loader(self, dataset, batch_size, for_train=True, extra_kwargs=None):
        if dataset is None:
            return None
        kw = dict(batch_size=batch_size, shuffle=for_train, drop_last=False)
        if isinstance(extra_kwargs, dict):
            kw.update(extra_kwargs)
        return DataLoader(dataset, **kw)

    def _to_device(self, batch, device):
        if isinstance(batch, dict):
            out = {}
            for k, v in batch.items():
                try:
                    out[k] = v.to(device)
                except Exception:
                    out[k] = v
            return out
        return batch

    def train(self, num_epochs=1, batch_size=32, callbacks=None, num_epochs_per_validation=1):
        callbacks = callbacks or []
        device = getattr(self.model, 'device', 'cpu')
        train_loader = self._make_loader(self.train_dataset, batch_size, for_train=True,
                                         extra_kwargs=self.train_dataloader_kwargs)
        valid_loader = self._make_loader(self.valid_dataset, batch_size, for_train=False,
                                         extra_kwargs=self.valid_dataloader_kwargs)

        for epoch in range(num_epochs):
            self.model.train()
            if hasattr(self.model, 'begin_epoch'):
                self.model.begin_epoch()

            for batch_idx, batch in enumerate(train_loader):
                batch = self._to_device(batch, device)

                # allow model to return (preds, loss) or just preds
                out = self.model.forward(batch, batch_idx=batch_idx)
                if isinstance(out, tuple) and len(out) == 2:
                    preds, loss = out
                else:
                    preds = out
                    loss = self.model.criterion(preds, batch['observation'])

                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                self.optim.step()

                # keep running average in model._loss if present
                if hasattr(self.model, '_loss'):
                    try:
                        n = batch['observation'].shape[0]
                    except Exception:
                        n = 1
                    self.model._loss.update(float(loss.item()), n=n)

            # validation step
            if valid_loader is not None and (epoch + 1) % max(1, num_epochs_per_validation) == 0:
                self.model.eval()
                with torch.no_grad():
                    if hasattr(self.model, 'begin_epoch'):
                        self.model.begin_epoch()
                    for batch_idx, batch in enumerate(valid_loader):
                        batch = self._to_device(batch, device)
                        if hasattr(self.model, 'forward_pass_on_evauluation_step'):
                            _ = self.model.forward_pass_on_evauluation_step(batch, batch_idx=batch_idx)
                        else:
                            out = self.model.forward(batch, batch_idx=batch_idx)
                            if isinstance(out, tuple) and len(out) == 2:
                                preds, loss = out
                            else:
                                preds = out
                                loss = self.model.criterion(preds, batch['observation'])
                            if hasattr(self.model, '_loss'):
                                self.model._loss.update(float(loss.item()))

            metrics = self.model.end_epoch_metrics() if hasattr(self.model, 'end_epoch_metrics') else {}
            self.history.append(metrics)
            for cb in callbacks:
                if hasattr(cb, 'on_epoch_end'):
                    cb.on_epoch_end(self, metrics)
                elif callable(cb):
                    cb(self, metrics)
                if getattr(cb, 'should_stop', False):
                    return

    def predict(self, dataset, callbacks=None, dataloader_kwargs=None):
        callbacks = callbacks or []
        device = getattr(self.model, 'device', 'cpu')
        loader = self._make_loader(dataset,
                                   batch_size=(dataloader_kwargs or {}).get('batch_size', 32),
                                   for_train=False,
                                   extra_kwargs=dataloader_kwargs or {})
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'begin_epoch'):
                self.model.begin_epoch()
            for batch_idx, batch in enumerate(loader):
                batch = self._to_device(batch, device)
                if hasattr(self.model, 'forward_pass_on_evauluation_step'):
                    _ = self.model.forward_pass_on_evauluation_step(batch, batch_idx=batch_idx)
                else:
                    _ = self.model.forward(batch, batch_idx=batch_idx)
        metrics = self.model.end_epoch_metrics() if hasattr(self.model, 'end_epoch_metrics') else {}
        for cb in callbacks:
            if hasattr(cb, 'on_predict_end'): cb.on_predict_end(self, metrics)
            elif hasattr(cb, 'on_epoch_end'): cb.on_epoch_end(self, metrics)
            elif callable(cb): cb(self, metrics)
