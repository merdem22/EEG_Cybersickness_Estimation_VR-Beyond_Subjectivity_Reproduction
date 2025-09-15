# torchutils/models.py
import inspect
import torch
import torch.nn as nn
from .metrics import AverageScore
from .resolver import resolve_model

class TrainerModel(nn.Module):
    def __init__(self, model=None, device='cpu', writable_scores=None,
                 criterion_name='mse_loss', **kwds):
        super().__init__()

        # expose a scratch dict; your MyTrainerModel uses this
        self._buffer = {}

        # instantiate user model
        if isinstance(model, str):
            ctor = resolve_model(model)
            sig = inspect.signature(ctor.__init__)
            ctor_kwargs = {k: v for k, v in kwds.items() if k in sig.parameters}
            try:
                self.model = ctor(**ctor_kwargs)
            except TypeError:
                self.model = ctor()
        elif isinstance(model, nn.Module):
            self.model = model
        else:
            self.model = None

        # device
        want = device.lower()
        if want == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif want == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.to(self.device)

        # metrics/loss slots
        self._score_avgs = {}
        self._last_scores = {}
        self._writable_scores = set(writable_scores or [])
        self._loss = AverageScore()
        self.criterion_name = criterion_name
        self.criterion = nn.MSELoss()

        # model-side optimizer handle (Trainer will set this)
        self.optimizer = None

    # exposed API
    def writable_scores(self):
        return tuple(sorted(self._writable_scores))

    def log_score(self, name, value):
        avg = self._score_avgs.get(name)
        if avg is None:
            avg = AverageScore(name=name)
            self._score_avgs[name] = avg
        avg.update(value)
        self._last_scores[name] = float(value)
        self._writable_scores.add(name)

    def begin_epoch(self):
        self._loss.reset()
        for a in self._score_avgs.values():
            a.reset()

    def end_epoch_metrics(self):
        out = {k: v.value for k, v in self._score_avgs.items()}
        out[self.criterion_name] = self._loss.value
        return out

    def log(self, msg):
        print(msg)
