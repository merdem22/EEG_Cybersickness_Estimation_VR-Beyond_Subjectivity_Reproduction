import logging

class EarlyStopping:
    def __init__(self, monitor='mse_loss', goal='minimize', patience=10, delta=0.0, verbose=20):
        self.monitor = monitor
        self.goal = goal
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best = None
        self.wait = 0
        self.should_stop = False
        self.lgr = logging.getLogger(__name__)
    def on_epoch_end(self, trainer, metrics: dict):
        val = metrics.get(self.monitor)
        if val is None:
            return
        improve = (self.best is None or
                   (self.goal == 'minimize' and val < self.best - self.delta) or
                   (self.goal != 'minimize' and val > self.best + self.delta))
        if improve:
            self.best = val
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
                self.lgr.log(self.verbose, f'EarlyStopping triggered on {self.monitor}: best={self.best:.6f}')
    # allow calling as a function
    def __call__(self, *args, **kwargs):
        return self.on_epoch_end(*args, **kwargs)

class AverageScoreLogger:
    def __init__(self, *names, level=20):
        self.names = names
        self.level = level
        self.lgr = logging.getLogger(__name__)
    def on_epoch_end(self, trainer, metrics: dict):
        msg = ', '.join(f'{n}={metrics.get(n)}' for n in self.names if n in metrics)
        if msg:
            self.lgr.log(self.level, msg)
    def on_predict_end(self, trainer, metrics: dict):
        self.on_epoch_end(trainer, metrics)
