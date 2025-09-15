# torchutils/metrics.py
class AverageScore:
    def __init__(self, name=None, **kwargs):
        self.name = name
        self.reset()
    def reset(self):
        self.sum = 0.0
        self.count = 0
    def update(self, value, n: int = 1):
        try:
            v = float(value)
        except Exception:
            v = 0.0
        self.sum += v * n
        self.count += n
    @property
    def value(self):
        return self.sum / max(self.count, 1)
    def __float__(self):
        return self.value
