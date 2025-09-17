
from __future__ import annotations

class AverageScore:
    def __init__(self, name: str = "score"):
        self.name = name
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        try:
            v = float(value)
        except Exception:
            return
        self.total += v * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def __repr__(self):
        return f"AverageScore(name={self.name!r}, avg={self.avg:.6f}, count={self.count})"
