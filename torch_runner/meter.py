import math

from pydantic import BaseModel
from pydantic import PrivateAttr
from registry import ObjectRegistry


class MeterRegistry(ObjectRegistry["AvgMeter"]):
    @classmethod
    def validate_key(cls, key: "AvgMeter") -> str:
        if isinstance(key, str):
            return key
        return key.name


@MeterRegistry.register_class_instances
class AvgMeter(BaseModel):
    name: str
    _sum: float = PrivateAttr(0)
    _count: float = PrivateAttr(0)
    _value: float = PrivateAttr(math.nan)

    @classmethod
    def new_meter(cls, name: str):
        if name in MeterRegistry.keys():
            return MeterRegistry.get_registry_item(name)
        else:
            return cls(name=name)

    def update(self, value: float, n: int = 1) -> float:
        assert isinstance(
            value, (int, float)
        ), f"{value} is not a number, found {type(value).__name__}"
        assert isinstance(n, int), f"{n} is not an integer, found {type(n).__name__}"
        assert n > 0, f"{n} is not a positive integer"
        self._value = value
        self._sum += n * value
        self._count += n
        return value

    def compute(self) -> float:
        return float(self._sum / self._count) if self._count != 0 else math.nan

    def reset(self) -> None:
        self._value = math.nan
        self._sum = 0
        self._count = 0
