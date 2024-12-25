import inspect
from contextlib import contextmanager

import torch
from annotated_types import Ge
from pydantic import AfterValidator
from pydantic import BaseModel
from pydantic import InstanceOf
from pydantic import field_serializer
from pydantic import field_validator

from .._typing_compat import Annotated
from .._typing_compat import Callable
from .._typing_compat import ClassVar
from .._typing_compat import Dict
from .._typing_compat import Optional
from .._typing_compat import Tuple


class TorchTensorValidationError(ValueError):
    """Custom exception for tensor validation errors."""


class TorchTensorValidator(BaseModel):
    DEVICE_AVAILABILITY_PREDICATES: ClassVar[Dict[str, Callable[[], bool]]] = {
        "cpu": lambda: True,
        "cuda": torch.cuda.is_available,
    }
    DTYPE_STRING_REPR: ClassVar[Dict[str, InstanceOf[torch.dtype]]] = {
        name: dtype
        for name, dtype in dict(inspect.getmembers(torch)).items()
        if isinstance(dtype, torch.dtype)
    }
    device: Optional[InstanceOf[torch.device]] = (
        None  # Expected device, e.g., "cpu" or "cuda"
    )
    shape: Optional[Tuple[Optional[Annotated[int, Ge(-1)]], ...]] = (
        None  # Expected shape of the tensor, use -1 for any size in a dimension
    )
    dtype: Optional[InstanceOf[torch.dtype]] = (
        None  # Expected data type, e.g., torch.float32
    )

    strict: bool = True

    @field_serializer("device", when_used="json")
    def serialize_device(v: torch.device):
        return str(v)

    @field_validator("device", mode="before")
    def check_device(cls, v):
        if isinstance(v, str):
            return torch.device(v)
        return v

    @field_validator("shape", mode="after")
    def check_shape(cls, v):
        if sum(filter(lambda dim: dim == -1, v)) < -1:
            raise TorchTensorValidationError("Cannot have more than one -1 in dims!")
        return v

    @field_validator("dtype", mode="before")
    def check_dtype(cls, v):
        if isinstance(v, str):
            if v not in cls.DTYPE_STRING_REPR:
                valid_types = ", ".join(cls.DTYPE_STRING_REPR.keys())
                raise TorchTensorValidationError(
                    f"Invalid type {v}, must be either of {valid_types}."
                )
            return cls.DTYPE_STRING_REPR[v]
        return v

    @field_serializer("dtype", when_used="json")
    def serialize_dtype(v: torch.dtype):
        v = str(v)
        if v.startswith("torch."):
            return v[6:]
        return v

    @field_validator("device", mode="after")
    def check_device_availability(cls, v: Optional[torch.device]):
        if isinstance(v, torch.device):
            if v.type not in cls.DEVICE_AVAILABILITY_PREDICATES:
                valid_keys = ", ".join(cls.DEVICE_AVAILABILITY_PREDICATES.keys())
                raise TorchTensorValidationError(
                    f"Unknown device: {v.type}, must be either of {valid_keys}."
                )
            elif not cls.DEVICE_AVAILABILITY_PREDICATES[v.type]():
                predicate = cls.DEVICE_AVAILABILITY_PREDICATES[v.type].__name__
                raise TorchTensorValidationError(
                    f"Unavailable device {v.type}: {predicate} returns False."
                )

        return v

    def validate_tensor(
        self, tensor: torch.Tensor, strict: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Validates the tensor against the expected properties.

        Parameters:
        tensor (torch.Tensor): The tensor to validate.

        Returns:
        tensor (torch.Tensor): The validated tensor if it passes all checks.

        Raises:
        TorchTensorValidationError: If the tensor does not meet the expected properties.
        """
        if strict is None:
            strict = self.strict
        # Check device
        if self.device and (
            tensor.device.type != self.device.type
            or self.device.index
            and torch.device.index != self.device.index
        ):
            if strict:
                raise TorchTensorValidationError(
                    f"Expected tensor on device '{self.device}', but got '{tensor.device.type}'"
                )
            else:
                tensor = tensor.to(device=self.device)

        # Check dtype
        if self.dtype and tensor.dtype != self.dtype:
            if strict:
                raise TorchTensorValidationError(
                    f"Expected tensor with dtype {self.dtype}, but got {tensor.dtype}"
                )
            else:
                tensor = tensor.to(dtype=self.dtype)

        # Check shape
        if self.shape:
            if len(tensor.shape) != len(self.shape):
                raise TorchTensorValidationError(
                    f"Expected tensor with shape {self.shape}, but got {tensor.shape}"
                )

            for expected_dim, actual_dim in zip(self.shape, tensor.shape):
                if expected_dim is not None and expected_dim != actual_dim:
                    raise TorchTensorValidationError(
                        f"Expected dimension {expected_dim} but got {actual_dim} in tensor shape {tensor.shape}"
                    )

        return tensor

    def tensor_validator(self) -> AfterValidator:
        return AfterValidator(self.validate_tensor)

    @contextmanager
    def validate_tensor_context(self, tensor: torch.Tensor):
        """
        Context manager to validate a tensor within a block of code.
        """
        validated_tensor = self.validate_tensor(tensor)
        try:
            yield validated_tensor
        except TorchTensorValidationError as e:
            raise e
