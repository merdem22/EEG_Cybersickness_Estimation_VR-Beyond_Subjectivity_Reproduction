import torch
from pydantic import BaseModel
from pydantic import InstanceOf

from .._typing_compat import Callable
from .._typing_compat import Dict
from .._typing_compat import Tuple
from .._typing_compat import Union
from .torch_tensor_validator import TorchTensorValidationError
from .torch_tensor_validator import TorchTensorValidator

TorchTensorBatchType = Union[
    Tuple[InstanceOf[torch.Tensor], ...], Dict[str, InstanceOf[torch.Tensor]]
]


class TorchBatchValidationError(ValueError):
    """Custom exception for batch validation errors."""


class TorchBatchValidator(BaseModel):
    collate_fn: Callable[[TorchTensorBatchType], Dict[str, InstanceOf[torch.Tensor]]]
    validators: Dict[str, TorchTensorValidator]

    def validate_batch(self, batch: TorchTensorBatchType) -> Dict[str, torch.Tensor]:
        """
        Validates the batch of tensors against the specified validators.

        Parameters:
        batch (Tuple[torch.Tensor, ...] | Dict[str, torch.Tensor]): A dictionary
                                        where keys are names (e.g., 'input', 'target')
                                        and values are the tensors to validate.

        Returns:
        Dict[str, torch.Tensor]: The validated batch if all tensors pass their checks.

        Raises:
        TorchBatchValidationError: If any tensor in the batch does not meet the expected properties.
        """
        collated_batch = self.collate_fn(batch)
        validated_batch = {}
        for key, tensor in collated_batch.items():
            if key in self.validators:
                try:
                    validated_batch[key] = self.validators[key].validate_tensor(tensor)
                except TorchTensorValidationError as e:
                    raise TorchBatchValidationError(
                        f"Validation error for '{key}': {e}"
                    )
            else:
                validated_batch[key] = (
                    tensor  # If no validator is specified, pass the tensor as-is
                )

        return validated_batch

    def validate_batch_context(self, batch: TorchTensorBatchType):
        """
        Context manager to validate a batch of tensors within a block of code.
        """
        validated_batch = self.validate_batch(batch)
        try:
            yield validated_batch
        except TorchBatchValidationError as e:
            raise e
