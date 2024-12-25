import torch
from validation import TorchBatchValidator
from validation import TorchTensorValidator

batch_validator = TorchBatchValidator(
    validators=dict(
        input=TorchTensorValidator(
            shape=(None, 1, 256, 256), device="cuda", strict=False
        ),
        target=TorchTensorValidator(
            shape=(None, 4, 256, 256), device="cuda", strict=False
        ),
    ),
    collate_fn=lambda batch: {"input": batch[0], "target": batch[1]},
)


inp = torch.rand(8, 1, 256, 256)
trg = torch.rand(8, 4, 256, 256)

batch = inp, trg
