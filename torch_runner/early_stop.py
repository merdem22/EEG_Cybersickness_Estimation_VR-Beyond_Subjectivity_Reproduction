# Copyright © 2022 Adnan Harun Dogan
import enum
import logging
from typing import Any

import pydantic
from callpyback.basics import delegatedmethod

from .meter import MeterRegistry


class GoalEnum(enum.Enum):
    minimize = "minimize"
    maximize = "maximize"


def _goodness_condition(score: float, threshold: float, maximize: bool) -> bool:
    if maximize:
        return score > threshold
    return score < threshold


class StopTraining(Exception):
    pass


class BaseCallback(pydantic.BaseModel):
    logger: pydantic.InstanceOf[logging.Logger] = logging.getLogger(__name__)

    @delegatedmethod
    def add_handler(self, hdlr: logging.Handler):
        self.logger.addHandler(hdlr)

    @delegatedmethod
    def remove_handler(self, hdlr: logging.Handler):
        self.logger.removeHandler(hdlr)


class EarlyStopping(BaseCallback):
    """Early stops the training if validation loss doesn't improve after a given patience."""

    monitor: str
    """The metric or quantity to be monitored."""

    goal: GoalEnum
    """How improving the monitored qunatity defined. If GoalEnum.maximize, the maximum is better, otherwise, vice versa"""

    patience: int = 7
    """How long to wait after last time the monitored quantity has improved."""

    verbose: bool
    """If True, prints a message for each improvement in the monitored quantity."""

    reset_on_begin: bool = False
    """If true all hidden parameters will be reset during on_training_begin call."""

    delta: float = pydantic.Field(0.0, default_validate=True)
    """Minimum change in the monitored quantity to qualify as an improvement."""

    _counter: int = pydantic.PrivateAttr(0)
    _score: float = pydantic.PrivateAttr(float("-inf"))
    _epoch: int = pydantic.PrivateAttr(0)
    _nepv: int = pydantic.PrivateAttr()

    def model_post_init(self, context: Any):
        self.logger.setLevel(logging.INFO if self.verbose else logging.ERROR)
        self._score = float("-inf") if self.maximize else float("inf")

    @pydantic.field_validator("delta")
    def validate_delta(cls, value, values: pydantic.ValidationInfo):
        return (1 + value) if values.data["goal"].value == "maximize" else (1 - value)

    @pydantic.computed_field
    def maximize(self) -> bool:
        return self.goal == GoalEnum.maximize

    @pydantic.computed_field
    def threshold(self) -> float:
        threshold = self._score * self.delta
        return -(threshold) if self.maximize else threshold

    def __hash__(self) -> int:
        return hash(self.model_dump_json(exclude={"logger"}))

    @delegatedmethod
    def on_training_begin(self, hparams):
        if self.reset_on_begin:
            self._counter: int = 0
            self._epoch: int = 0
            self._score = float("-inf") if self.maximize else float("inf")
        self._nepv = hparams["num_epochs_per_validation"]
        if hparams["num_epochs_per_validation"] == 0:
            self.logger.error("EarlyStopping will not be called while training.")

    @delegatedmethod
    def on_validation_run_begin(self, epoch_index: int):
        self._epoch = epoch_index

    @delegatedmethod
    def on_validation_run_end(self):
        # todo write a proper wrapper function
        score = MeterRegistry.get_registry_item(self.monitor).compute()

        threshold = self._score * self.delta

        self.logger.debug(
            f"score: {score} best score: {self._score} threshold: {threshold}"
        )
        if _goodness_condition(score, threshold, self.maximize):  # type: ignore
            self.logger.info(
                f"Score {score:.3e}, thresholding {threshold:.3e}, set to the best."
            )
            self._score = score
            self._counter = 0
        elif self._counter < self.patience:
            self.logger.info(
                f"Score {score:.3e}, making plateau in the last [{self._counter}/{self.patience}] validation run."
            )
            self._counter += self._nepv
        else:
            delta = (self.delta - 1) if self.maximize else (1 - self.delta)

            stopping_message = (
                f"the best {self.monitor}={self._score:.3e} score "
                f"not optimized {delta * 100:.2f}% in the last {self.patience} validation run."
            )

            self.logger.warn(
                f"Stopping training early at epoch {self._epoch} as " + stopping_message
            )
            raise StopTraining(stopping_message)
