import logging
from collections import defaultdict
from functools import partial
from functools import wraps
import inspect
import random
import pdb


import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from callpyback import CallbackHandler
from callpyback import ThreadingPolicy
from callpyback import DictReduction

from ._typing_compat import Any
from ._typing_compat import Callable
from ._typing_compat import Literal
from ._typing_compat import Optional
from ._typing_compat import Set
from ._typing_compat import Union
from .dev_utils import BasicProfiler
from .dev_utils import ContextProfiler
from .dev_utils import TorchProfiler
from .early_stop import StopTraining
from .meter import AvgMeter

"""
def meter_save(
    y_pred: Any, y_true: Any, metric: Callable[[Any, Any], Any], meter: AvgMeter
) -> Any:
    return meter.update(float(metric(y_pred, y_true)), n=1)
"""


def _get_callable_name(fn: Union[Callable, partial]):
    if isinstance(fn, partial):
        return _get_callable_name(fn=fn.func)
    if inspect.isfunction(fn):
        return fn.__name__
    elif isinstance(fn, object):
        return fn.__class__.__name__
    else:
        raise TypeError(f"unknown type for {fn}")


def _prepare_dataloader(
    dataset: Dataset,
    batch_size: int,
    train: bool,
    seed: int = 42,
    **kwds,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        worker_init_fn=lambda worker_id: np.random.seed(seed, worker_id) ** kwds,
    )  # Adjust batch size as needed


# Set random seeds for reproducibility
def set_reproducibility_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def profiler_to_meters_callback(prof: Union[BasicProfiler, TorchProfiler]):
    for key, val in prof.stats.items():
        AvgMeter.new_meter(key).update(val)


def create_metric_handler(
    metrics: Set[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
) -> CallbackHandler:
    hdlr = CallbackHandler()
    hdlr.add_delegation("compute_metrics", allowed_parameters={"y_pred", "y_true"})
    hdlr.add_delegation("reset_meters", allowed_parameters=set())
    # TODO: buraya mean Reduction eklemek lazim
    # boylece execute calistiginda return value olarak reduction cikar
    # hdlr.add_delegation("get_meters", allowed_parameters=set())

    def metric_update(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        meter: AvgMeter,
        metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        return meter.update(float(metric(y_pred, y_true)), n=len(y_true))

    for metric in metrics:
        metric_name = _get_callable_name(metric)
        meter = AvgMeter.new_meter(name=metric_name)

        hdlr.delegate_function(
            compute_metrics=partial(metric_update, meter=meter, metric=metric),
            reset_meters=meter.reset,
            # get_meters=meter.compute
        )

    return hdlr


def create_callback_handler(callbacks: Set[Callable]) -> CallbackHandler:
    policy = ThreadingPolicy(exceptions=StopTraining)
    hdlr = CallbackHandler(policy=policy)

    hdlr.add_delegation("add_handler", allowed_parameters={"hdlr"})
    hdlr.add_delegation("remove_handler", allowed_parameters={"hdlr"})

    hdlr.add_delegation("on_training_begin", allowed_parameters={"hparams"})
    hdlr.add_delegation("on_validation_run_begin", allowed_parameters={"epoch_index"})
    hdlr.add_delegation("on_validation_run_end", allowed_parameters=set())

    for cb in callbacks:
        hdlr.delegate_class(cb)

    return hdlr


def train(
    model: torch.nn.Module,
    dataset: Dataset,
    num_epochs: int,
    batch_size: int,
    criterion: Union[
        torch.nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ],  # Mean Squared Error Loss
    learning_rate: float = 0.001,
    test_dataset: Optional[Dataset] = None,
    metrics: Set[Callable] = set(),
    callbacks: Set[Callable] = set(),
    handlers: Set[logging.Handler] = set(),
    # device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    device = next(model.parameters()).device
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    set_reproducibility_seed(42)
    train_data_loader = _prepare_dataloader(
        dataset, batch_size=batch_size, train=True, drop_last=False
    )
    if test_dataset:
        test_data_loader = _prepare_dataloader(
            test_dataset, batch_size=batch_size, train=False, drop_last=False
        )
    else:
        test_data_loader = None

    # 3. Set up the loss function and optimizer
    # criterion = torch.nn.MSELoss()  # Mean Squared Error Loss
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )  # Adam optimizer with learning rate

    # 4. prepare metric handlers
    mt_hdlr = create_metric_handler(metrics=metrics)

    # 5. create callback handler
    cb_hdlr = create_callback_handler(callbacks=callbacks)

    # add handlers
    for hdlr in handlers:
        logger.addHandler(hdlr)
        cb_hdlr.execute("add_handler", hdlr=hdlr)

    # call on_training_begin
    hparams = dict(
        model_name=model.__class__.__name__,
        loss_name=_get_callable_name(criterion),
        num_epochs_per_validation=1,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    logger.info(f"training parameters: {hparams}")
    cb_hdlr.execute("on_training_begin", hparams=hparams)

    # with ContextProfiler(
    #     profiler=TorchProfiler(
    #         # name="Epochs",
    #         profile_memory=True,
    #         # profile_time=True,
    #         with_flops=True,
    #         on_trace_ready=profiler_to_meters_callback,
    #         schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, skip_first=1),
    #     )
    # ) as prof:
    if True:
        for epoch in range(num_epochs):
            train_step(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                data_loader=train_data_loader,
                num_epochs=num_epochs,
                current_epoch=epoch,
                device=device,
                log_func=logger.info,
                profiler=None,
            )
            if test_dataset:
                mt_hdlr.execute("reset_meters")
                try:
                    eval_step(
                        epoch=epoch,
                        model=model,
                        callback_handler=cb_hdlr,
                        data_loader=test_data_loader,
                        device=device,
                        metric_handler=mt_hdlr,
                        log_func=logger.info,
                    )
                except StopTraining as msg:
                    logger.info(
                        f"Training stopped at epoch [{epoch+1}/{num_epochs}]! "
                        "Possibly a callback purposefully raised `StopTraining` "
                        "exception during its execution."
                    )
                    logger.debug(f"Traceback Exception Message: {msg}")
                    break

    for hdlr in handlers:
        logger.removeHandler(hdlr)
        cb_hdlr.execute("remove_handler", hdlr=hdlr)
    return model


# @Profiler(with_flops=True)
def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[Any, Any], Any],
    data_loader: DataLoader,
    num_epochs: int,
    current_epoch: int,
    log_func: Callable[[str], None] = print,
    device: Union[torch.device, str] = "cuda" if torch.cuda.is_available() else "cpu",
    profiler: Union[TorchProfiler, BasicProfiler, None] = None,
):
    # if profiler:
    #     profiler.start()
    model.train()
    running_loss = 0.0
    for i, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)

        # Forward pass
        # p = model.cnnfoils[0](x)
        # u = model.cnnfoils[1](x)
        # v = model.cnnfoils[2](x)
        # t = model.cnnfoils[3](x)
        # y_hat = torch.cat([p, u, v, t], dim=1)
        # pdb.set_trace()
        y_hat = model(x, y)

        # Compute loss
        loss = criterion(y_hat, y)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.detach().item()

        if (i + 1) % 7 == 0:  # Print every 10 batches
            log_func(
                (
                    "Epoch [{current_epoch}/{total_epochs}], "
                    "Step [{current_step}/{total_steps}], "
                    "Loss: {step_loss:.4e}"
                ).format(
                    current_epoch=current_epoch + 1,
                    total_epochs=num_epochs,
                    current_step=i + 1,
                    total_steps=len(data_loader),
                    step_loss=loss.detach().item(),
                )
            )
        if profiler:
            profiler.step()

    # if profiler:
    #     profiler.stop()
    # Compute epoch loss
    epoch_loss = running_loss / len(data_loader)
    log_func(
        "Epoch [{current_epoch}/{total_epochs}], Average Loss: {epoch_loss:.4e}".format(
            current_epoch=current_epoch + 1,
            total_epochs=num_epochs,
            epoch_loss=epoch_loss,
        )
    )


# @Profiler(with_flops=True)
@torch.inference_mode()
@torch.no_grad()
def eval_step(
    epoch: int,
    model: torch.nn.Module,
    data_loader: DataLoader,
    callback_handler: CallbackHandler,
    metric_handler: CallbackHandler,
    device: Union[torch.device, str] = "cuda" if torch.cuda.is_available() else "cpu",
    log_func: Callable[[str], None] = print,
):
    callback_handler.execute("on_validation_run_begin", epoch_index=epoch + 1)

    # 5. Evaluation
    model.eval()
    metric_scores = defaultdict(list)

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        y_hat = model(x)

        metric_handler.execute("compute_metrics", y_pred=y_hat, y_true=y)
        # for metric in metrics:
        #     metric_name = _get_callable_name(metric)
        #     metric_score = metric(y_pred=y_hat, y_true=y)
        #     metric_scores[metric_name].append(metric_score)

    # metric_scores = {
    #     name: sum(scores) / len(scores) if scores else None
    #     for name, scores in metric_scores.items()
    # }

    # log_func(
    #     f"Metrics: "
    #     + ", ".join(f"{name}={value:.4f}" for name, value in metric_scores.items())
    # )

    callback_handler.execute("on_validation_run_end")
