import logging
import math
import os
import time
from typing import Any
from typing import ClassVar
from typing import Dict
from typing import Optional

import psutil
import torch
import torch.profiler as P
from pydantic import BaseModel
from pydantic import Field
from pydantic import InstanceOf
from typing_extensions import Callable

if logging.getLevelName(15) != "PROFILE":
    logging.addLevelName(15, "PROFILE")


def _float_to_metric_unit(f: float, unit="B", base=1024) -> str:
    if f == 0:
        deg = 0
        val = 0
    else:
        deg = int(math.log(abs(f)) // math.log(base))
        val = f / base**deg

    POS_UNIT_PREFIX = [None, "K", "M", "G", "T", "P"]
    NEG_UNIT_PREFIX = ["f", "p", "n", "μ", "m"]

    if deg > 0:
        unit = POS_UNIT_PREFIX[deg] + unit
    elif deg < 0:
        unit = NEG_UNIT_PREFIX[deg] + unit
    else:
        unit = unit + " "

    return "{val:7.2f} {unit:s}".format(val=val, unit=unit)


class BasicProfiler(BaseModel):
    name: str
    stats: Dict[str, float] = dict()
    logger: InstanceOf[logging.Logger] = Field(exclude=True)
    schedule: Callable[[int], P.ProfilerAction] = Field(
        default=lambda _: P.ProfilerAction.RECORD, exclude=True
    )
    """
    Example:
    schedule=torch.profiler.scheduler(wait=1, warmup=1, active=2, repeat=1)

    profiler will skip the first step/iteration,
    start warming up on the second, record
    the third and the forth iterations,
    after which the trace will become available
    and on_trace_ready (when set) is called;
    the cycle repeats starting with the next step
    """

    start_time: float = math.nan
    memory_info: Dict[str, float] = dict()
    time_info: Dict[str, float] = dict()
    process: InstanceOf[psutil.Process] = Field(exclude=True)

    on_trace_ready: Optional[Callable[[Any], Any]] = Field(None, exclude=True)

    PROFILE: ClassVar[int] = 15
    TIME_KEYS: ClassVar = ("system", "user", "elapsed")
    TIME_EXTRA_KEYS: ClassVar = (
        "system",
        "user",
        "elapsed",
        "children_user",
        "children_system",
    )
    MEMORY_KEYS: ClassVar = (
        "shared",
        "rss",
        "vms",
    )
    MEMORY_EXTRA_KEYS: ClassVar = (
        "dirty",
        "lib",
        "data",
        "text",
        "rss",
        "vms",
        "shared",
    )

    step_iter: int = 0
    profile_memory: bool = False
    profile_time: bool = False
    extra: bool = False

    ""

    def __init__(self, name: str, **kwds):
        lgr_name = __name__ + "." + self.__class__.__name__
        kwds.setdefault("logger", logging.getLogger(lgr_name))
        kwds.setdefault("process", psutil.Process(os.getpid()))
        super().__init__(name=name, **kwds)

    def model_post_init(self, __context):
        self.logger.setLevel(self.PROFILE)

    def _log_stats(
        self,
        title: str,
        /,
        sort_keys: bool = False,
        units: str = "B",
        base: int = 1024,
        **stats: float,
    ):
        def prepare_keys(key: str) -> str:
            return " ".join(map(str.capitalize, key.split("_")))

        stat_items = sorted(stats.items()) if sort_keys else stats.items()
        key, val = zip(*stat_items)
        key = map(prepare_keys, key)
        self.stats.update(dict(zip(key, val)))

        self.log(
            (
                "{title:>20s} -- ".format(title=title)
                + " || ".join(
                    prepare_keys(key)
                    + ": "
                    + str(_float_to_metric_unit(val, unit=units, base=base))
                    for key, val in stat_items
                )
            ),
        )

    def log(self, msg: str, **kwds):
        self.logger.log(self.PROFILE, msg=msg, **kwds)

    def set_time_info(self, time_info: os.times_result):
        keys = self.TIME_EXTRA_KEYS if self.extra else self.TIME_KEYS
        self.time_info = {key: getattr(time_info, key) for key in keys}

    def set_memory_info(self, memory_info: psutil._pslinux.pmem):
        keys = self.MEMORY_EXTRA_KEYS if self.extra else self.MEMORY_KEYS
        self.memory_info = {key: getattr(memory_info, key) for key in keys}

    def _start(self):
        # start time tracking
        if self.profile_time:
            self.start_time = time.time()
            self.set_time_info(os.times())

        # start memory tracking
        if self.profile_memory:
            self.set_memory_info(self.process.memory_info())

    def _stop(self):
        # Print time elapse
        if self.profile_time:
            assert not math.isnan(self.start_time), "Call start() first!"
            end_time_info = os.times()
            diff_time_info = {
                key: getattr(end_time_info, key) - value
                for key, value in self.time_info.items()
            }
            if self.extra:
                diff_time_info["precise_elapsed"] = time.time() - self.start_time
            self._log_stats("Time Elapse", units="sec", base=1000, **diff_time_info)
            self.set_time_info(end_time_info)

        # Print memory usage
        if self.profile_memory:
            end_memory_info = self.process.memory_info()
            diff_memory_info = {
                key: getattr(end_memory_info, key) - value
                for key, value in self.memory_info.items()
            }
            self._log_stats("Memory Usage", sort_keys=False, **diff_memory_info)
            self.set_memory_info(end_memory_info)

    def start(self):
        if self.schedule(self.step_iter) != P.ProfilerAction.NONE:
            self._start()

    def stop(self):
        if self.schedule(self.step_iter) != P.ProfilerAction.NONE:
            self._stop()

    def step(self):
        if self.schedule(self.step_iter) != P.ProfilerAction.NONE:
            self._stop()
        if (
            self.schedule(self.step_iter) == P.ProfilerAction.RECORD
            and self.on_trace_ready
        ):
            print("executed!")
            self.on_trace_ready(self)
        self.step_iter += 1
        if self.schedule(self.step_iter) != P.ProfilerAction.NONE:
            self._start()


class TorchProfiler(P.profile):
    def __init__(self, extra: bool = False, **kwds):
        # Start torch profiler
        super().__init__(**kwds)
        self.extra = extra
        self.TIME_KEYS = (
            ("system", "user", "elapsed", "children_user", "children_system")
            if self.extra
            else ("system", "user", "elapsed")
        )
        self.MEMORY_KEYS = (
            ("dirty", "lib", "data", "text", "rss", "vms", "shared")
            if self.extra
            else ("rss", "vms", "shared")
        )  #

        self.logger = logging.getLogger(__name__ + "." + self.__class__.__qualname__)
        self.logger.setLevel("PROFILE")
        self._log_func = lambda msg: self.logger.log(15, msg)
        self.stats = dict()

        if kwds.get("with_flops", False):
            old_on_trace_ready = self.on_trace_ready

            def compute_gflops(prof):
                events = prof.events()
                # events = filter(lambda event: isinstance(event, int), events)
                # total_flops = sum(map(lambda event: int(event.flops), events))
                total_flops = sum(
                    int(evt.flops) for evt in events if isinstance(evt.flops, int)
                )
                events.clear()
                self._log_stats(
                    "Floating-Point Op.", total_flops=total_flops, units="FLOP"
                )
                if callable(old_on_trace_ready):
                    old_on_trace_ready(prof)

            self.on_trace_ready = compute_gflops

    def _log_stats(
        self,
        title: str,
        /,
        sort_keys: bool = False,
        units: str = "B",
        base: int = 1024,
        **stats: float,
    ):
        def prepare_keys(key: str) -> str:
            return " ".join(map(str.capitalize, key.split("_")))

        stat_items = sorted(stats.items()) if sort_keys else stats.items()
        key, val = zip(*stat_items)
        key = map(prepare_keys, key)
        self.stats.update(dict(zip(key, val)))

        self._log_func(
            (
                "{title:>20s} -- ".format(title=title)
                + " || ".join(
                    prepare_keys(key)
                    + ": "
                    + str(_float_to_metric_unit(val, unit=units, base=base))
                    for key, val in stat_items
                )
            )
        )

    def set_time_info(self, time_info: os.times_result):
        self.time_info = {key: getattr(time_info, key) for key in self.TIME_KEYS}

    def set_memory_info(self, memory_info: psutil._pslinux.pmem):
        self.memory_info = {key: getattr(memory_info, key) for key in self.MEMORY_KEYS}

    def start(self):
        # start torch profiler
        super().start()
        self.stats.clear()

        # Start time tracking
        self.start_time = time.time()
        self.set_time_info(os.times())

        # Start memory tracking
        self.process = psutil.Process(os.getpid())
        self.set_memory_info(self.process.memory_info())

        # Start GPU memory tracking if using CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_gpu_mem_alloc = torch.cuda.memory_allocated()
            self.start_gpu_mem_reser = torch.cuda.memory_reserved()
            torch.cuda.empty_cache()  # empty cache
            torch.cuda.reset_peak_memory_stats()

    def stop(self):
        super().stop()

    def step(self):
        # Stop torch profiler
        super().step()

        # Print elapsed time
        end_time_info = os.times()
        diff_time_info = {
            key: getattr(end_time_info, key) - value
            for key, value in self.time_info.items()
        }
        if self.extra:
            end_time = time.time()
            diff_time_info["precise_elapsed"] = end_time - self.start_time
            self.start_time = end_time
        self._log_stats("Elapsed Time", units="sec", base=1000, **diff_time_info)
        self.set_time_info(end_time_info)

        # Print memory usage
        end_memory_info = self.process.memory_info()
        diff_memory_info = {
            key: getattr(end_memory_info, key) - value
            for key, value in self.memory_info.items()
        }
        self._log_stats("Memory Usage", sort_keys=False, **diff_memory_info)
        self.set_memory_info(end_memory_info)

        # Calculate GPU memory usage if using CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.end_gpu_mem_alloc = torch.cuda.memory_allocated()
            self._log_stats(
                "GPU Allocated Memory",
                start=self.start_gpu_mem_alloc,
                end=self.end_gpu_mem_alloc,
                diff=self.end_gpu_mem_alloc - self.start_gpu_mem_alloc,
                peak=torch.cuda.max_memory_allocated(),
            )

            self.end_gpu_mem_reserved = torch.cuda.memory_reserved()
            self._log_stats(
                "GPU Reserved Memory",
                start=self.start_gpu_mem_reser,
                end=self.end_gpu_mem_reserved,
                diff=self.end_gpu_mem_reserved - self.start_gpu_mem_reser,
                peak=torch.cuda.max_memory_reserved(),
            )

            # self._log_func(torch.cuda.memory_summary())

        # Display torch profiler results
        # self._log_func(self.key_averages().table(sort_by="cpu_time_total", row_limit=10))


if __name__ == "__main__":
    import torchvision.models as tv_mdl
    from delegator import FunctionProfiler

    logging.basicConfig(
        level="DEBUG", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    profiler = TorchProfiler(profile_memory=True, with_flops=True, extra=True)
    profiler = FunctionProfiler(profiler=profiler)
    net = profiler(tv_mdl.resnet101().to(device="cuda"))
    loss_fn = profiler(torch.nn.MSELoss())

    # net = torch.nn.Linear(12, 2).to(device='cuda')
    inp1 = torch.rand(20, 3, 224, 224, device="cuda")
    inp2 = torch.rand(20, 3, 224, 224, device="cuda")
    trg = torch.rand(20, 1000, device="cuda")

    loss1 = loss_fn(net(inp1), trg)
