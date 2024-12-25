import inspect
import logging
import typing
from functools import partial
from functools import wraps
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Union

from pydantic import BaseModel
from pydantic import InstanceOf
from pydantic import PrivateAttr
from typing_extensions import Callable

from .profiler import BasicProfiler
from .profiler import TorchProfiler


class ContextProfiler(BaseModel):
    profiler: Union[BasicProfiler, InstanceOf[TorchProfiler]]

    def __enter__(self) -> Union[BasicProfiler, TorchProfiler]:
        if isinstance(self.profiler, TorchProfiler):
            return self.profiler.__enter__()
        else:
            self.profiler.start()
            return self

    def step(self):
        self.profiler.step()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if isinstance(self.profiler, TorchProfiler):
            return self.profiler.__exit__(exc_type, exc_val, exc_tb)
        else:
            print("profiled")
            self.profiler.stop()


class FunctionProfiler(BaseModel):
    profiler: Union[BasicProfiler, InstanceOf[TorchProfiler]]

    def __call__(self, fn: typing.Callable):

        if isinstance(fn, partial):
            name = fn.func.__name__
            func = fn.func
            wrap = "partial"
        elif inspect.isfunction(fn):
            name = fn.__name__
            func = fn
            wrap = "function"
        elif inspect.isclass(fn):
            name = fn.__name__
            func = fn
            wrap = "class"
        elif hasattr(fn, "__call__"):
            name = fn.__class__.__name__
            func = fn.__class__.__call__
            wrap = "object"
        else:
            raise TypeError(f"Unknow type {fn}")

        if isinstance(self.profiler, TorchProfiler):
            prof = self.profiler.__enter__()

            @wraps(func)
            def wrapper(*args, **kwds):
                prof._log_func("profiling " + name)
                res = func(*args, **kwds)
                prof.step()
                return res

            setattr(wrapper, "__prof__", prof)
        else:
            prof = self.profiler.__enter__()

            @wraps(func)
            def wrapper(*args, **kwds):
                prof.log("profiling " + name)
                ret = func(*args, **kwds)
                prof.step()
                return ret

            setattr(wrapper, "__prof__", prof)

        if wrap == "object":
            fn.__class__.__call__ = wrapper
            return fn
        elif wrap in ["partial", "function"]:
            return wrapper
        else:
            fn.__init__ = wrapper
            return fn

    def stop_profiler(self, fn: Callable):
        if isinstance(fn, partial):
            func = fn.func
            wrap = "partial"
        elif inspect.isfunction(fn):
            func = fn
            wrap = "function"
        elif inspect.isclass(fn):
            func = fn.__init__
            wrap = "class"
        elif hasattr(fn, "__call__"):
            func = fn.__class__.__call__
            wrap = "object"
        else:
            raise TypeError(f"Unknow type {fn}")

        if hasattr(func, "__prof__"):
            getattr(func, "__prof__").stop()
        else:
            raise TypeError(f"Unprofiled callable {fn}. Must be profiled first!")

    def start_profiler(self, fn: Callable):
        if isinstance(fn, partial):
            func = fn.func
            wrap = "partial"
        elif inspect.isfunction(fn):
            func = fn
            wrap = "function"
        elif inspect.isclass(fn):
            func = fn.__init__
            wrap = "class"
        elif hasattr(fn, "__call__"):
            func = fn.__class__.__call__
            wrap = "object"
        else:
            raise TypeError(f"Unknow type {fn}")

        if hasattr(func, "__prof__"):
            getattr(func, "__prof__").start()
        else:
            raise TypeError(f"Unprofiled callable {fn}. Must be profiled first!")


class IteratorProfiler(BaseModel):
    iterator: Any
    kwds: Dict[str, Any]

    def __init__(self, /, iterator: Iterable[Any], **kwds):
        super().__init__(iterator=iterator, kwds=kwds)

    class InnerIteratorProfiler(BaseModel):
        iterable: Iterable[Any]
        profiler: Union[BasicProfiler, InstanceOf[TorchProfiler]]
        _iterable = PrivateAttr()

        def __iter__(self):
            if isinstance(self.profiler, TorchProfiler):
                return self.profiler.__enter__()
            else:
                self._iterable = iter(self.iterable)
                return self

        def __next__(self):
            if isinstance(self.profiler, TorchProfiler):
                try:
                    elem = next(self._iterable)
                    self.profiler.step()
                    return elem
                except StopIteration:
                    self.profiler.__exit__(None, None, None)
            else:
                self.profiler.start()
                res = next(self._iterable)
                self.profiler.stop()
                return res

    def __iter__(self):
        yield from self.__class__.InnerIteratorProfiler(
            iterable=iter(self.iterator),
            logger=logging.getLogger(__name__ + "." + self.__class__.__name__),
            **self.kwds,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    prof = BasicProfiler(
        name="simple function", with_memory=True, with_time=True, extra=True
    )
    fn = FunctionProfiler(profiler=prof)

    # @fn
    def foo():
        a = 0
        for i in range(100_000):
            a += i
        return 2

    foo()
    """
    # tracker = BasicProfiler('foo', with_memory=True, with_time=True, extra=True)
    # tracker.start()
    # tracker.stop()

    """
    with ContextProfiler(profiler=prof):
        foo()

    # x = BasicProfiler(
    #     name="iterator", iterator=range(3)
    # )

    x = IteratorProfiler(iterator=range(4), profiler=prof)

    for i in x:
        foo()
        print("inner 1")

    for i in x:
        foo()
        print("inner 2")

    for i in x:
        foo()
        foo()
        foo()
        print("inner 3")
