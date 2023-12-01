import torch
from torch.profiler import profile, ProfilerActivity, schedule
import types
import typing as t

from . import LOG_PATH


class TrainProfiler(object):
    def __init__(self, *args, enable: bool = False, **kwargs) -> None:
        super(TrainProfiler, self).__init__()
        self.__enabled = enable

        if not self.__enabled:
            self.__profiler = None
            return

        self.__profiler = profile(
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(LOG_PATH)),
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=1, active=1, repeat=1),
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
        )

    def step(self):
        if not self.__profiler:
            return
        self.__profiler.step()

    def __enter__(self):
        return self

    def __exit__(self,
                 _0: t.Optional[type[BaseException]],
                 _1: t.Optional[BaseException],
                 _2: t.Optional[types.TracebackType]) -> None:
        if not self.__profiler:
            return None
        self.__profiler.stop()
