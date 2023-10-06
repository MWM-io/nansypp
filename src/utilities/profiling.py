import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator, Literal, Optional, TypeVar, Union

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.timer import Timer

from src.utilities.types import P

DO_PROFILING = False


def synchronize_and_time() -> float:
    torch.cuda.synchronize()
    return time.time()


@contextmanager
def cuda_synchronized_profile_context_manager(
    do_profiling: bool = DO_PROFILING,
    prefix: str = "",
    time_unit: Union[Literal["s"], Literal["ms"]] = "s",
) -> Generator[None, None, None]:
    do_cuda_synchronize = True
    valid_time_units = ["s", "ms"]
    assert (
        valid_time_units.count(time_unit) > 0
    ), f"Unrecognized time_unit option {time_unit}, valid values are: {valid_time_units}"

    if do_profiling and torch.cuda.is_initialized():
        if do_cuda_synchronize:
            torch.cuda.synchronize()
        start_time = time.perf_counter()
    else:
        start_time = None
    try:
        yield
    finally:
        if do_profiling and torch.cuda.is_initialized():
            assert start_time is not None
            if do_cuda_synchronize:
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            duration_seconds = end_time - start_time
            print(
                prefix,
                f"{duration_seconds * (1000 if time_unit == 'ms' else 1):.3f}{time_unit}",
            )


T = TypeVar("T")

TIMER_PREFIX_WIDTH = 40


def cuda_synchronized_timer(
    do_profiling: bool,
    prefix: str = "",
    time_unit: Union[Literal["s"], Literal["ms"]] = "ms",
    before_timing: Callable[[], None] = torch.cuda.synchronize,
    format_specifier: str = "8.2f",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start: Optional[float] = None
            if do_profiling and torch.cuda.is_initialized():
                before_timing()
                start = time.perf_counter()

            func_value = func(*args, **kwargs)

            if start is not None and do_profiling and torch.cuda.is_initialized():
                before_timing()
                run_time_seconds = time.perf_counter() - start
                full_prefix = (
                    f"[{prefix}{'::' if len(prefix)>0 else ''}{func.__name__}] "
                )
                print(
                    full_prefix.ljust(TIMER_PREFIX_WIDTH)
                    + f"{run_time_seconds * (1000 if time_unit == 'ms' else 1):{format_specifier}}{time_unit}",
                )
            return func_value

        return wrapper

    return decorator


class LightningBatchTimer(Timer):
    """A Pytorch-Lightning timer Callback to measure the duration of epochs and batches"""

    batch_time: Optional[float] = None
    time_elapsed_epoch: float = 0
    average_batch_duration: float = 0
    remaining_warmup_batches: int = 0
    num_warmup_batches: int

    def __init__(
        self,
        num_batches_per_epoch: Optional[int],
        num_warmup_batches: int,
        *args: Any,
    ) -> None:
        super().__init__(*args)
        self.num_batches_per_epoch = num_batches_per_epoch
        self.num_warmup_batches = num_warmup_batches
        self.num_effective_batches_per_epoch = (
            (self.num_batches_per_epoch - self.num_warmup_batches)
            if self.num_batches_per_epoch is not None
            else None
        )

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        torch.cuda.synchronize()
        self.time_elapsed_epoch = self.time_elapsed()
        self.average_batch_duration = 0
        self.remaining_warmup_batches = self.num_warmup_batches

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        prev_time = self.batch_time
        torch.cuda.synchronize()
        new_time = self.time_elapsed()
        if prev_time is not None and self.num_effective_batches_per_epoch is not None:
            batch_duration = new_time - prev_time
            # print(f"[MyTimer] Batch: {batch_duration:.3f}s")
            if self.remaining_warmup_batches == 0:
                self.average_batch_duration += batch_duration / (
                    self.num_effective_batches_per_epoch
                )
        self.batch_time = new_time
        self.remaining_warmup_batches = max(0, self.remaining_warmup_batches - 1)

    def on_train_epoch_end(self, trainer: Trainer, *args: Any, **kwargs: Any) -> None:
        torch.cuda.synchronize()
        print(
            f"[MyTimer] Time elapsed on epoch: {self.time_elapsed() - self.time_elapsed_epoch:.3f}s",
            flush=True,
        )
        print(
            f"[MyTimer] Average batch: {self.average_batch_duration:.3f}s",
            flush=True,
        )
