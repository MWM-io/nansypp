"""
A decorator that copies a function's docstring and signature to another.

Taken from https://stackoverflow.com/a/75898999/5798793.
Useful to enable typing of nn.Module().__call__() methods by copying
the relevant information from the nn.Module().forward() method.

Usage:

```
from typing import Tuple

from torch import Tensor

class ExampleModule(nn.Module):
    def forward(
        self, x: Tensor, y: Tensor, some_flag: boolean = True
    ) -> Tuple[Tensor, Tensor]:
        ...

    @copy_docstring_and_signature(forward)
    # mypy will complain that this declaration is untyped,
    # hence the type-system 
    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return super().__call__(*args, **kwargs)
```

In the context of nn.Modules' forward / __call__, an alternative
would be to explicitly declare the __call__() method and add type-hints
there copying the ones from the forward() method, but this has the
downside of necessitating manual maintenance to ensure that the
types in __call__() are always up-to-date if making modifications to
the arguments or return type of the forward() method.
This is discussed here:
https://github.com/pytorch/pytorch/issues/44605#issuecomment-692344479
"""

import sys
from functools import partial
from typing import Any, Callable, TypeVar, cast

from hydra.utils import instantiate

if sys.version_info < (3, 10):
    from typing_extensions import ParamSpec
else:
    from typing import ParamSpec


P = ParamSpec("P")
T = TypeVar("T")


def copy_docstring_and_signature(
    from_func: Callable[P, T]
) -> Callable[[Callable], Callable[P, T]]:
    """
    A typed implementation of functools.wraps.

    Taken from https://stackoverflow.com/a/75898999/5798793.
    """

    def decorator(func: Callable) -> Callable[P, T]:
        func.__doc__ = from_func.__doc__
        return cast(Callable[P, T], func)

    return decorator


def typed_instantiate(_: Callable[P, T], *args: Any, **kwargs: Any) -> Callable[P, T]:
    """Return a typed hydra instantiate lambda with the provided function's call signature"""
    return cast(Callable[P, T], partial(instantiate, *args, **kwargs))
