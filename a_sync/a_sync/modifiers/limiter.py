# mypy: disable-error-code=valid-type
# mypy: disable-error-code=misc
import asyncio

from aiolimiter import AsyncLimiter

from a_sync import aliases, exceptions
from a_sync._typing import *


LimiterSpec = Union[int, AsyncLimiter]

@overload
def apply_rate_limit(
    runs_per_minute: int,
) -> AsyncDecorator[P, T]:
    """Decorator to apply a rate limit to an asynchronous function.

    Args:
        runs_per_minute: The number of allowed executions per minute.
    """
    
@overload
def apply_rate_limit(
    coro_fn: CoroFn[P, T],
    runs_per_minute: LimiterSpec,
) -> CoroFn[P, T]:
    """Decorator to apply a rate limit to an asynchronous function.

    Args:
        coro_fn: The coroutine function to be rate-limited.
        runs_per_minute: The number of allowed executions per minute or an AsyncLimiter instance.
    """
    
def apply_rate_limit(
    coro_fn: Optional[Union[CoroFn[P, T], int]] = None,
    runs_per_minute: Optional[LimiterSpec] = None,
) -> AsyncDecoratorOrCoroFn[P, T]:
    """Applies a rate limit to an asynchronous function.

    This function can be used as a decorator to limit the number of times
    an asynchronous function can be called per minute. It can be configured
    with either an integer specifying the number of runs per minute or an
    AsyncLimiter instance.

    Args:
        coro_fn: The coroutine function to be rate-limited. If an integer is provided, it is treated as runs per minute.
        runs_per_minute: The number of allowed executions per minute or an AsyncLimiter instance. If coro_fn is an integer, this should be None.

    Raises:
        TypeError: If 'runs_per_minute' is neither an integer nor an AsyncLimiter when 'coro_fn' is None.
        exceptions.FunctionNotAsync: If 'coro_fn' is not an asynchronous function.
    """
    # Parse Inputs
    if isinstance(coro_fn, (int, AsyncLimiter)):
        assert runs_per_minute is None
        runs_per_minute = coro_fn
        coro_fn = None

    elif coro_fn is None:
        if runs_per_minute is not None and not isinstance(runs_per_minute, (int, AsyncLimiter)):
            raise TypeError("'runs_per_minute' must be an integer.", runs_per_minute)

    elif not asyncio.iscoroutinefunction(coro_fn):
        raise exceptions.FunctionNotAsync(coro_fn)

    def rate_limit_decorator(coro_fn: CoroFn[P, T]) -> CoroFn[P, T]:
        limiter = (
            runs_per_minute
            if isinstance(runs_per_minute, AsyncLimiter)
            else AsyncLimiter(runs_per_minute) if runs_per_minute else aliases.dummy
        )

        async def rate_limit_wrap(*args: P.args, **kwargs: P.kwargs) -> T:
            async with limiter:  # type: ignore [attr-defined]
                return await coro_fn(*args, **kwargs)

        return rate_limit_wrap

    return rate_limit_decorator if coro_fn is None else rate_limit_decorator(coro_fn)