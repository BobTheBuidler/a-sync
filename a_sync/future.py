# type: ignore [var-annotated]

"""
future.py module provides functionality for handling asynchronous futures, including a decorator for converting callables into ASyncFuture objects and utilities for managing asynchronous computations.

Functions:
    future(callable: Union[Callable[P, Awaitable[T]], Callable[P, T]] = None, **kwargs: Unpack[ModifierKwargs]) -> Callable[P, Union[T, "ASyncFuture[T]"]]: A decorator to convert a callable into an ASyncFuture.
    _gather_check_and_materialize(*things: Unpack[MaybeAwaitable[T]]) -> List[T]: Gathers and materializes a list of awaitable or non-awaitable items.
    _check_and_materialize(thing: T) -> T: Checks if an item is awaitable and materializes it.
    _materialize(meta: "ASyncFuture[T]") -> T: Materializes the result of an ASyncFuture.

Classes:
    ASyncFuture: Represents an asynchronous future result.
    _ASyncFutureWrappedFn: A callable class to wrap functions and return ASyncFuture objects.
    _ASyncFutureInstanceMethod: A class to handle instance methods wrapped as ASyncFuture.
"""

import asyncio
import concurrent.futures
from functools import partial, wraps
from inspect import isawaitable

from a_sync._typing import *


def future(
    callable: AnyFn[P, T] = None,
    **kwargs: Unpack[ModifierKwargs],
) -> Callable[P, Union[T, "ASyncFuture[T]"]]:
    """
    A decorator function to convert a callable into an ASyncFuture.

    Args:
        callable (Union[Callable[P, Awaitable[T]], Callable[P, T]], optional): The callable to convert. Defaults to None.
        **kwargs: Additional keyword arguments for the modifier.

    Returns:
        Callable[P, Union[T, "ASyncFuture[T]"]]: The wrapped callable returning an ASyncFuture.
    """
    return _ASyncFutureWrappedFn(callable, **kwargs)


async def _gather_check_and_materialize(*things: Unpack[MaybeAwaitable[T]]) -> List[T]:
    """
    Gathers and materializes a list of awaitable or non-awaitable items.

    Args:
        *things (Unpack[MaybeAwaitable[T]]): Items to gather and materialize.

    Returns:
        List[T]: A list of materialized items.
    """
    return await asyncio.gather(*[_check_and_materialize(thing) for thing in things])


async def _check_and_materialize(thing: T) -> T:
    """
    Checks if an item is awaitable and materializes it.

    Args:
        thing (T): The item to check and materialize.

    Returns:
        T: The materialized item.
    """
    return await thing if isawaitable(thing) else thing


def _materialize(meta: "ASyncFuture[T]") -> T:
    """
    Materializes the result of an ASyncFuture.

    Args:
        meta (ASyncFuture[T]): The ASyncFuture to materialize.

    Returns:
        T: The materialized result.

    Raises:
        RuntimeError: If the result is not set and the event loop is running.
    """
    try:
        return asyncio.get_event_loop().run_until_complete(meta)
    except RuntimeError as e:
        raise RuntimeError(
            f"{meta} result is not set and the event loop is running, you will need to await it first"
        ) from e


MetaNumeric = Union[
    Numeric, "ASyncFuture[int]", "ASyncFuture[float]", "ASyncFuture[Decimal]"
]


class ASyncFuture(concurrent.futures.Future, Awaitable[T]):
    """
    A class representing an asynchronous future result.
    """

    __slots__ = "__awaitable__", "__dependencies", "__dependants", "__task"

    def __init__(self, awaitable: Awaitable[T], dependencies: List["ASyncFuture"] = []) -> None:
        """
        Initializes an ASyncFuture with an awaitable and optional dependencies.

        Args:
            awaitable (Awaitable[T]): The awaitable object.
            dependencies (List[ASyncFuture], optional): A list of dependencies. Defaults to [].
        """
        self.__awaitable__ = awaitable
        """The awaitable object."""
        self.__dependencies = dependencies
        """A list of dependencies."""
        for dependency in dependencies:
            assert isinstance(dependency, ASyncFuture)
            dependency.__dependants.append(self)
        self.__dependants: List[ASyncFuture] = []
        """A list of dependants."""
        self.__task = None
        """The task associated with the awaitable."""
        super().__init__()

    def __hash__(self) -> int:
        return hash(self.__awaitable__)

    def __repr__(self) -> str:
        string = f"<{self.__class__.__name__} {self._state} for {self.__awaitable__}"
        if self.cancelled():
            pass
        elif self.done():
            string += (
                f" exception={self.exception()}"
                if self.exception()
                else f" result={super().result()}"
            )
        return string + ">"

    def __list_dependencies(self, other) -> List["ASyncFuture"]:
        """
        Lists dependencies for the ASyncFuture.

        Args:
            other: The other dependency to list.

        Returns:
            List[ASyncFuture]: A list of dependencies.
        """
        if isinstance(other, ASyncFuture):
            return [self, other]
        return [self]

    @property
    def result(self) -> Union[Callable[[], T], Any]:
        """
        If this future is not done, it will work like cf.Future.result. It will block, await the awaitable, and return the result when ready.
        If this future is done and the result has attribute `results`, will return `getattr(future_result, 'result')`
        If this future is done and the result does NOT have attribute `results`, will again work like cf.Future.result
        """
        if self.done():
            if hasattr(r := super().result(), "result"):
                # can be property, method, whatever. should work.
                return r.result
            # the result should be callable like an asyncio.Future
            return super().result
        return lambda: _materialize(self)

    def __getattr__(self, attr: str) -> Any:
        return getattr(_materialize(self), attr)

    def __getitem__(self, key) -> Any:
        return _materialize(self)[key]

    # NOTE: broken, do not use. I think
    def __setitem__(self, key, value) -> None:
        _materialize(self)[key] = value

    # not sure what to call these
    def __contains__(self, key: Any) -> bool:
        return _materialize(
            ASyncFuture(
                self.__contains(key), dependencies=self.__list_dependencies(key)
            )
        )

    def __await__(self) -> Generator[Any, None, T]:
        return self.__await().__await__()

    async def __await(self) -> T:
        if not self.done():
            self.set_result(await self.__task__)
        return self._result

    @property
    def __task__(self) -> "asyncio.Task[T]":
        if self.__task is None:
            self.__task = asyncio.create_task(self.__awaitable__)
        return self.__task

    def __iter__(self):
        return _materialize(self).__iter__()

    def __next__(self):
        return _materialize(self).__next__()

    def __enter__(self):
        return _materialize(self).__enter__()

    def __exit__(self, *args):
        return _materialize(self).__exit__(*args)

    @overload
    def __add__(self: "ASyncFuture[int]", other: int) -> "ASyncFuture[int]": ...
    @overload
    def __add__(self: "ASyncFuture[float]", other: float) -> "ASyncFuture[float]": ...
    @overload
    def __add__(self: "ASyncFuture[float]", other: int) -> "ASyncFuture[float]": ...
    @overload
    def __add__(self: "ASyncFuture[int]", other: float) -> "ASyncFuture[float]": ...
    @overload
    def __add__(
        self: "ASyncFuture[Decimal]", other: Decimal
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    def __add__(self: "ASyncFuture[Decimal]", other: int) -> "ASyncFuture[Decimal]": ...
    @overload
    def __add__(self: "ASyncFuture[int]", other: Decimal) -> "ASyncFuture[Decimal]": ...
    @overload
    def __add__(
        self: "ASyncFuture[int]", other: Awaitable[int]
    ) -> "ASyncFuture[int]": ...
    @overload
    def __add__(
        self: "ASyncFuture[float]", other: Awaitable[float]
    ) -> "ASyncFuture[float]": ...
    @overload
    def __add__(
        self: "ASyncFuture[float]", other: Awaitable[int]
    ) -> "ASyncFuture[float]": ...
    @overload
    def __add__(
        self: "ASyncFuture[int]", other: Awaitable[float]
    ) -> "ASyncFuture[float]": ...
    @overload
    def __add__(
        self: "ASyncFuture[Decimal]", other: Awaitable[Decimal]
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    def __add__(
        self: "ASyncFuture[Decimal]", other: Awaitable[int]
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    def __add__(
        self: "ASyncFuture[int]", other: Awaitable[Decimal]
    ) -> "ASyncFuture[Decimal]": ...
    def __add__(self, other: MetaNumeric) -> "ASyncFuture":
        return ASyncFuture(
            self.__add(other), dependencies=self.__list_dependencies(other)
        )

    @overload
    def __sub__(self: "ASyncFuture[int]", other: int) -> "ASyncFuture[int]": ...
    @overload
    def __sub__(self: "ASyncFuture[float]", other: float) -> "ASyncFuture[float]": ...
    @overload
    def __sub__(self: "ASyncFuture[float]", other: int) -> "ASyncFuture[float]": ...
    @overload
    def __sub__(self: "ASyncFuture[int]", other: float) -> "ASyncFuture[float]": ...
    @overload
    def __sub__(
        self: "ASyncFuture[Decimal]", other: Decimal
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    def __sub__(self: "ASyncFuture[Decimal]", other: int) -> "ASyncFuture[Decimal]": ...
    @overload
    def __sub__(self: "ASyncFuture[int]", other: Decimal) -> "ASyncFuture[Decimal]": ...
    @overload
    def __sub__(
        self: "ASyncFuture[int]", other: Awaitable[int]
    ) -> "ASyncFuture[int]": ...
    @overload
    def __sub__(
        self: "ASyncFuture[float]", other: Awaitable[float]
    ) -> "ASyncFuture[float]": ...
    @overload
    def __sub__(
        self: "ASyncFuture[float]", other: Awaitable[int]
    ) -> "ASyncFuture[float]": ...
    @overload
    def __sub__(
        self: "ASyncFuture[int]", other: Awaitable[float]
    ) -> "ASyncFuture[float]": ...
    @overload
    def __sub__(
        self: "ASyncFuture[Decimal]", other: Awaitable[Decimal]
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    def __sub__(
        self: "ASyncFuture[Decimal]", other: Awaitable[int]
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    def __sub__(
        self: "ASyncFuture[int]", other: Awaitable[Decimal]
    ) -> "ASyncFuture[Decimal]": ...
    def __sub__(self, other: MetaNumeric) -> "ASyncFuture":
        return ASyncFuture(
            self.__sub(other), dependencies=self.__list_dependencies(other)
        )

    def __mul__(self, other) -> "ASyncFuture":
        return ASyncFuture(
            self.__mul(other), dependencies=self.__list_dependencies(other)
        )

    def __pow__(self, other) -> "ASyncFuture":
        return ASyncFuture(
            self.__pow(other), dependencies=self.__list_dependencies(other)
        )

    def __truediv__(self, other) -> "ASyncFuture":
        return ASyncFuture(
            self.__truediv(other), dependencies=self.__list_dependencies(other)
        )

    def __floordiv__(self, other) -> "ASyncFuture":
        return ASyncFuture(
            self.__floordiv(other), dependencies=self.__list_dependencies(other)
        )

    def __pow__(self, other) -> "ASyncFuture":
        return ASyncFuture(
            self.__pow(other), dependencies=self.__list_dependencies(other)
        )

    @overload
    def __radd__(self: "ASyncFuture[int]", other: int) -> "ASyncFuture[int]": ...
    @overload
    def __radd__(self: "ASyncFuture[float]", other: float) -> "ASyncFuture[float]": ...
    @overload
    def __radd__(self: "ASyncFuture[float]", other: int) -> "ASyncFuture[float]": ...
    @overload
    def __radd__(self: "ASyncFuture[int]", other: float) -> "ASyncFuture[float]": ...
    @overload
    def __radd__(
        self: "ASyncFuture[Decimal]", other: Decimal
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    def __radd__(
        self: "ASyncFuture[Decimal]", other: int
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    def __radd__(
        self: "ASyncFuture[int]", other: Decimal
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    def __radd__(
        self: "ASyncFuture[int]", other: Awaitable[int]
    ) -> "ASyncFuture[int]": ...
    @overload
    def __radd__(
        self: "ASyncFuture[float]", other: Awaitable[float]
    ) -> "ASyncFuture[float]": ...
    @overload
    def __radd__(
        self: "ASyncFuture[float]", other: Awaitable[int]
    ) -> "ASyncFuture[float]": ...
    @overload
    def __radd__(
        self: "ASyncFuture[int]", other: Awaitable[float]
    ) -> "ASyncFuture[float]": ...
    @overload
    def __radd__(
        self: "ASyncFuture[Decimal]", other: Awaitable[Decimal]
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    def __radd__(
        self: "ASyncFuture[Decimal]", other: Awaitable[int]
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    def __radd__(
        self: "ASyncFuture[int]", other: Awaitable[Decimal]
    ) -> "ASyncFuture[Decimal]": ...
    def __radd__(self, other) -> "ASyncFuture":
        return ASyncFuture(
            self.__radd(other), dependencies=self.__list_dependencies(other)
        )

    @overload
    def __rsub__(self: "ASyncFuture[int]", other: int) -> "ASyncFuture[int]": ...
    @overload
    def __rsub__(self: "ASyncFuture[float]", other: float) -> "ASyncFuture[float]": ...
    @overload
    def __rsub__(self: "ASyncFuture[float]", other: int) -> "ASyncFuture[float]": ...
    @overload
    def __rsub__(self: "ASyncFuture[int]", other: float) -> "ASyncFuture[float]": ...
    @overload
    def __rsub__(
        self: "ASyncFuture[Decimal]", other: Decimal
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    def __rsub__(
        self: "ASyncFuture[Decimal]", other: int
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    def __rsub__(
        self: "ASyncFuture[int]", other: Decimal
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    def __rsub__(
        self: "ASyncFuture[int]", other: Awaitable[int]
    ) -> "ASyncFuture[int]": ...
    @overload
    def __rsub__(
        self: "ASyncFuture[float]", other: Awaitable[float]
    ) -> "ASyncFuture[float]": ...
    @overload
    def __rsub__(
        self: "ASyncFuture[float]", other: Awaitable[int]
    ) -> "ASyncFuture[float]": ...
    @overload
    def __rsub__(
        self: "ASyncFuture[int]", other: Awaitable[float]
    ) -> "ASyncFuture[float]": ...
    @overload
    def __rsub__(
        self: "ASyncFuture[Decimal]", other: Awaitable[Decimal]
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    def __rsub__(
        self: "ASyncFuture[Decimal]", other: Awaitable[int]
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    def __rsub__(
        self: "ASyncFuture[int]", other: Awaitable[Decimal]
    ) -> "ASyncFuture[Decimal]": ...
    def __rsub__(self, other) -> "ASyncFuture":
        return ASyncFuture(
            self.__rsub(other), dependencies=self.__list_dependencies(other)
        )

    def __rmul__(self, other) -> "ASyncFuture":
        return ASyncFuture(
            self.__rmul(other), dependencies=self.__list_dependencies(other)
        )

    def __rtruediv__(self, other) -> "ASyncFuture":
        return ASyncFuture(
            self.__rtruediv(other), dependencies=self.__list_dependencies(other)
        )

    def __rfloordiv__(self, other) -> "ASyncFuture":
        return ASyncFuture(
            self.__rfloordiv(other), dependencies=self.__list_dependencies(other)
        )

    def __rpow__(self, other) -> "ASyncFuture":
        return ASyncFuture(
            self.__rpow(other), dependencies=self.__list_dependencies(other)
        )

    def __eq__(self, other) -> "ASyncFuture":
        return bool(
            ASyncFuture(self.__eq(other), dependencies=self.__list_dependencies(other))
        )

    def __gt__(self, other) -> "ASyncFuture":
        return ASyncFuture(
            self.__gt(other), dependencies=self.__list_dependencies(other)
        )

    def __ge__(self, other) -> "ASyncFuture":
        return ASyncFuture(
            self.__ge(other), dependencies=self.__list_dependencies(other)
        )

    def __lt__(self, other) -> "ASyncFuture":
        return ASyncFuture(
            self.__lt(other), dependencies=self.__list_dependencies(other)
        )

    def __le__(self, other) -> "ASyncFuture":
        return ASyncFuture(
            self.__le(other), dependencies=self.__list_dependencies(other)
        )

    # Maths

    @overload
    async def __add(self: "ASyncFuture[int]", other: int) -> "ASyncFuture[int]": ...
    @overload
    async def __add(
        self: "ASyncFuture[float]", other: float
    ) -> "ASyncFuture[float]": ...
    @overload
    async def __add(self: "ASyncFuture[float]", other: int) -> "ASyncFuture[float]": ...
    @overload
    async def __add(self: "ASyncFuture[int]", other: float) -> "ASyncFuture[float]": ...
    @overload
    async def __add(
        self: "ASyncFuture[Decimal]", other: Decimal
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    async def __add(
        self: "ASyncFuture[Decimal]", other: int
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    async def __add(
        self: "ASyncFuture[int]", other: Decimal
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    async def __add(
        self: "ASyncFuture[int]", other: Awaitable[int]
    ) -> "ASyncFuture[int]": ...
    @overload
    async def __add(
        self: "ASyncFuture[float]", other: Awaitable[float]
    ) -> "ASyncFuture[float]": ...
    @overload
    async def __add(
        self: "ASyncFuture[float]", other: Awaitable[int]
    ) -> "ASyncFuture[float]": ...
    @overload
    async def __add(
        self: "ASyncFuture[int]", other: Awaitable[float]
    ) -> "ASyncFuture[float]": ...
    @overload
    async def __add(
        self: "ASyncFuture[Decimal]", other: Awaitable[Decimal]
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    async def __add(
        self: "ASyncFuture[Decimal]", other: Awaitable[int]
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    async def __add(
        self: "ASyncFuture[int]", other: Awaitable[Decimal]
    ) -> "ASyncFuture[Decimal]": ...
    async def __add(self, other) -> "Any":
        a, b = await _gather_check_and_materialize(self, other)
        return a + b

    @overload
    async def __sub(self: "ASyncFuture[int]", other: int) -> "ASyncFuture[int]": ...
    @overload
    async def __sub(
        self: "ASyncFuture[float]", other: float
    ) -> "ASyncFuture[float]": ...
    @overload
    async def __sub(self: "ASyncFuture[float]", other: int) -> "ASyncFuture[float]": ...
    @overload
    async def __sub(self: "ASyncFuture[int]", other: float) -> "ASyncFuture[float]": ...
    @overload
    async def __sub(
        self: "ASyncFuture[Decimal]", other: Decimal
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    async def __sub(
        self: "ASyncFuture[Decimal]", other: int
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    async def __sub(
        self: "ASyncFuture[int]", other: Decimal
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    async def __sub(
        self: "ASyncFuture[int]", other: Awaitable[int]
    ) -> "ASyncFuture[int]": ...
    @overload
    async def __sub(
        self: "ASyncFuture[float]", other: Awaitable[float]
    ) -> "ASyncFuture[float]": ...
    @overload
    async def __sub(
        self: "ASyncFuture[float]", other: Awaitable[int]
    ) -> "ASyncFuture[float]": ...
    @overload
    async def __sub(
        self: "ASyncFuture[int]", other: Awaitable[float]
    ) -> "ASyncFuture[float]": ...
    @overload
    async def __sub(
        self: "ASyncFuture[Decimal]", other: Awaitable[Decimal]
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    async def __sub(
        self: "ASyncFuture[Decimal]", other: Awaitable[int]
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    async def __sub(
        self: "ASyncFuture[int]", other: Awaitable[Decimal]
    ) -> "ASyncFuture[Decimal]": ...
    async def __sub(self, other) -> "Any":
        a, b = await _gather_check_and_materialize(self, other)
        return a - b

    async def __mul(self, other) -> "Any":
        a, b = await _gather_check_and_materialize(self, other)
        return a * b

    async def __truediv(self, other) -> "Any":
        a, b = await _gather_check_and_materialize(self, other)
        return a / b

    async def __floordiv(self, other) -> "Any":
        a, b = await _gather_check_and_materialize(self, other)
        return a // b

    async def __pow(self, other) -> "Any":
        a, b = await _gather_check_and_materialize(self, other)
        return a**b

    # rMaths
    @overload
    async def __radd(self: "ASyncFuture[int]", other: int) -> "ASyncFuture[int]": ...
    @overload
    async def __radd(
        self: "ASyncFuture[float]", other: float
    ) -> "ASyncFuture[float]": ...
    @overload
    async def __radd(
        self: "ASyncFuture[float]", other: int
    ) -> "ASyncFuture[float]": ...
    @overload
    async def __radd(
        self: "ASyncFuture[int]", other: float
    ) -> "ASyncFuture[float]": ...
    @overload
    async def __radd(
        self: "ASyncFuture[Decimal]", other: Decimal
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    async def __radd(
        self: "ASyncFuture[Decimal]", other: int
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    async def __radd(
        self: "ASyncFuture[int]", other: Decimal
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    async def __radd(
        self: "ASyncFuture[int]", other: Awaitable[int]
    ) -> "ASyncFuture[int]": ...
    @overload
    async def __radd(
        self: "ASyncFuture[float]", other: Awaitable[float]
    ) -> "ASyncFuture[float]": ...
    @overload
    async def __radd(
        self: "ASyncFuture[float]", other: Awaitable[int]
    ) -> "ASyncFuture[float]": ...
    @overload
    async def __radd(
        self: "ASyncFuture[int]", other: Awaitable[float]
    ) -> "ASyncFuture[float]": ...
    @overload
    async def __radd(
        self: "ASyncFuture[Decimal]", other: Awaitable[Decimal]
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    async def __radd(
        self: "ASyncFuture[Decimal]", other: Awaitable[int]
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    async def __radd(
        self: "ASyncFuture[int]", other: Awaitable[Decimal]
    ) -> "ASyncFuture[Decimal]": ...
    async def __radd(self, other) -> "Any":
        a, b = await _gather_check_and_materialize(other, self)
        return a + b

    @overload
    async def __rsub(self: "ASyncFuture[int]", other: int) -> "ASyncFuture[int]": ...
    @overload
    async def __rsub(
        self: "ASyncFuture[float]", other: float
    ) -> "ASyncFuture[float]": ...
    @overload
    async def __rsub(
        self: "ASyncFuture[float]", other: int
    ) -> "ASyncFuture[float]": ...
    @overload
    async def __rsub(
        self: "ASyncFuture[int]", other: float
    ) -> "ASyncFuture[float]": ...
    @overload
    async def __rsub(
        self: "ASyncFuture[Decimal]", other: Decimal
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    async def __rsub(
        self: "ASyncFuture[Decimal]", other: int
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    async def __rsub(
        self: "ASyncFuture[int]", other: Decimal
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    async def __rsub(
        self: "ASyncFuture[int]", other: Awaitable[int]
    ) -> "ASyncFuture[int]": ...
    @overload
    async def __rsub(
        self: "ASyncFuture[float]", other: Awaitable[float]
    ) -> "ASyncFuture[float]": ...
    @overload
    async def __rsub(
        self: "ASyncFuture[float]", other: Awaitable[int]
    ) -> "ASyncFuture[float]": ...
    @overload
    async def __rsub(
        self: "ASyncFuture[int]", other: Awaitable[float]
    ) -> "ASyncFuture[float]": ...
    @overload
    async def __rsub(
        self: "ASyncFuture[Decimal]", other: Awaitable[Decimal]
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    async def __rsub(
        self: "ASyncFuture[Decimal]", other: Awaitable[int]
    ) -> "ASyncFuture[Decimal]": ...
    @overload
    async def __rsub(
        self: "ASyncFuture[int]", other: Awaitable[Decimal]
    ) -> "ASyncFuture[Decimal]": ...
    async def __rsub(self, other) -> "Any":
        a, b = await _gather_check_and_materialize(other, self)
        return a - b

    async def __rmul(self, other) -> "Any":
        a, b = await _gather_check_and_materialize(other, self)
        return a * b

    async def __rtruediv(self, other) -> "Any":
        a, b = await _gather_check_and_materialize(other, self)
        return a / b

    async def __rfloordiv(self, other) -> "Any":
        a, b = await _gather_check_and_materialize(other, self)
        return a // b

    async def __rpow(self, other) -> "Any":
        a, b = await _gather_check_and_materialize(other, self)
        return a**b

    async def __iadd(self, other) -> "Any":
        a, b = await _gather_check_and_materialize(self, other)
        self._result = a + b
        return self._result

    async def __isub(self, other) -> "Any":
        a, b = await _gather_check_and_materialize(self, other)
        self._result = a - b
        return self._result

    async def __imul(self, other) -> "Any":
        a, b = await _gather_check_and_materialize(self, other)
        self._result = a * b
        return self._result

    async def __itruediv(self, other) -> "Any":
        a, b = await _gather_check_and_materialize(self, other)
        self._result = a / b
        return self._result

    async def __ifloordiv(self, other) -> "Any":
        a, b = await _gather_check_and_materialize(self, other)
        self._result = a // b
        return self._result

    async def __ipow(self, other) -> "Any":
        a, b = await _gather_check_and_materialize(self, other)
        self._result = a**b
        return self._result

    # Comparisons
    async def __eq(self, other) -> bool:
        a, b = await _gather_check_and_materialize(self, other)
        return a == b

    async def __gt(self, other) -> bool:
        a, b = await _gather_check_and_materialize(self, other)
        return a > b

    async def __ge(self, other) -> bool:
        a, b = await _gather_check_and_materialize(self, other)
        return a >= b

    async def __lt(self, other) -> bool:
        a, b = await _gather_check_and_materialize(self, other)
        return a < b

    async def __le(self, other) -> bool:
        a, b = await _gather_check_and_materialize(self, other)
        return a <= b

    # not sure what to call these
    async def __contains(self, item: Any) -> bool:
        _self, _item = await _gather_check_and_materialize(self, item)
        return _item in _self

    # conversion
    # NOTE: We aren't allowed to return ASyncFutures here :(
    def __bool__(self) -> bool:
        return bool(_materialize(self))

    def __bytes__(self) -> bytes:
        return bytes(_materialize(self))

    def __str__(self) -> str:
        return str(_materialize(self))

    def __int__(self) -> int:
        return int(_materialize(self))

    def __float__(self) -> float:
        return float(_materialize(self))

    # WIP internals

    @property
    def __dependants__(self) -> Set["ASyncFuture"]:
        """
        Returns the set of dependants for this ASyncFuture.

        Returns:
            Set[ASyncFuture]: The set of dependant futures.
        """
        dependants = set()
        for dep in self.__dependants:
            dependants.add(dep)
            dependants.union(dep.__dependants__)
        return dependants

    @property
    def __dependencies__(self) -> Set["ASyncFuture"]:
        """
        Returns the set of dependencies for this ASyncFuture.

        Returns:
            Set[ASyncFuture]: The set of dependencies.
        """
        dependencies = set()
        for dep in self.__dependencies:
            dependencies.add(dep)
            dependencies.union(dep.__dependencies__)
        return dependencies

    def __sizeof__(self) -> int:
        if isinstance(self.__awaitable__, Coroutine):
            return sum(
                sys.getsizeof(v) for v in self.__awaitable__.cr_frame.f_locals.values()
            )
        elif isinstance(self.__awaitable__, asyncio.Future):
            raise NotImplementedError
        raise NotImplementedError


@final
class _ASyncFutureWrappedFn(Callable[P, ASyncFuture[T]]):
    """
    A callable class to wrap functions and return ASyncFuture objects.

    Attributes:
        callable (Union[Callable[P, Awaitable[T]], Callable[P, T]]): The callable function.
        wrapped (Callable[P, ASyncFuture[T]]): The wrapped function returning ASyncFuture.
        _callable_name (str): The name of the callable function.
    """
    __slots__ = "callable", "wrapped", "_callable_name"

    def __init__(
        self,
        callable: AnyFn[P, T] = None,
        **kwargs: Unpack[ModifierKwargs],
    ):
        from a_sync import a_sync

        if callable:
            self.callable = callable
            self._callable_name = callable.__name__
            a_sync_callable = a_sync(callable, default="async", **kwargs)

            @wraps(callable)
            def future_wrap(*args: P.args, **kwargs: P.kwargs) -> "ASyncFuture[T]":
                return ASyncFuture(a_sync_callable(*args, **kwargs, sync=False))

            self.wrapped = future_wrap
        else:
            self.wrapped = partial(_ASyncFutureWrappedFn, **kwargs)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> ASyncFuture[T]:
        return self.wrapped(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.callable}>"

    def __get__(
        self, instance: I, owner: Type[I]
    ) -> Union[Self, "_ASyncFutureInstanceMethod[I, P, T]"]:
        if owner is None:
            return self
        else:
            return _ASyncFutureInstanceMethod(self, instance)


@final
class _ASyncFutureInstanceMethod(Generic[I, P, T]):
    # NOTE: probably could just replace this with functools.partial
    """
    A class to handle instance methods wrapped as ASyncFuture.
    """

    __module__: str
    """The module name of the wrapper."""

    __name__: str
    """The name of the wrapper."""

    __qualname__: str
    """The qualified name of the wrapper."""

    __doc__: Optional[str]
    """The docstring of the wrapper."""

    __annotations__: Dict[str, Any]
    """The annotations of the wrapper."""

    __instance: I
    """The instance to which the method is bound."""

    __wrapper: _ASyncFutureWrappedFn[P, T]
    """The wrapper function."""
    
    def __init__(
        self,
        wrapper: _ASyncFutureWrappedFn[P, T],
        instance: I,
    ) -> None:
        try:
            self.__module__ = wrapper.__module__
        except AttributeError:
            pass
        try:
            self.__name__ = wrapper.__name__
        except AttributeError:
            pass
        try:
            self.__qualname__ = wrapper.__qualname__
        except AttributeError:
            pass
        try:
            self.__doc__ = wrapper.__doc__
        except AttributeError:
            pass
        try:
            self.__annotations__ = wrapper.__annotations__
        except AttributeError:
            pass
        try:
            self.__dict__.update(wrapper.__dict__)
        except AttributeError:
            pass
        self.__instance = instance
        self.__wrapper = wrapper
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} for {self.__wrapper.callable} bound to {self.__instance}>"

    def __call__(self, /, *fn_args: P.args, **fn_kwargs: P.kwargs) -> T:
        return self.__wrapper(self.__instance, *fn_args, **fn_kwargs)


__all__ = ["future", "ASyncFuture"]
