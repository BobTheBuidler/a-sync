"""
This module provides utility functions for handling and merging asynchronous iterators. It includes functions to 
exhaust async iterators, merge multiple async iterators into a single async iterator, and manage the processing 
flow of items in an asynchronous context.
"""

import asyncio
import asyncio.futures
import traceback
from logging import DEBUG, getLogger
from types import TracebackType

import a_sync.asyncio
from a_sync._typing import *
from a_sync.primitives.queue import Queue

logger = getLogger(__name__)


async def exhaust_iterator(
    iterator: AsyncIterator[T], *, queue: Optional[asyncio.Queue] = None
) -> None:
    """
    Asynchronously iterates over items from the given async iterator and optionally places them into a queue.

    This function is a utility to exhaust an async iterator, with an option to forward the iterated items to a provided queue-like object.
    The queue should have a `put_nowait` method. This is particularly useful when dealing with asynchronous operations that produce items
    to be consumed by other parts of an application, enabling a producer-consumer pattern.

    Args:
        iterator (AsyncIterator[T]): The async iterator to exhaust.
        queue (Optional[asyncio.Queue]): An optional queue-like object where iterated items will be placed.
            The queue should support the `put_nowait` method. If None, items are simply consumed.

    Example:
        >>> async def example():
        >>>     await exhaust_iterator(some_async_iterator, queue=my_queue)

    See Also:
        - :func:`exhaust_iterators`
        - :func:`as_yielded`
    """
    if queue is None:
        async for thing in iterator:
            pass
    elif logger.isEnabledFor(DEBUG):
        async for thing in iterator:
            logger._log(DEBUG, "putting %s from %s to queue %s", (thing, iterator, queue))
            queue.put_nowait(thing)
    else:
        async for thing in iterator:
            queue.put_nowait(thing)


async def exhaust_iterators(
    iterators, *, queue: Optional[asyncio.Queue] = None, join: bool = False
) -> None:
    """
    Asynchronously iterates over multiple async iterators concurrently and optionally places their items into a queue.

    This function leverages :func:`asyncio.gather` to concurrently exhaust multiple async iterators. It's useful in scenarios where items
    from multiple async sources need to be processed or collected together, supporting concurrent operations and efficient multitasking.

    Args:
        iterators: A sequence of async iterators to be exhausted concurrently.
        queue (Optional[Queue]): An optional queue-like object where items from all iterators will be placed. If None, items are simply consumed.
        join (Optional[bool]): If a queue was provided and join is True, this coroutine will continue to run until all queue items have been processed.

    Raises:
        ValueError: If `join` is True but no `queue` is provided.

    Example:
        >>> async def example():
        >>>     await exhaust_iterators([iterator1, iterator2], queue=my_queue, join=True)

    See Also:
        - :func:`exhaust_iterator`
        - :func:`as_yielded`
    """
    if queue is None and join:
        raise ValueError("You must provide a `queue` to use kwarg `join`")

    for x in await asyncio.gather(
        *[exhaust_iterator(iterator, queue=queue) for iterator in iterators],
        return_exceptions=True,
    ):
        if isinstance(x, Exception):
            # raise it with its original traceback instead of from here
            raise x.with_traceback(x.__traceback__)

    if queue is not None:
        queue.put_nowait(_Done())
        if join:
            await queue.join()


T0 = TypeVar("T0")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")
T6 = TypeVar("T6")
T7 = TypeVar("T7")
T8 = TypeVar("T8")
T9 = TypeVar("T9")


@overload
def as_yielded(*iterators: AsyncIterator[T]) -> AsyncIterator[T]: ...
@overload
def as_yielded(
    iterator0: AsyncIterator[T0],
    iterator1: AsyncIterator[T1],
    iterator2: AsyncIterator[T2],
    iterator3: AsyncIterator[T3],
    iterator4: AsyncIterator[T4],
    iterator5: AsyncIterator[T5],
    iterator6: AsyncIterator[T6],
    iterator7: AsyncIterator[T7],
    iterator8: AsyncIterator[T8],
    iterator9: AsyncIterator[T9],
) -> AsyncIterator[Union[T0, T1, T2, T3, T4, T5, T6, T7, T8, T9]]: ...
@overload
def as_yielded(
    iterator0: AsyncIterator[T0],
    iterator1: AsyncIterator[T1],
    iterator2: AsyncIterator[T2],
    iterator3: AsyncIterator[T3],
    iterator4: AsyncIterator[T4],
    iterator5: AsyncIterator[T5],
    iterator6: AsyncIterator[T6],
    iterator7: AsyncIterator[T7],
    iterator8: AsyncIterator[T8],
) -> AsyncIterator[Union[T0, T1, T2, T3, T4, T5, T6, T7, T8]]: ...
@overload
def as_yielded(
    iterator0: AsyncIterator[T0],
    iterator1: AsyncIterator[T1],
    iterator2: AsyncIterator[T2],
    iterator3: AsyncIterator[T3],
    iterator4: AsyncIterator[T4],
    iterator5: AsyncIterator[T5],
    iterator6: AsyncIterator[T6],
    iterator7: AsyncIterator[T7],
) -> AsyncIterator[Union[T0, T1, T2, T3, T4, T5, T6, T7]]: ...
@overload
def as_yielded(
    iterator0: AsyncIterator[T0],
    iterator1: AsyncIterator[T1],
    iterator2: AsyncIterator[T2],
    iterator3: AsyncIterator[T3],
    iterator4: AsyncIterator[T4],
    iterator5: AsyncIterator[T5],
    iterator6: AsyncIterator[T6],
) -> AsyncIterator[Union[T0, T1, T2, T3, T4, T5, T6]]: ...
@overload
def as_yielded(
    iterator0: AsyncIterator[T0],
    iterator1: AsyncIterator[T1],
    iterator2: AsyncIterator[T2],
    iterator3: AsyncIterator[T3],
    iterator4: AsyncIterator[T4],
    iterator5: AsyncIterator[T5],
) -> AsyncIterator[Union[T0, T1, T2, T3, T4, T5]]: ...
@overload
def as_yielded(
    iterator0: AsyncIterator[T0],
    iterator1: AsyncIterator[T1],
    iterator2: AsyncIterator[T2],
    iterator3: AsyncIterator[T3],
    iterator4: AsyncIterator[T4],
) -> AsyncIterator[Union[T0, T1, T2, T3, T4]]: ...
@overload
def as_yielded(
    iterator0: AsyncIterator[T0],
    iterator1: AsyncIterator[T1],
    iterator2: AsyncIterator[T2],
    iterator3: AsyncIterator[T3],
) -> AsyncIterator[Union[T0, T1, T2, T3]]: ...
@overload
def as_yielded(
    iterator0: AsyncIterator[T0],
    iterator1: AsyncIterator[T1],
    iterator2: AsyncIterator[T2],
) -> AsyncIterator[Union[T0, T1, T2]]: ...
@overload
def as_yielded(
    iterator0: AsyncIterator[T0], iterator1: AsyncIterator[T1]
) -> AsyncIterator[Union[T0, T1]]: ...
@overload
def as_yielded(
    iterator0: AsyncIterator[T0],
    iterator1: AsyncIterator[T1],
    iterator2: AsyncIterator[T2],
    *iterators: AsyncIterator[T],
) -> AsyncIterator[Union[T0, T1, T2, T]]: ...
async def as_yielded(*iterators: AsyncIterator[T]) -> AsyncIterator[T]:  # type: ignore [misc]
    """
    Merges multiple async iterators into a single async iterator that yields items as they become available from any of the source iterators.

    This function is designed to streamline the handling of multiple asynchronous data streams by consolidating them into a single asynchronous iteration context.
    It enables concurrent fetching and processing of items from multiple sources, improving efficiency and simplifying code structure when dealing with asynchronous operations.

    The merging process is facilitated by the :func:`exhaust_iterators` function, which concurrently processes the source iterators and places their items into a queue.
    This mechanism ensures that the merged stream of items is delivered in an order determined by the availability of items from the source iterators, rather than their original sequence.

    The function handles exceptions and ensures robustness and reliability by using asyncio tasks and queues. It manages edge cases such as early termination and exception management.
    The :class:`_Done` sentinel class is used internally to signal the completion of processing.

    Args:
        *iterators: Variable length list of :class:`~collections.abc.AsyncIterator` objects to be merged.

    Note:
        This implementation leverages asyncio tasks and queues to efficiently manage the asynchronous iteration and merging process.
        It handles edge cases such as early termination and exception management, ensuring robustness and reliability.
        The :class:`_Done` sentinel class is used internally to signal the completion of processing.

    Example:
        >>> async def example():
        >>>     async for item in as_yielded(iterator1, iterator2):
        >>>         print(item)

    See Also:
        - :func:`exhaust_iterator`
        - :func:`exhaust_iterators`
    """
    # hypothesis idea: _Done should never be exposed to user, works for all desired input types
    queue: Queue[Union[T, _Done]] = Queue()

    def _as_yielded_done_callback(t: asyncio.Task) -> None:
        if t.cancelled():
            return
        if e := t.exception():
            traceback.extract_stack
            traceback.clear_frames(e.__traceback__)
            queue.put_nowait(_Done(e))

    task = a_sync.asyncio.create_task(
        coro=exhaust_iterators(iterators, queue=queue, join=True),
        name=f"a_sync.as_yielded queue populating task for {iterators}",
    )

    task.add_done_callback(_as_yielded_done_callback)

    while not task.done():
        try:
            items = await queue.get_all()
        except asyncio.CancelledError:
            # cleanup lingering objects and reraise
            del task
            del queue
            raise
        for item in items:
            queue.task_done()
            if isinstance(item, _Done):
                assert queue.empty()
                await task
                del task
                del queue
                if item._exc:
                    raise type(item._exc)(*item._exc.args).with_traceback(
                        item._tb
                    ) from item._exc.__cause__
                return
            yield item

    # ensure it isn't done due to an internal exception
    await task


class _Done:
    """
    A sentinel class used to signal the completion of processing in the :func:`as_yielded` function.

    This class acts as a marker to indicate that all items have been processed and the asynchronous iteration can be concluded.
    It is used internally within the implementation of :func:`as_yielded` to efficiently manage the termination of the iteration process once all source iterators have been exhausted.

    Args:
        exc (Optional[Exception]): An optional exception to be associated with the completion.
    """

    def __init__(self, exc: Optional[Exception] = None) -> None:
        self._exc = exc

    @property
    def _tb(self) -> TracebackType:
        """Returns the traceback associated with the exception, if any."""
        return self._exc.__traceback__  # type: ignore [union-attr]


__all__ = ["as_yielded", "exhaust_iterator", "exhaust_iterators"]
