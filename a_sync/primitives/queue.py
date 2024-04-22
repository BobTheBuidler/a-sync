import asyncio
import functools
import heapq
import logging
import sys

from a_sync._typing import *

logger = logging.getLogger(__name__)

if sys.version_info < (3, 9):
    class _Queue(asyncio.Queue, Generic[T]):
        __slots__ = "_maxsize", "_loop", "_getters", "_putters", "_unfinished_tasks", "_finished"
else:
    class _Queue(asyncio.Queue[T]):
        __slots__ = "_maxsize", "_getters", "_putters", "_unfinished_tasks", "_finished"

class Queue(_Queue[T]):
    # for type hint support, no functional difference
    async def get(self) -> T:
        self._queue
        return await _Queue.get(self)
    def get_nowait(self) -> T:
        return _Queue.get_nowait(self)
    async def put(self, item: T) -> None:
        return _Queue.put(self, item)
    def put_nowait(self, item: T) -> None:
        return _Queue.put_nowait(self, item)
    
    async def get_all(self) -> List[T]:
        """returns 1 or more items"""
        try:
            return self.get_all_nowait()
        except asyncio.QueueEmpty:
            return [await self.get()]
    def get_all_nowait(self) -> List[T]:
        """returns 1 or more items, or raises asyncio.QueueEmpty"""
        values: List[T] = []
        while True:
            try:
                values.append(self.get_nowait())
            except asyncio.QueueEmpty as e:
                if not values:
                    raise asyncio.QueueEmpty from e
                return values
            
    async def get_multi(self, i: int, can_return_less: bool = False) -> List[T]:
        _validate_args(i, can_return_less)
        items = []
        while len(items) < i and not can_return_less:
            try:
                items.extend(self.get_multi_nowait(i - len(items), can_return_less=True))
            except asyncio.QueueEmpty:
                items = [await self.get()]
        return items
    def get_multi_nowait(self, i: int, can_return_less: bool = False) -> List[T]:
        """
        Just like `asyncio.Queue.get_nowait`, but will return `i` items instead of 1.
        Set `can_return_less` to True if you want to receive up to `i` items.
        """
        _validate_args(i, can_return_less)
        items = []
        for _ in range(i):
            try:
                items.append(self.get_nowait())
            except asyncio.QueueEmpty:
                if items and can_return_less:
                    return items
                # put these back in the queue since we didn't return them
                for value in items:
                    self.put_nowait(value)
                raise asyncio.QueueEmpty from None
        return items


class ProcessingQueue(_Queue[Tuple[P, "asyncio.Future[V]"]], Generic[P, V]):
    __slots__ = "func", "num_workers"
    def __init__(
        self, 
        func: Callable[P, Awaitable[V]], 
        num_workers: int, 
        *, return_data: bool = True, 
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        if sys.version_info < (3, 10):
            super().__init__(loop=loop)
        elif loop:
            raise NotImplementedError(f"You cannot pass a value for `loop` in python {sys.version_info}")
        else:
            super().__init__()
        self.func = func
        self.num_workers = num_workers
        self._no_futs = not return_data
    def __repr__(self) -> str:
        return f"<{type(self).__name__} func={self.func} num_workers={self.num_workers} pending={self._unfinished_tasks}>"
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> "asyncio.Future[V]":
        return self.put_nowait(*args, **kwargs)
    def __del__(self) -> None:
        if self._unfinished_tasks > 0:
            context = {
                'message': f'{self} was destroyed but has work pending!',
            }
            asyncio.get_event_loop().call_exception_handler(context)
    async def put(self, *args: P.args, **kwargs: P.kwargs) -> "asyncio.Future[V]":
        self._workers
        if self._no_futs:
            return await super().put((args, kwargs))
        fut = asyncio.get_event_loop().create_future()
        await super().put((args, kwargs, fut))
        return fut
    def put_nowait(self, *args: P.args, **kwargs: P.kwargs) -> "asyncio.Future[V]":
        self._workers
        if self._no_futs:
            return super().put_nowait((args, kwargs))
        fut = self._create_future()
        super().put_nowait((args, kwargs, fut))
        return fut
    def _create_future(self) -> "asyncio.Future[V]":
        return asyncio.get_event_loop().create_future()
    @functools.cached_property
    def _workers(self) -> "asyncio.Task[NoReturn]":
        from a_sync.task import create_task
        logger.debug("starting worker task for %s", self)
        task = create_task(asyncio.gather(*[self._worker_coro() for _ in range(self.num_workers)]), name=repr(self))
        task._log_destroy_pending = False
        return task
    async def _worker_coro(self) -> NoReturn:
        args: P.args
        kwargs: P.kwargs
        if self._no_futs:
            while True:
                try:
                    args, kwargs = await self.get()
                    await self.func(*args, **kwargs)
                except Exception as e:
                    logger.error("%s in worker for %s!", type(e).__name__, self)
                    logger.exception(e)
                self.task_done()
        else:
            fut: asyncio.Future[V]
            while True:
                try:
                    args, kwargs, fut = await self.get()
                    fut.set_result(await self.func(*args, **kwargs))
                except Exception as e:
                    try:
                        fut.set_result(e)
                    except UnboundLocalError as u:
                        logger.error("%s for %s is broken!!!", type(self).__name__, self.func)
                        if str(e) != "local variable 'fut' referenced before assignment":
                            logger.exception(u)
                            raise u
                        logger.exception(e)
                        raise e
                self.task_done()


def _validate_args(i: int, can_return_less: bool) -> None:
    if not isinstance(i, int):
        raise TypeError(f"`i` must be an integer greater than 1. You passed {i}")
    if not isinstance(can_return_less, bool):
        raise TypeError(f"`can_return_less` must be boolean. You passed {can_return_less}")
    if i <= 1:
        raise ValueError(f"`i` must be an integer greater than 1. You passed {i}")


class SmartFuture(asyncio.Future, Generic[T]):
    # classvar holds default value for instances
    _waiters: Set["asyncio.Task[T]"] = set()
    def __repr__(self):
        return f"<{type(self).__name__} waiters={self.num_waiters} {self._state}>"
    def __await__(self):
        logger.info("entering %s", self)
        if self.done():
            return self.result()  # May raise too.
        logger.info("awaiting %s", self)
        self._asyncio_future_blocking = True
        self._waiters.add(current_task := asyncio.current_task(self._loop))
        logger.info("%s waiters: %s", self, self._waiters)
        yield self  # This tells Task to wait for completion.
        self._waiters.remove(current_task)
        if not self.done():
            raise RuntimeError("await wasn't used with future")
        return self.result()  # May raise too.
    def __lt__(self, other: "SmartFuture") -> bool:
        """heap considers lower values as higher priority so a future with more waiters will be 'less than' a future with less waiters."""
        return self.num_waiters > other.num_waiters
    @property
    def num_waiters(self) -> int:
        return len(self._waiters)


class _PriorityQueueMixin(Generic[T]):
    def _init(self, maxsize):
        self._queue: List[T] = []
    def _put(self, item, heappush=heapq.heappush):
        heappush(self._queue, item)
    def _get(self, heappop=heapq.heappop):
        return heappop(self._queue)

class PriorityProcessingQueue(_PriorityQueueMixin[T], ProcessingQueue[T, V]):
    async def put(self, priority: Any, *args: P.args, **kwargs: P.kwargs) -> "asyncio.Future[V]":
        self._workers
        fut = asyncio.get_event_loop().create_future()
        await super().put(self, (priority, args, kwargs, fut))
        return fut
    def put_nowait(self, priority: Any, *args: P.args, **kwargs: P.kwargs) -> "asyncio.Future[V]":
        self._workers
        fut = self._create_future()
        super().put_nowait(self, (priority, args, kwargs, fut))
        return fut
    def _get(self, heappop=heapq.heappop):
        priority, args, kwargs, fut = heappop(self._queue)
        return args, kwargs, fut

class _VariablePriorityQueueMixin(_PriorityQueueMixin[T]):
    def _get(self, heapify=heapq.heapify, heappop=heapq.heappop):
        "Resort the heap to consider any changes in priorities and pop the smallest value"
        # resort the heap
        heapify(self._queue)
        # take the job with the most waiters
        return heappop(self._queue)
    def _create_future(self) -> "asyncio.Future[V]":
        return SmartFuture(loop=asyncio.get_event_loop())

class VariablePriorityQueue(_VariablePriorityQueueMixin[T], asyncio.PriorityQueue):
    """A PriorityQueue subclass that allows priorities to be updated (or computed) on the fly"""

class SmartProcessingQueue(_VariablePriorityQueueMixin[T], ProcessingQueue[Concatenate[T, P], V]):
    """A PriorityProcessingQueue subclass that will execute jobs with the most waiters first"""
    _no_futs = False
    def __init__(
        self, 
        func: Callable[Concatenate[T, P], Awaitable[V]], 
        num_workers: int, 
        *, 
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        super().__init__(func, num_workers, return_data=True, loop=loop)
    async def put(self, *args: P.args, **kwargs: P.kwargs) -> SmartFuture[V]:
        self._workers
        fut = asyncio.get_event_loop().create_future()
        await Queue.put(self, (fut, args, kwargs))
        return fut
    def put_nowait(self, *args: P.args, **kwargs: P.kwargs) -> SmartFuture[V]:
        self._workers
        fut = self._create_future()
        Queue.put_nowait(self, (fut, args, kwargs))
        return fut
    def _get(self):
        fut, args, kwargs = super()._get()
        return args, kwargs, fut
    async def _worker_coro(self) -> NoReturn:
        args: P.args
        kwargs: P.kwargs
        fut: asyncio.Future[V]
        while True:
            try:
                args, kwargs, fut = await self.get()
                fut.set_result(await self.func(*args, **kwargs))
            except Exception as e:
                try:
                    fut.set_result(e)
                except UnboundLocalError as u:
                    logger.error("%s for %s is broken!!!", type(self).__name__, self.func)
                    if str(e) != "local variable 'fut' referenced before assignment":
                        logger.exception(u)
                        raise u
                    logger.exception(e)
                    raise e
            self.task_done()
