"""
This module provides several executor classes that facilitate running synchronous functions asynchronously using `asyncio`.

With these executors, you can run sync functions in your executor with `await executor.run(fn, *args, **kwargs)`.
The `executor.submit(fn, *args, **kwargs)` method works similarly to the `concurrent.futures` implementation but
returns an `asyncio.Future` instead of a `concurrent.futures.Future`.

Executor Classes:
    - :class:`AsyncProcessPoolExecutor`: A process pool executor providing asynchronous run and submit methods, with support for synchronous mode
    - :class:`AsyncThreadPoolExecutor`: A thread pool executor providing asynchronous run and submit methods, with support for synchronous mode
    - :class:`PruningThreadPoolExecutor`: An :class:`AsyncThreadPoolExecutor` that prunes inactive threads after a timeout, ensuring at least one thread remains active to prevent locks.

See Also:
    - :mod:`concurrent.futures` for the original synchronous executor implementations.
"""

import asyncio
import concurrent.futures
import multiprocessing.context
import queue
import threading
import weakref
from concurrent.futures import _base, thread
from functools import cached_property

from a_sync._typing import *
from a_sync.primitives._debug import _DebugDaemonMixin


TEN_MINUTES = 60 * 10

Initializer = Callable[..., object]


class _AsyncExecutorMixin(concurrent.futures.Executor, _DebugDaemonMixin):
    """
    A mixin for Executors to provide asynchronous run and submit methods.

    This mixin allows executors to operate in both asynchronous (normal) mode and synchronous mode.
    In asynchronous (normal) mode, functions are submitted to the executor and awaited.
    In synchronous mode, functions are executed directly in the current thread.

    Examples:
        >>> async def example():
        >>>     result = await executor.run(some_function, arg1, arg2, kwarg1=value1)
        >>>     print(result)

    See Also:
        - :meth:`submit` for submitting functions to the executor.
    """

    _max_workers: int

    _workers: str
    """The type of workers used."""

    __slots__ = "_max_workers", "_initializer", "_initargs", "_broken", "_shutdown_lock"

    async def run(self, fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs):
        """
        A shorthand way to call `await asyncio.get_event_loop().run_in_executor(this_executor, fn, *args)`.
        Doesn't `await this_executor.run(fn, *args)` look so much better?

        In synchronous mode, the function is executed directly in the current thread.
        In asynchronous mode, the function is submitted to the executor and awaited.

        Args:
            fn: The function to run.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Examples:
            >>> async def example():
            >>>     result = await executor.run(some_function, arg1, arg2, kwarg1=value1)
            >>>     print(result)

        See Also:
            - :meth:`submit` for submitting functions to the executor.
        """
        return fn(*args, **kwargs) if self.sync_mode else await self.submit(fn, *args, **kwargs)

    def submit(self, fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> "asyncio.Future[T]":  # type: ignore [override]
        """
        Submits a job to the executor and returns an :class:`asyncio.Future` that can be awaited for the result without blocking.

        Args:
            fn: The function to submit.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Examples:
            >>> future = executor.submit(some_function, arg1, arg2, kwarg1=value1)
            >>> result = await future
            >>> print(result)

        See Also:
            - :meth:`run` for running functions with the executor.
        """
        if self.sync_mode:
            fut = self._get_loop().create_future()
            try:
                fut.set_result(fn(*args, **kwargs))
            except Exception as e:
                fut.set_exception(e)
        else:
            fut = asyncio.futures.wrap_future(super().submit(fn, *args, **kwargs))  # type: ignore [assignment]
            self._start_debug_daemon(fut, fn, *args, **kwargs)
        return fut

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} object at {hex(id(self))} [{self.worker_count_current}/{self._max_workers} {self._workers}]>"

    def __len__(self) -> int:
        # NOTE: should this be queue length instead? probably
        return self.worker_count_current

    @cached_property
    def sync_mode(self) -> bool:
        """
        Indicates if the executor is in synchronous mode (max_workers == 0).

        Examples:
            >>> if executor.sync_mode:
            >>>     print("Executor is in synchronous mode.")
        """
        return self._max_workers == 0

    @property
    def worker_count_current(self) -> int:
        """
        Returns the current number of workers.

        Examples:
            >>> print(f"Current worker count: {executor.worker_count_current}")
        """
        return len(getattr(self, f"_{self._workers}"))

    async def _debug_daemon(self, fut: asyncio.Future, fn, *args, **kwargs) -> None:
        """
        Runs until manually cancelled by the finished work item.

        Args:
            fut: The future being debugged.
            fn: The function being executed.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        See Also:
            - :meth:`_start_debug_daemon` to start the debug daemon.
        """
        # TODO: make prettier strings for other types
        if type(fn).__name__ == "function":
            fnid = getattr(fn, "__qualname__", fn.__name__)
            if fn.__module__:
                fnid = f"{fn.__module__}.{fnid}"
        else:
            fnid = fn

        msg = f"%s processing %s{args}"
        if kwargs:
            msg = f"{msg[:-1]} {', '.join(f'{k}={v}' for k, v in kwargs.items())})"
        else:
            msg = f"{msg[:-2]})"

        while not fut.done():
            await asyncio.sleep(15)
            if not fut.done():
                self.logger.debug(msg, self, fnid)


# Process


class AsyncProcessPoolExecutor(_AsyncExecutorMixin, concurrent.futures.ProcessPoolExecutor):
    """
    A :class:`concurrent.futures.ProcessPoolExecutor' subclass providing asynchronous
    run and submit methods that support kwargs, with support for synchronous mode

    Examples:
        >>> executor = AsyncProcessPoolExecutor(max_workers=4)
        >>> future = executor.submit(some_function, arg1, arg2, kwarg1='kwarg1')
        >>> result = await future
    """

    _workers = "processes"
    """The type of workers used, set to "processes"."""

    __slots__ = (
        "_mp_context",
        "_processes",
        "_pending_work_items",
        "_call_queue",
        "_result_queue",
        "_queue_management_thread",
        "_queue_count",
        "_shutdown_thread",
        "_work_ids",
        "_queue_management_thread_wakeup",
    )

    def __init__(
        self,
        max_workers: Optional[int] = None,
        mp_context: Optional[multiprocessing.context.BaseContext] = None,
        initializer: Optional[Initializer] = None,
        initargs: Tuple[Any, ...] = (),
    ) -> None:
        """
        Initializes the AsyncProcessPoolExecutor.

        Args:
            max_workers: The maximum number of workers. Defaults to None.
            mp_context: The multiprocessing context. Defaults to None.
            initializer: An initializer callable. Defaults to None.
            initargs: Arguments for the initializer. Defaults to ().

        Examples:
            >>> executor = AsyncProcessPoolExecutor(max_workers=4)
            >>> future = executor.submit(some_function, arg1, arg2)
            >>> result = await future
        """
        if max_workers == 0:
            super().__init__(1, mp_context, initializer, initargs)
            self._max_workers = 0
        else:
            super().__init__(max_workers, mp_context, initializer, initargs)


# Thread


class AsyncThreadPoolExecutor(_AsyncExecutorMixin, concurrent.futures.ThreadPoolExecutor):
    """
    A :class:`concurrent.futures.ThreadPoolExecutor' subclass providing asynchronous
    run and submit methods that support kwargs, with support for synchronous mode

    Examples:
        >>> executor = AsyncThreadPoolExecutor(max_workers=10, thread_name_prefix="MyThread")
        >>> future = executor.submit(some_function, arg1, arg2, kwarg1='kwarg1')
        >>> result = await future
    """

    _workers = "threads"
    """The type of workers used, set to "threads"."""

    __slots__ = (
        "_work_queue",
        "_idle_semaphore",
        "_threads",
        "_shutdown",
        "_thread_name_prefix",
    )

    def __init__(
        self,
        max_workers: Optional[int] = None,
        thread_name_prefix: str = "",
        initializer: Optional[Initializer] = None,
        initargs: Tuple[Any, ...] = (),
    ) -> None:
        """
        Initializes the AsyncThreadPoolExecutor.

        Args:
            max_workers: The maximum number of workers. Defaults to None.
            thread_name_prefix: Prefix for thread names. Defaults to ''.
            initializer: An initializer callable. Defaults to None.
            initargs: Arguments for the initializer. Defaults to ().

        Examples:
            >>> executor = AsyncThreadPoolExecutor(max_workers=10, thread_name_prefix="MyThread")
            >>> future = executor.submit(some_function, arg1, arg2)
            >>> result = await future
        """
        if max_workers == 0:
            super().__init__(1, thread_name_prefix, initializer, initargs)
            self._max_workers = 0
        else:
            super().__init__(max_workers, thread_name_prefix, initializer, initargs)


# For backward-compatibility
ProcessPoolExecutor = AsyncProcessPoolExecutor
ThreadPoolExecutor = AsyncThreadPoolExecutor

# Pruning thread pool


def _worker(executor_reference, work_queue, initializer, initargs, timeout):  # NOTE: NEW 'timeout'
    """
    Worker function for the PruningThreadPoolExecutor.

    Args:
        executor_reference: A weak reference to the executor.
        work_queue: The work queue.
        initializer: The initializer function.
        initargs: Arguments for the initializer.
        timeout: Timeout duration for pruning inactive threads.

    See Also:
        - :class:`PruningThreadPoolExecutor` for more details on thread pruning.
    """
    if initializer is not None:
        try:
            initializer(*initargs)
        except BaseException:
            _base.LOGGER.critical("Exception in initializer:", exc_info=True)
            executor = executor_reference()
            if executor is not None:
                executor._initializer_failed()
            return

    try:
        while True:
            try:  # NOTE: NEW
                work_item = work_queue.get(block=True, timeout=timeout)  # NOTE: NEW
            except queue.Empty:  # NOTE: NEW
                # Its been 'timeout' seconds and there are no new work items.  # NOTE: NEW
                # Let's suicide the thread.  # NOTE: NEW
                executor = executor_reference()  # NOTE: NEW

                with executor._adjusting_lock:  # NOTE: NEW
                    # NOTE: We keep a minimum of one thread active to prevent locks
                    if len(executor) > 1 or executor._idle_semaphore.acquire(
                        blocking=False
                    ):  # NOTE: NEW
                        t = threading.current_thread()  # NOTE: NEW
                        executor._threads.remove(t)  # NOTE: NEW
                        thread._threads_queues.pop(t)  # NOTE: NEW
                        # Let the executor know we have one less idle thread available
                        executor._idle_semaphore.acquire(blocking=False)  # NOTE: NEW
                        return  # NOTE: NEW
                continue

            if work_item is not None:
                work_item.run()
                # Delete references to object. See issue16284
                del work_item

                # attempt to increment idle count
                executor = executor_reference()
                if executor is not None:
                    executor._idle_semaphore.release()
                del executor
                continue

            executor = executor_reference()
            # Exit if:
            #   - The interpreter is shutting down OR
            #   - The executor that owns the worker has been collected OR
            #   - The executor that owns the worker has been shutdown OR
            if thread._shutdown or executor is None or executor._shutdown:
                # Flag the executor as shutting down as early as possible if it is not gc-ed yet.
                if executor is not None:
                    executor._shutdown = True
                # Notice other workers
                work_queue.put(None)
                return
            del executor
    except BaseException:
        _base.LOGGER.critical("Exception in worker", exc_info=True)


class PruningThreadPoolExecutor(AsyncThreadPoolExecutor):
    """
    This :class:`~AsyncThreadPoolExecutor` implementation prunes inactive threads after 'timeout' seconds without a work item.
    Pruned threads will be automatically recreated as needed for future workloads. Up to 'max_threads' can be active at any one time.
    The executor ensures that at least one active thread remains to prevent locks.

    Note:
        The `_worker` function includes a check (`len(executor) > 1`) to ensure that at least one thread remains active.
        This prevents the executor from having zero active threads, which could lead to deadlocks.

    Examples:
        >>> executor = PruningThreadPoolExecutor(max_workers=5, timeout=300)
        >>> future = executor.submit(some_function, arg1, arg2, kwarg1='kwarg1')
        >>> result = await future
    """

    __slots__ = "_timeout", "_adjusting_lock"

    def __init__(
        self,
        max_workers=None,
        thread_name_prefix="",
        initializer=None,
        initargs=(),
        timeout=TEN_MINUTES,
    ):
        """
        Initializes the PruningThreadPoolExecutor.

        Args:
            max_workers: The maximum number of workers. Defaults to None.
            thread_name_prefix: Prefix for thread names. Defaults to ''.
            initializer: An initializer callable. Defaults to None.
            initargs: Arguments for the initializer. Defaults to ().
            timeout: Timeout duration for pruning inactive threads. Defaults to TEN_MINUTES.

        Examples:
            >>> executor = PruningThreadPoolExecutor(max_workers=5, timeout=300)
            >>> future = executor.submit(some_function, arg1, arg2)
            >>> result = await future
        """

        self._timeout = timeout
        """Timeout duration for pruning inactive threads."""

        self._adjusting_lock = threading.Lock()
        """Lock used to adjust the number of threads."""

        super().__init__(max_workers, thread_name_prefix, initializer, initargs)

    def __len__(self) -> int:
        return len(self._threads)

    def _adjust_thread_count(self):
        """
        Adjusts the number of threads based on workload and idle threads.

        See Also:
            - :func:`_worker` for the worker function that handles thread pruning.
        """
        with self._adjusting_lock:
            # if idle threads are available, don't spin new threads
            if self._idle_semaphore.acquire(timeout=0):
                return

            # When the executor gets lost, the weakref callback will wake up
            # the worker threads.
            def weakref_cb(_, q=self._work_queue):
                q.put(None)

            num_threads = len(self._threads)
            if num_threads < self._max_workers:
                thread_name = "%s_%d" % (self._thread_name_prefix or self, num_threads)
                t = threading.Thread(
                    name=thread_name,
                    target=_worker,
                    args=(
                        weakref.ref(self, weakref_cb),
                        self._work_queue,
                        self._initializer,
                        self._initargs,
                        self._timeout,
                    ),
                )
                t.daemon = True
                t.start()
                self._threads.add(t)
                thread._threads_queues[t] = self._work_queue


executor = PruningThreadPoolExecutor(128)

__all__ = [
    "AsyncThreadPoolExecutor",
    "AsyncProcessPoolExecutor",
    "PruningThreadPoolExecutor",
]
