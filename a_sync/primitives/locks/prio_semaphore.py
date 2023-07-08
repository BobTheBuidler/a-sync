
import asyncio
import heapq
from typing import (Dict, Generic, List, Literal, Optional, Protocol, Type,
                    TypeVar)

from a_sync import Semaphore


T = TypeVar('T', covariant=True)

class Priority(Protocol):
    def __lt__(self, other) -> bool:
        ...

PT = TypeVar('PT', bound=Priority)
    
CM = TypeVar('CM', bound="_AbstractPrioritySemaphoreContextManager[Priority]")

class _AbstractPrioritySemaphore(Semaphore, Generic[PT, CM]):
    name: Optional[str]
    _value: int
    _waiters: List["_AbstractPrioritySemaphoreContextManager[PT]"]  # type: ignore [assignment]

    @property
    def _context_manager_class(self) -> Type["_AbstractPrioritySemaphoreContextManager[PT]"]:
        raise NotImplementedError
    
    @property
    def _top_priority(self) -> PT:
        # You can use this so you can set priorities with non numeric comparable values
        raise NotImplementedError

    def __init__(self, value: int = 1, *, name: Optional[str] = None) -> None:
        self._context_managers: Dict[PT, _AbstractPrioritySemaphoreContextManager[PT]] = {}
        self._capacity = value
        super().__init__(value, name=name)
        self._waiters = []

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} capacity={self._capacity} value={self._value} waiters={self._waiters}>"

    async def __aenter__(self) -> None:
        await self[self._top_priority].acquire()

    async def __aexit__(self, *_) -> None:
        self[self._top_priority].release()
    
    def __getitem__(self, priority: Optional[PT]) -> "_AbstractPrioritySemaphoreContextManager[PT]":
        priority = self._top_priority if priority is None else priority
        if priority not in self._context_managers:
            context_manager = self._context_manager_class(self, priority, name=self.name)
            heapq.heappush(self._waiters, context_manager)  # type: ignore [misc]
            self._context_managers[priority] = context_manager
        return self._context_managers[priority]
    
    def _wake_up_next(self) -> None:
        if self._waiters:
            manager = heapq.heappop(self._waiters)
            manager._wake_up_next()
            self._heap_push(manager)
    
    def _heap_push(self, manager: "_AbstractPrioritySemaphoreContextManager[PT]") -> None:
        if len(manager):
            # There are still waiters, put the manager back
            heapq.heappush(self._waiters, manager)  # type: ignore [misc]
        else:
            # There are no more waiters, get rid of the empty manager
            self._context_managers.pop(manager._priority)

class _AbstractPrioritySemaphoreContextManager(Semaphore, Generic[PT]):
    _loop: asyncio.AbstractEventLoop
    _waiters: List[asyncio.Future]  # type: ignore [assignment]
    
    @property
    def _priority_name(self) -> str:
        raise NotImplementedError
    
    def __init__(self, parent: _AbstractPrioritySemaphore, priority: PT, name: Optional[str] = None) -> None:
        self._parent = parent
        self._priority = priority
        super().__init__(0, name=name)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} parent={self._parent} {self._priority_name}={self._priority} waiters={len(self)}>"
    
    def __lt__(self, other) -> bool:
        if type(other) is not type(self):
            raise TypeError(f"{other} is not type {self.__class__.__name__}")
        return self._priority < other._priority
    
    async def acquire(self) -> Literal[True]:
        """Acquire a semaphore.

        If the internal counter is larger than zero on entry,
        decrement it by one and return True immediately.  If it is
        zero on entry, block, waiting until some other coroutine has
        called release() to make it larger than 0, and then return
        True.
        """
        while self._parent._value <= 0:
            fut = self._loop.create_future()
            self._waiters.append(fut)
            try:
                await fut
            except:
                # See the similar code in Queue.get.
                fut.cancel()
                if self._parent._value > 0 and not fut.cancelled():
                    self._parent._wake_up_next()
                raise
        self._parent._value -= 1
        return True
    def release(self) -> None:
        self._parent.release()
    
class _PrioritySemaphoreContextManager(_AbstractPrioritySemaphoreContextManager[int]):
    _priority_name = "priority"

class PrioritySemaphore(_AbstractPrioritySemaphore[int, _PrioritySemaphoreContextManager]):  # type: ignore [type-var]
    _context_manager_class = _PrioritySemaphoreContextManager
    _top_priority = -1
    """
    It's kinda like a regular Semaphore but you must give each waiter a priority:

    ```
    priority_semaphore = PrioritySemaphore(10)

    async with priority_semaphore[priority]:
        await do_stuff()
    ```
    
    You can aenter and aexit this semaphore without a priority and it will process those first. Like so:
    
    ```
    priority_semaphore = PrioritySemaphore(10)
    
    async with priority_semaphore:
        await do_stuff()
    ```
    """
