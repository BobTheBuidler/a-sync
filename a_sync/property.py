
import async_property as ap  # type: ignore [import]

from a_sync import _helpers, config
from a_sync._bound import ASyncMethodDescriptorAsyncDefault
from a_sync._descriptor import ASyncDescriptor, clean_default_from_modifiers
from a_sync._typing import *


class _ASyncPropertyDescriptorBase(ASyncDescriptor[T]):
    _fget: HiddenMethod[ASyncInstance, T]
    def __init__(self, _fget: Callable[Concatenate[ASyncInstance, P], Awaitable[T]], field_name=None, **modifiers: config.ModifierKwargs):
        super().__init__(_fget, field_name, **modifiers)
        self.hidden_method_name = f"__{self.field_name}__"
        hidden_modifiers = clean_default_from_modifiers(self, self.modifiers)
        self.hidden_method_descriptor =  ASyncMethodDescriptorAsyncDefault(self.get, self.hidden_method_name, **hidden_modifiers)
    async def get(self, instance: ASyncInstance) -> T:
        return await super().__get__(instance, self)
    def __get__(self, instance: ASyncInstance, owner) -> T:
        awaitable = super().__get__(instance, owner)
        return _helpers._await(awaitable) if instance.__a_sync_instance_should_await__ else awaitable

class ASyncPropertyDescriptor(_ASyncPropertyDescriptorBase[T], ap.base.AsyncPropertyDescriptor):
    pass
        
class ASyncCachedPropertyDescriptor(_ASyncPropertyDescriptorBase[T], ap.cached.AsyncCachedPropertyDescriptor):
    def __init__(self, _fget, _fset=None, _fdel=None, field_name=None, **modifiers: Unpack[ModifierKwargs]):
        super().__init__(_fget, field_name, **modifiers)
        self._fset = _fset
        self._fdel = _fdel
        self._check_method_sync(_fset, 'setter')
        self._check_method_sync(_fdel, 'deleter')

class ASyncPropertyDescriptorSyncDefault(ASyncPropertyDescriptor[T]):
    """This is a helper class used for type checking. You will not run into any instance of this in prod."""

class ASyncPropertyDescriptorAsyncDefault(ASyncPropertyDescriptor[T]):
    """This is a helper class used for type checking. You will not run into any instance of this in prod."""
    def __get__(self, instance, owner) -> Awaitable[T]:
        return super().__get__(instance, owner)


ASyncPropertyDecorator = Callable[[Property[T]], ASyncPropertyDescriptor[T]]
ASyncPropertyDecoratorSyncDefault = Callable[[Property[T]], ASyncPropertyDescriptorSyncDefault[T]]
ASyncPropertyDecoratorAsyncDefault = Callable[[Property[T]], ASyncPropertyDescriptorAsyncDefault[T]]

@overload
def a_sync_property(  # type: ignore [misc]
    func: Literal[None],
    default: Literal["sync"],
    **modifiers: Unpack[ModifierKwargs],
) -> ASyncPropertyDecoratorSyncDefault[T]:...

@overload
def a_sync_property(  # type: ignore [misc]
    func: Literal[None],
    default: Literal["async"],
    **modifiers: Unpack[ModifierKwargs],
) -> ASyncPropertyDecoratorAsyncDefault[T]:...

@overload
def a_sync_property(  # type: ignore [misc]
    func: Literal[None],
    default: DefaultMode = config.DEFAULT_MODE,
    **modifiers: Unpack[ModifierKwargs],
) -> ASyncPropertyDecorator[T]:...
    
@overload
def a_sync_property(  # type: ignore [misc]
    default: Literal["sync"],
    **modifiers: Unpack[ModifierKwargs],
) -> ASyncPropertyDecoratorSyncDefault[T]:...
    
@overload
def a_sync_property(  # type: ignore [misc]
    default: Literal["async"],
    **modifiers: Unpack[ModifierKwargs],
) -> ASyncPropertyDecoratorAsyncDefault[T]:...
    
@overload
def a_sync_property(  # type: ignore [misc]
    func: Property[T],
    default: Literal["sync"],
    **modifiers: Unpack[ModifierKwargs],
) -> ASyncPropertyDescriptorSyncDefault[T]:...
    
@overload
def a_sync_property(  # type: ignore [misc]
    func: Property[T],
    default: Literal["async"],
    **modifiers: Unpack[ModifierKwargs],
) -> ASyncPropertyDescriptorAsyncDefault[T]:...
    
@overload
def a_sync_property(  # type: ignore [misc]
    func: Property[T],
    default: DefaultMode = config.DEFAULT_MODE,
    **modifiers: Unpack[ModifierKwargs],
) -> ASyncPropertyDescriptor[T]:...
    
def a_sync_property(  # type: ignore [misc]
    func: Union[Property[T], DefaultMode] = None,
    default: DefaultMode = config.DEFAULT_MODE,
    **modifiers: Unpack[ModifierKwargs],
) -> Union[
    ASyncPropertyDescriptor[T],
    ASyncPropertyDescriptorSyncDefault[T], 
    ASyncPropertyDescriptorAsyncDefault[T], 
    ASyncPropertyDecorator[T],
    ASyncPropertyDecoratorSyncDefault[T],
    ASyncPropertyDecoratorAsyncDefault[T],
]:
    if func in ['sync', 'async']:
        modifiers['default'] = func
        func = None
    def decorator(func: Property[T]) -> ASyncPropertyDescriptor[T]:
        return ASyncPropertyDescriptor(func, **modifiers)
    return decorator if func is None else decorator(func)  # type: ignore [arg-type]


class ASyncCachedPropertyDescriptorSyncDefault(ASyncCachedPropertyDescriptor[T]):
    """This is a helper class used for type checking. You will not run into any instance of this in prod."""

class ASyncCachedPropertyDescriptorAsyncDefault(ASyncCachedPropertyDescriptor[T]):
    """This is a helper class used for type checking. You will not run into any instance of this in prod."""
    def __get__(self, instance, owner) -> Awaitable[T]:
        return super().__get__(instance, owner)

ASyncCachedPropertyDecorator = Callable[[Property[T]], ASyncCachedPropertyDescriptor[T]]
ASyncCachedPropertyDecoratorSyncDefault = Callable[[Property[T]], ASyncCachedPropertyDescriptorSyncDefault[T]]
ASyncCachedPropertyDecoratorAsyncDefault = Callable[[Property[T]], ASyncCachedPropertyDescriptorAsyncDefault[T]]

@overload
def a_sync_cached_property(  # type: ignore [misc]
    func: Literal[None],
    default: Literal["sync"],
    **modifiers: Unpack[ModifierKwargs],
) -> ASyncCachedPropertyDecoratorSyncDefault[T]:...

@overload
def a_sync_cached_property(  # type: ignore [misc]
    func: Literal[None],
    default: Literal["async"],
    **modifiers: Unpack[ModifierKwargs],
) -> ASyncCachedPropertyDecoratorAsyncDefault[T]:...

@overload
def a_sync_cached_property(  # type: ignore [misc]
    func: Literal[None],
    default: DefaultMode,
    **modifiers: Unpack[ModifierKwargs],
) -> ASyncCachedPropertyDecorator[T]:...

@overload
def a_sync_cached_property(  # type: ignore [misc]
    default: Literal["sync"],
    **modifiers: Unpack[ModifierKwargs],
) -> ASyncCachedPropertyDecoratorSyncDefault[T]:...

@overload
def a_sync_cached_property(  # type: ignore [misc]
    default: Literal["async"],
    **modifiers: Unpack[ModifierKwargs],
) -> ASyncCachedPropertyDecoratorAsyncDefault[T]:...
    
@overload
def a_sync_cached_property(  # type: ignore [misc]
    func: Property[T],
    default: Literal["sync"],
    **modifiers: Unpack[ModifierKwargs],
) -> ASyncCachedPropertyDescriptorSyncDefault[T]:... 

@overload
def a_sync_cached_property(  # type: ignore [misc]
    func: Property[T],
    default: Literal["async"],
    **modifiers: Unpack[ModifierKwargs],
) -> ASyncCachedPropertyDescriptorAsyncDefault[T]:... 

@overload
def a_sync_cached_property(  # type: ignore [misc]
    func: Property[T],
    default: DefaultMode = config.DEFAULT_MODE,
    **modifiers: Unpack[ModifierKwargs],
) -> ASyncCachedPropertyDescriptor[T]:...
    
def a_sync_cached_property(  # type: ignore [misc]
    func: Optional[Property[T]] = None,
    default: DefaultMode = config.DEFAULT_MODE,
    **modifiers: Unpack[ModifierKwargs],
) -> Union[
    ASyncCachedPropertyDescriptor[T],
    ASyncCachedPropertyDescriptorSyncDefault[T], 
    ASyncCachedPropertyDescriptorAsyncDefault[T], 
    ASyncCachedPropertyDecorator[T],
    ASyncCachedPropertyDecoratorSyncDefault[T],
    ASyncCachedPropertyDecoratorAsyncDefault[T],
]:
    def decorator(func: Property[T]) -> ASyncCachedPropertyDescriptor[T]:
        return ASyncCachedPropertyDescriptor(func, **modifiers)
    return decorator if func is None else decorator(func)
