from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])

try:
    from langchain_core.tools import tool
except ImportError:
    def tool(*args: Any, **kwargs: Any) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            return func

        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return cast(Callable[[F], F], decorator)(cast(F, args[0]))

        return decorator
