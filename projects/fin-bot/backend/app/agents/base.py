from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

from pydantic import BaseModel

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class BaseAgent(ABC, Generic[InputT, OutputT]):
    name: str
    system_prompt: str
    allowed_tools: Sequence[str]

    def __init__(self) -> None:
        self.runnable = self._build_runnable()

    def _build_runnable(self) -> object | None:
        try:
            from langchain_core.prompts import ChatPromptTemplate  # type: ignore
        except ImportError:
            return None

        return ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{input}"),
            ]
        )

    @abstractmethod
    def run(self, payload: InputT) -> OutputT:
        raise NotImplementedError
