from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ExecutionError:
    message: str
    raw_error: Optional[Exception] = None
    retryable: bool = False
    backtrace: Optional[Any] = None


@dataclass
class ExecutionResult:
    tool_call: ToolCall
    content: Optional[Any] = None
    error: Optional[ExecutionError] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def failed(self) -> bool:
        return self.error is not None


class Provider(str, Enum):
    OPENAI = "openai"
