"""
Sandshrew: A tool framework for building and executing structured tool calls.

This module provides utilities for defining tools, managing their execution,
and handling tool calls in a structured manner.
"""

from .base_tool import (
    BaseTool,
    ToolConfig,
    ToolError,
    ToolExecutionError,
    ToolValidationError,
    sand_tool,
)
from .data_types import ExecutionError, ExecutionResult, Provider, ToolCall
from .helpers import (
    AsyncExecutor,
    Executor,
    check_turn_completion,
    extract_assistant_message,
    extract_tool_calls,
    prepare_tools,
)

__version__ = "0.1.0"
__author__ = "Kavya-24"

__all__ = [
    "AsyncExecutor",
    "BaseTool",
    "Executor",
    "ExecutionError",
    "ExecutionResult",
    "Provider",
    "ToolCall",
    "ToolConfig",
    "ToolError",
    "ToolExecutionError",
    "ToolValidationError",
    "check_turn_completion",
    "extract_assistant_message",
    "extract_tool_calls",
    "prepare_tools",
    "sand_tool",
]
