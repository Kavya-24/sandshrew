from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from .base_tool import BaseTool
from .data_types import ExecutionError, ExecutionResult, Provider, ToolCall
from .llm_utils.openai import OpenAIUtils


def prepare_tools(provider: Provider, tools: List[BaseTool]) -> List[Dict[str, Any]]:
    if not tools or len(tools) == 0:
        return []

    tool_descriptions = []
    for tool in tools:
        tool_descriptions.append(tool.get_tool_description(provider))

    return tool_descriptions


def extract_assistant_message(
    provider: Provider, provider_completion_response: Any
) -> List[ToolCall]:
    """
    Based on the provider, return the value of tool-calls parsed from the response
    """
    match provider:
        case Provider.OPENAI:
            return OpenAIUtils.extract_assistant_message(provider_completion_response)
        case _:
            raise ValueError(f"Unsupported provider: {provider}")


def extract_tool_calls(provider: Provider, provider_completion_response: Any) -> List[ToolCall]:
    """
    Based on the provider, return the value of tool-calls parsed from the response
    """
    match provider:
        case Provider.OPENAI:
            return OpenAIUtils.parse_tool_calls(provider_completion_response)
        case _:
            raise ValueError(f"Unsupported provider: {provider}")


def check_turn_completion(provider: Provider, provider_completion_response: Any) -> bool:
    """
    Checks whether the turn has completed or not
    """
    match provider:
        case Provider.OPENAI:
            return OpenAIUtils.check_turn_completion(provider_completion_response)
        case _:
            raise ValueError(f"Unsupported provider: {provider}")


class Executor:
    def __init__(
        self,
        *,
        tool_list: List[BaseTool] = None,
        use_parallel: bool = False,
        max_concurrency: int = 5,
        provider: Provider = Provider.OPENAI,
        _injected_state: Optional[Any] = {},
        is_async: bool = False,
    ):
        self.tools = {tool.name: tool for tool in (tool_list or [])}
        self.use_parallel = use_parallel
        self.max_concurrency = max_concurrency
        self.provider = provider
        self._injected_state = _injected_state
        self.is_async = is_async

    def execute(self, provider_completion_response: Any) -> List[ExecutionResult]:
        """
        Execute tool calls synchronous calling .execute_single_tool() which calls tool.__call__
        """
        if self.is_async:
            raise RuntimeError("Cannot use .execute() when is_async=True. Use .aexecute() instead.")

        tool_calls: List[ToolCall] = extract_tool_calls(self.provider, provider_completion_response)

        if self.use_parallel:
            return self._execute_parallel(tool_calls)
        else:
            return self._execute_sequential(tool_calls)

    async def aexecute(self, provider_completion_response: Any) -> List[ExecutionResult]:
        """
        Desc:
            Execute tool calls asynchronously calling .execute_single_tool() which calls tool.__acall__
        """
        if not self.is_async:
            raise RuntimeError(
                "Cannot use .aexecute() when is_async=False. Use .execute() instead."
            )

        tool_calls: List[ToolCall] = extract_tool_calls(self.provider, provider_completion_response)

        if self.use_parallel:
            return await self._execute_parallel_async(tool_calls)
        else:
            return await self._execute_sequential_async(tool_calls)

    def _execute_sequential(self, tool_calls: List[ToolCall]) -> List[ExecutionResult]:
        """Execute tool calls sequentially."""
        results: List[ExecutionResult] = []

        for tool_call in tool_calls:
            result = self._execute_single_tool(tool_call)
            results.append(result)

        return results

    def _execute_parallel(self, tool_calls: List[ToolCall]) -> List[ExecutionResult]:
        """Execute tool calls in parallel with max_concurrency limit."""
        results: Dict[str, ExecutionResult] = {}

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            future_to_tool_call = {
                executor.submit(self._execute_single_tool, tool_call): tool_call
                for tool_call in tool_calls
            }

            for future in as_completed(future_to_tool_call):
                result = future.result()
                results[result.tool_call.id] = result

        return [results[tool_call.id] for tool_call in tool_calls]

    async def _execute_sequential_async(self, tool_calls: List[ToolCall]) -> List[ExecutionResult]:
        """Execute tool calls sequentially and asynchronously."""
        results: List[ExecutionResult] = []

        for tool_call in tool_calls:
            result = await self._execute_single_tool(tool_call)
            results.append(result)

        return results

    async def _execute_parallel_async(self, tool_calls: List[ToolCall]) -> List[ExecutionResult]:
        """Execute tool calls in parallel using asyncio.gather with concurrency limit."""
        import asyncio

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def execute_with_semaphore(tool_call: ToolCall) -> ExecutionResult:
            async with semaphore:
                return await self._execute_single_tool(tool_call)

        results = await asyncio.gather(*[execute_with_semaphore(tc) for tc in tool_calls])
        return list(results)

    def _execute_single_tool(self, tool_call: ToolCall) -> ExecutionResult | Any:
        """Execute a single tool call and return the result."""
        try:
            if tool_call.name not in self.tools:
                error = ExecutionError(
                    message=f"Tool '{tool_call.name}' not found", retryable=False
                )
                return ExecutionResult(tool_call=tool_call, error=error)

            tool = self.tools[tool_call.name]

            # Tool call arguments must be a mapping
            tool_call_args = tool_call.arguments or {}

            if not isinstance(tool_call_args, dict):
                raise TypeError(f"tool_call.arguments must be a dict, got {type(tool_call_args)}")

            if self.is_async:
                # Async execution
                if tool.config.inject_state:
                    return tool.__acall__(self._injected_state, **tool_call_args)
                else:
                    return tool.__acall__(**tool_call_args)
            else:
                # Sync execution
                if tool.config.inject_state:
                    content = tool(self._injected_state, **tool_call_args)
                else:
                    content = tool(**tool_call_args)
                return ExecutionResult(tool_call=tool_call, content=content)

        except Exception as e:
            error = ExecutionError(
                message=str(e),
                raw_error=e,
                retryable=False,
                backtrace=traceback.format_exc(),
            )
            return ExecutionResult(tool_call=tool_call, error=error)
