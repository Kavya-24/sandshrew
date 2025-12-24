"""Tests for sandshrew base_tool decorator and BaseTool class."""

import pytest

from sandshrew import (
    BaseTool,
    Executor,
    Provider,
    prepare_tools,
    sand_tool,
)


@sand_tool(tags=["test"])
def simple_add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@sand_tool(retry_count=1, tags=["test"])
def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@sand_tool(inject_state=True, tags=["test"])
def get_user_email(_injected_state: dict) -> str:
    """Get user email from state."""
    return _injected_state.get("user_email", "unknown")


# Async tool definitions for testing
@sand_tool(tags=["async", "test"])
async def async_add(a: int, b: int) -> int:
    """Add two numbers asynchronously."""
    return a + b


@sand_tool(retry_count=2, tags=["async", "test"])
async def async_divide(a: float, b: float) -> float:
    """Divide two numbers asynchronously."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@sand_tool(inject_state=True, tags=["async", "test"])
async def async_get_user_email(_injected_state: dict) -> str:
    """Get user email from state asynchronously."""
    return _injected_state.get("user_email", "unknown")


class TestSandToolDecorator:
    """Test the sand_tool decorator."""

    def test_basic_tool_creation(self):
        """Test that decorator creates a BaseTool instance."""
        assert isinstance(simple_add, BaseTool)
        assert simple_add.name == "simple_add"

    def test_tool_execution(self):
        """Test basic tool execution."""
        result = simple_add(2, 3)
        assert result == 5

    def test_tool_metadata(self):
        """Test tool metadata extraction."""
        metadata = simple_add.get_metadata()
        assert metadata["name"] == "simple_add"
        assert metadata["tags"] == ["test"]
        assert "parameters" in metadata

    def test_tool_description_openai(self):
        """Test OpenAI tool description generation."""
        description = simple_add.get_tool_description(Provider.OPENAI)
        assert description["type"] == "function"
        assert description["function"]["name"] == "simple_add"


class TestToolExecution:
    """Test tool execution with error handling."""

    def test_successful_execution(self):
        """Test successful tool execution."""
        result = simple_add(5, 10)
        assert result == 15

    def test_error_handling(self):
        """Test error handling in tool execution."""
        with pytest.raises(Exception):
            divide(10, 0)

    def test_retry_logic(self):
        """Test retry configuration."""
        assert divide.config.retry_count == 1


class TestStateInjection:
    """Test state injection in tools."""

    def test_state_injection(self):
        """Test injecting state into tool."""
        state = {"user_email": "test@example.com"}
        result = get_user_email(_injected_state=state)
        assert result == "test@example.com"

    def test_missing_state_key(self):
        """Test handling missing state keys."""
        state = {}
        result = get_user_email(_injected_state=state)
        assert result == "unknown"


class TestExecutor:
    """Test the Executor class."""

    def test_executor_initialization(self):
        """Test executor creation."""
        executor = Executor(tool_list=[simple_add], provider=Provider.OPENAI)
        assert "simple_add" in executor.tools

    def test_executor_with_state(self):
        """Test executor with injected state."""
        state = {"user_email": "test@example.com"}
        executor = Executor(
            tool_list=[get_user_email], provider=Provider.OPENAI, _injected_state=state
        )
        assert executor._injected_state == state


class TestToolPreparation:
    """Test tool preparation for LLM."""

    def test_prepare_tools(self):
        """Test preparing tools for LLM consumption."""
        tools = prepare_tools(Provider.OPENAI, [simple_add, divide])
        assert len(tools) == 2
        assert all(tool["type"] == "function" for tool in tools)

    def test_prepare_empty_tools(self):
        """Test preparing empty tool list."""
        tools = prepare_tools(Provider.OPENAI, [])
        assert tools == []


# ============================================================================
# Async Tests
# ============================================================================


class TestAsyncToolDetection:
    """Test async tool detection."""

    def test_sync_tool_is_not_async(self):
        """Test that sync tools are detected correctly."""
        assert simple_add.is_async is False
        assert divide.is_async is False

    def test_async_tool_is_async(self):
        """Test that async tools are detected correctly."""
        assert async_add.is_async is True
        assert async_divide.is_async is True


class TestAsyncToolExecution:
    """Test async tool execution."""

    @pytest.mark.asyncio
    async def test_async_tool_execution(self):
        """Test async tool can be executed with __acall__."""
        result = await async_add.__acall__(2, 3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_async_tool_with_state(self):
        """Test async tool with injected state."""
        state = {"user_email": "async@example.com"}
        result = await async_get_user_email.__acall__(_injected_state=state)
        assert result == "async@example.com"

    @pytest.mark.asyncio
    async def test_sync_tool_via_acall(self):
        """Test that sync tools can be called via __acall__."""
        result = await simple_add.__acall__(5, 10)
        assert result == 15

    @pytest.mark.asyncio
    async def test_async_tool_error_handling(self):
        """Test async error handling."""
        with pytest.raises(Exception):
            await async_divide.__acall__(10, 0)


class TestAsyncExecutor:
    """Test the Executor class with is_async=True."""

    def test_async_executor_initialization(self):
        """Test Executor creation with is_async=True."""
        executor = Executor(tool_list=[async_add], provider=Provider.OPENAI, is_async=True)
        assert "async_add" in executor.tools
        assert executor.is_async is True

    def test_async_executor_with_state(self):
        """Test Executor with injected state in async mode."""
        state = {"user_email": "test@example.com"}
        executor = Executor(
            tool_list=[async_get_user_email],
            provider=Provider.OPENAI,
            _injected_state=state,
            is_async=True,
        )
        assert executor._injected_state == state

    def test_async_executor_parallel_config(self):
        """Test Executor parallel configuration in async mode."""
        executor = Executor(
            tool_list=[async_add],
            provider=Provider.OPENAI,
            use_parallel=True,
            max_concurrency=10,
            is_async=True,
        )
        assert executor.use_parallel is True
        assert executor.max_concurrency == 10
        assert executor.is_async is True

    def test_sync_execute_error_async_mode(self):
        """Test that .execute() raises error when is_async=True."""
        executor = Executor(is_async=True)
        with pytest.raises(RuntimeError):
            executor.execute("some_response")

    @pytest.mark.asyncio
    async def test_async_execute_error_sync_mode(self):
        """Test that .aexecute() raises error when is_async=False."""
        executor = Executor(is_async=False)
        with pytest.raises(RuntimeError):
            await executor.aexecute("some_response")
