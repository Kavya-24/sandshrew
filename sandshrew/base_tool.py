import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, get_type_hints

from pydantic import Field

from .data_types import Provider
from .llm_utils.openai import OpenAIUtils


@dataclass
class ToolConfig:
    """Configuration for a sand_tool decorator."""

    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    retry_count: int = 0
    timeout: Optional[float] = None
    inject_state: bool = False

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ToolError(Exception):
    """Base exception for tool execution errors."""

    pass


class ToolExecutionError(ToolError):
    """Raised when tool execution fails."""

    pass


class ToolValidationError(ToolError):
    """Raised when tool validation fails."""

    pass


class BaseTool:
    """Base class for wrapped tool functions with production reliability features."""

    def __init__(self, func: Callable, config: ToolConfig):
        self.func = func
        self.config = config or ToolConfig()
        self.name = config.name or func.__name__
        self.description = config.description or func.__doc__ or ""
        self._signature = inspect.signature(func)
        self._type_hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}

    def __call__(self, *args, **kwargs) -> Any:
        """Execute the tool with error handling and retry logic."""
        attempt = 0
        while attempt <= self.config.retry_count:
            try:
                return self.func(*args, **kwargs)
            except Exception as e:
                attempt += 1
                if attempt > self.config.retry_count:
                    raise ToolExecutionError(
                        f"Tool '{self.name}' failed after {self.config.retry_count + 1} attempts: {str(e)}"
                    ) from e

    @property
    def is_async(self) -> bool:
        """Check if the wrapped function is async."""
        return inspect.iscoroutinefunction(self.func)

    async def __acall__(self, *args, **kwargs) -> Any:
        """Execute the tool asynchronously with error handling and retry logic."""
        if not self.is_async:
            # For sync functions, just call them directly
            return self.__call__(*args, **kwargs)

        attempt = 0
        while attempt <= self.config.retry_count:
            try:
                return await self.func(*args, **kwargs)
            except Exception as e:
                attempt += 1
                if attempt > self.config.retry_count:
                    raise ToolExecutionError(
                        f"Tool '{self.name}' failed after {self.config.retry_count + 1} attempts: {str(e)}"
                    ) from e

    def get_tool_description(self, provider: Provider) -> Dict[str, Any]:
        """Generate OpenAI-compatible tool description from function signature and docstring."""
        params = self._extract_parameters()
        name = self.name
        description = self.description

        match provider:
            case Provider.OPENAI:
                return OpenAIUtils.get_tool_description(name, description, params)
            case _:
                raise ValueError(f"Unsupported provider: {provider}")

    def _extract_parameters(self) -> Dict[str, Any]:
        """Extract parameters from function signature, skipping injected state."""
        properties = {}
        required = []

        for param_name, param in self._signature.parameters.items():
            if param_name in ("self", "cls"):
                continue

            # Skip injected state parameter
            if self.config.inject_state and param_name.startswith("_injected_state"):
                continue

            param_type = self._type_hints.get(param_name, str)
            param_description = self._get_param_description(param_name, param)

            properties[param_name] = {
                "type": self._python_type_to_json_schema(param_type),
                "description": param_description,
            }

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {"properties": properties, "required": required}

    def _get_param_description(self, param_name: str, param: inspect.Parameter) -> str:
        """Extract parameter description from Pydantic Field or default value."""
        # Check if default value is a Pydantic Field with description
        if param.default != inspect.Parameter.empty:
            if isinstance(param.default, type(Field())):
                if hasattr(param.default, "description") and param.default.description:
                    return param.default.description

        return f"Parameter {param_name}"

    @staticmethod
    def _python_type_to_json_schema(python_type: Any) -> str:
        """Convert Python type to JSON schema type."""
        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        if python_type in type_mapping:
            return type_mapping[python_type]

        origin = getattr(python_type, "__origin__", None)
        if origin is list:
            return "array"
        if origin is dict:
            return "object"

        return "string"

    def get_metadata(self) -> Dict[str, Any]:
        """Get tool metadata including config and description."""
        return {
            "name": self.name,
            "description": self.description,
            "tags": self.config.tags,
            "retry_count": self.config.retry_count,
            "timeout": self.config.timeout,
            "parameters": self._extract_parameters(),
        }


def sand_tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    retry_count: int = 0,
    timeout: Optional[float] = None,
    inject_state: bool = False,
) -> Callable:
    """
    Decorator to wrap a function as a sand_tool with production reliability features.

    Args:
        func: The function to wrap (when used without parentheses)
        name: Custom name for the tool (defaults to function name)
        description: Custom description (defaults to function docstring)
        tags: List of tags for categorizing the tool
        retry_count: Number of retries on failure (default: 0)
        timeout: Execution timeout in seconds (default: None)
        inject_state: Whether to inject state as first parameter (default: False)

    Returns:
        Decorated function wrapped in BaseTool
    """

    def decorator(f: Callable) -> BaseTool:
        config = ToolConfig(
            name=name,
            description=description,
            tags=tags or [],
            retry_count=retry_count,
            timeout=timeout,
            inject_state=inject_state,
        )
        return BaseTool(f, config)

    if func is None:
        return decorator
    else:
        return decorator(func)
