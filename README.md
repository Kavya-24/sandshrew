# Sandshrew

A lightweight Python library using a decorator-based approach to:
- Prepare **LLM Provider Agnostic Tool-Function Descriptions** without duplication using Pydantic Field Notation
- Execute tool_calls `Sequentially` as well as `In Parallel` depending on use case
- Internal retry mechanism with a very consistent tool output consisting of Tool-Response and Error Response
- Support for using **injected-state** inside tools which are not generated as inputs from LLM calls
- Helpers for **LLM Provider Agnostic Tool-Call Extractions and Finish Loop**


## Quick Start


### Installation
Install Python Package from https://pypi.org/project/sandshrew/

Create virtual environment 
```
python3 -m venv venv 
source venv/bin/activate
```

```
pip install sandshrew
```
Only internal dependency needed is Pydantic as of 17/12/2025


### Defining Tools (`@sand_tool`)
For the function that needs to be converted to a tool, attach @sand_tool to it.

##### Simple Tool Example
```python
from sandshrew import sand_tool
from pydantic import Field

@sand_tool(
    name="custom_name",                   # Custom name for the tool (defaults to function name)
    description="custom description",     # Custom description (defaults to function docstring)
    tags=["tag1", "tag2"],                # Optional List of tags for categorizing the tool
    retry_count=3,                        # Number of retries on failure (default: 0)
    timeout=30.0,                         # Execution timeout in seconds (default: None)
    inject_state=False,                   # Whether to inject state as first parameter (default: False)
)
def add(a: int = Field(description="First number"),
        b: int = Field(description="Second number"),
) -> int:
    """Add two numbers together."""
    return a + b

```

##### Optional Arguments
1. `name` - Custom name for the tool (defaults to function name)
2. `description` - Custom description (defaults to function docstring)
3. `tags` - List of tags for categorizing the tool
4. `retry_count` - Number of retries on failure (default: 0)
5. `timeout` - Execution timeout in seconds (default: None)
6. `inject_state` - Whether to inject state as first parameter (default: False)



##### Tool with injected state
💡 For an argument to be injected state, ensure the following:
- The argument is the first argument in the function
- The argument name begins with `_injected_state`


```python
from sandshrew import sand_tool
from pydantic import Field

@sand_tool(inject_state=True, tags=["email"])
def send_email(_injected_state: Dict[str, Any], content: str) -> str:
    """
    Send an email using injected state.
    Expected state:
        {
            "user_email": "user@example.com"
        }
    """
    user_email = _injected_state.get("user_email")
    if not user_email:
        return "no user email found in state."

    # Placeholder for actual email logic
    message = f"Sent {content} message to {user_email}..."
    return message
```



### Getting tool-descriptions (LLM Tool Representations)
`helpers.py` has provider-agnostic methods that can be used to prepare tool-descriptions for any provider.

```
from sandshrew.helpers import prepare_tools
from sandshrew.data_types import Provider


tool_list = [add, send_email]
tools = prepare_tools(Provider.OPENAI, tool_list)
```

Expected arguments:
1. `provider` - Name of the provider we want to prepare the tool-function descriptions for
2. `tool_list` - List of @sand_tool decorated BaseTool(s)

Expected Output format
1. List of Dict[str, Any] representing tool-call

### One click Tool Execution with Injected State

```
from sandshrew.helpers import Executor


provider_completion_response = client.chat.completions.create(
    model="gpt-4o-mini",
    tools=tools,
    messages=messages,
)


results = Executor(
    tool_list=tool_list,
    provider=Provider.OPENAI,
    _injected_state=_injected_state,
    use_parallel=False,
).execute(provider_completion_response)
```

Internal Flow
1. Based on provider, extracts the tool-calls if available
2. Wraps the injected state on arguments of the tool-calls that have `inject_state = True` and invokes the actual tool-calls
3. In cases of errors, retries up to `retry_count` times for the tool and then returns a defined error (See `Expected Output format`)
4. On success, returns a defined success output (See `Expected Output format`)

Expected arguments:
1. `tool_list` - List of @sand_tool decorated BaseTool(s)
2. `provider` - Name of the provider for which to prepare tool-function descriptions
3. `_injected_state` - Optional object representing the injected state the invoked tools may need
4. `use_parallel` - If number of tool_calls > 1, execute them in parallel (Default = False)



Expected Output format
1. List of ExecutionResult with the format of 


```
class ExecutionResult:
    tool_call: ToolCall
    content: Optional[Any] = None
    error: Optional[ExecutionError] = None
```

where `ToolCall` is the input ToolCall and the optional error is of the format `ExecutionError`
```
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
```

The type of `content` is Any, making it extensible for other use cases

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     LLM Provider Response                        │
│  (tool_calls with id, name, arguments from LLM)                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Executor.execute()                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1. Extract tool-calls based on provider                 │   │
│  │ 2. Inject state for tools with inject_state=True        │   │
│  │ 3. Execute sequentially or in parallel                  │   │
│  │ 4. Retry on failure (up to retry_count)                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ExecutionResult[]                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ {                                                        │   │
│  │   tool_call: ToolCall,                                  │   │
│  │   content: Any,           # Success output              │   │
│  │   error: ExecutionError   # Error details (if failed)   │   │
│  │ }                                                        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```



### Output Format: ExecutionResult
```python
@dataclass
class ExecutionResult:
    tool_call: ToolCall                    # Original tool call
    content: Optional[Any] = None          # Tool execution result
    error: Optional[ExecutionError] = None # Error if execution failed

# Unified Provider Agnostic ToolCall
@dataclass(frozen=True)
class ToolCall:
    id: str                      # Unique identifier from LLM
    name: str                    # Tool function name
    arguments: dict[str, Any]    # Arguments to pass to tool


@dataclass
class ExecutionError:
    message: str                           # Error description
    raw_error: Optional[Exception] = None  # Original exception
    retryable: bool = False                # Whether error is retryable
    backtrace: Optional[Any] = None        # Stack trace
```

## Examples

See [`example/example_tools.py`](example/example_tools.py) and [`example/main.py`](example/main.py) for complete working examples including:
- Math operations (add, subtract, multiply, divide)
- String utilities (greet, validate_email)
- Stateful tools (send_email, process_with_contextual_state)
- ReAct-style LLM interactions

![Architecture diagram](example/execution_sample.png)

## Development

### Setup

Create virtual environment 
```
python3 -m venv venv 
source venv/bin/activate
```

Install requirements

```bash
pip install -r requirements.txt
```


### Running examples
```
python -B -m example.main
```

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
ruff check .
ruff format .
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

## License

See [`LICENSE`](LICENSE) for details.

## Wiki
Article: https://goyalkavya.medium.com/using-llm-tool-executor-without-nightmares-sandshrew-858c9932ba60