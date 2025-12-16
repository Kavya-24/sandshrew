"""
Example usage of sand_tool decorator.
"""

from typing import Any, Dict, List, Literal

from pydantic import Field

from sandshrew import sand_tool

# ============================================================================
# Math Tools
# ============================================================================


@sand_tool(retry_count=1, tags=["math", "addition"])
def add(
    a: int = Field(description="First number"),
    b: int = Field(description="Second number"),
) -> int:
    """Add two numbers together."""
    return a + b


@sand_tool(retry_count=1, tags=["math", "subtraction"])
def subtract(
    a: int = Field(description="First number"),
    b: int = Field(description="Second number"),
) -> int:
    """Subtract two numbers."""
    return a - b


@sand_tool(retry_count=1, tags=["math", "multiplication"])
def multiply(
    a: int = Field(description="First number"),
    b: int = Field(description="Second number"),
) -> int:
    """Multiply two numbers."""
    return a * b


@sand_tool(retry_count=1, tags=["math", "division"])
def divide(
    a: float = Field(description="Numerator"),
    b: float = Field(description="Denominator"),
) -> float:
    """Divide two numbers.

    Raises:
        ValueError: If denominator is zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


# ============================================================================
# String / Utility Tools
# ============================================================================


@sand_tool(tags=["string"])
def greet(
    name: str = Field(description="Person's name"),
    greeting: str = Field(default="Hello", description="Greeting prefix"),
) -> str:
    """Generate a personalized greeting."""
    return f"{greeting}, {name}!"


@sand_tool(retry_count=1, tags=["validation"])
def validate_email(
    email: str = Field(description="Email address to validate"),
) -> bool:
    """Validate whether a string is a valid email format."""
    import re

    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


# ============================================================================
# Stateful Tools
# ============================================================================


@sand_tool(inject_state=True, tags=["email"])
def send_email(_injected_state: Dict[str, Any]) -> str:
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
    message = f"Sent email to {user_email}..."
    return message


# ============================================================================
# Stateful + Enum-like Operation Tool
# ============================================================================
@sand_tool(inject_state=True, tags=["stateful"])
def process_with_contextual_state(
    _injected_state: Dict[str, Any],
    column_name: Literal["response_time_ms", "num_database_calls"] = Field(
        description="Name of the column to operate on"
    ),
    operation: Literal["min", "max", "average"] = Field(
        default="average", description="Type of operation to perform"
    ),
) -> str:
    """
    Process data with access to read-only injected state
    Operate on columns and perform aggregations

    Expected state shape:
    {
        "records": [
            {"response_time_ms": 120, "num_database_calls": 3},
            ...
        ]
    }
    """
    records: List[Dict[str, Any]] = _injected_state.get("records", [])
    if not records:
        return "No records available in state"

    values = [
        r[column_name]
        for r in records
        if column_name in r and isinstance(r[column_name], (int, float))
    ]

    if not values:
        return f"No valid values found for column '{column_name}'"

    match operation:
        case "min":
            result = min(values)

        case "max":
            result = max(values)

        case "average":
            if len(values) == 0:
                return 0.0
            result = sum(values) / len(values)
        case _:
            return f"Unsupported operation: {operation}"

    return f"{operation}({column_name}) = {result}"
