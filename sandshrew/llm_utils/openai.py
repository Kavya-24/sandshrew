import json
from typing import Any, Dict, List

from ..data_types import ToolCall

"""
Provider = Provider.OPENAI
"""


class OpenAIUtils:
    @staticmethod
    def get_tool_description(name, description, params) -> Dict[str, Any]:
        """Generate OpenAI-compatible tool description from function signature and docstring."""

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": params["properties"],
                    "required": params["required"],
                },
            },
        }

    @staticmethod
    def parse_tool_calls(provider_completion_response: Any) -> List[ToolCall]:
        tool_calls = provider_completion_response.choices[0].message.tool_calls or []
        resultant_tool_calls = []

        for item in tool_calls:
            resultant_tool_calls.append(
                ToolCall(
                    name=item.function.name,
                    id=item.id,
                    arguments=json.loads(item.function.arguments),
                )
            )

        return resultant_tool_calls

    @staticmethod
    def extract_assistant_message(provider_completion_response: Any) -> Any:
        return provider_completion_response.choices[0].message.content

    @staticmethod
    def check_turn_completion(provider_completion_response: Any) -> bool:
        finish_reason = provider_completion_response.choices[0].finish_reason
        return finish_reason == "stop"
