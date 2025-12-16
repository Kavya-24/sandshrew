"""
Main entry point to demonstrate sand_tool examples.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

from sandshrew import (
    Executor,
    Provider,
    check_turn_completion,
    extract_assistant_message,
    prepare_tools,
)

from .example_tools import (
    add,
    divide,
    greet,
    multiply,
    process_with_contextual_state,
    send_email,
    subtract,
    validate_email,
)

load_dotenv()


def pretty_print(messages):
    for message in messages:
        print(f"{message['role']}: {message['content']}")


def chatbot(
    test_name,
    tool_list,
    system_message,
    initial_user_message,
    _injected_state={},
    use_parallel=False,
):
    print(f"\n\nRUNNING {test_name}")
    provider = Provider.OPENAI
    tools = prepare_tools(provider, tool_list)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    client = OpenAI(api_key=api_key)

    messages = [
        {
            "role": "system",
            "content": system_message,
        },
        {"role": "user", "content": initial_user_message},
    ]

    react_recurssion_limit = 10
    turns = 0

    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            tools=tools,
            messages=messages,
        )
        turns = turns + 1

        assistant_message = extract_assistant_message(provider, response)
        if assistant_message:
            messages.append(
                {"role": "assistant", "content": str(assistant_message)},
            )

        results = Executor(
            tool_list=tool_list,
            provider=Provider.OPENAI,
            _injected_state=_injected_state,
            use_parallel=use_parallel,
        ).execute(response)
        if results:
            messages.append(
                {"role": "system", "content": str(results)},
            )

        if check_turn_completion(provider, response) or turns >= react_recurssion_limit:
            break

    pretty_print(messages)


if __name__ == "__main__":
    # Example of execution of simple maths tools sequentially without any internal state access needed
    chatbot(
        "Calculator bot",
        [add, subtract, multiply, divide],
        "You are extremely proficient for doing math calculations fast",
        "What is 2 + 2 and 3+3",
        {},
    )

    #  Example of react based execution of multiple actions with injected state on invalid email
    chatbot(
        "Invalid email",
        [greet, validate_email, send_email],
        "You are an email helper. If you get any query to send emails, you should validate it first and if all looks good, you should send the email",
        "Can you send the email to user@example",
        {"user_email": "user@example.com"},
    )

    #  Example of react based execution of multiple actions with injected state on a valid email
    chatbot(
        "Valid email",
        [greet, validate_email, send_email],
        "You are an email helper. If you get any query to send emails, you should validate it first and if all looks good, you should send the email",
        "Can you send the email to user@example.com",
        {"user_email": "user@example.com"},
    )

    #  Example of react based execution of multiple actions with injected state on a valid email
    chatbot(
        "Request Analyzer",
        [process_with_contextual_state],
        "You have access to some request logs and you are quirky in answering",
        "Are there any requests that I need to be worried of having a lot of either DB calls or a lot of latency?",
        {
            "records": [
                {"response_time_ms": 120, "num_database_calls": 1},
                {"response_time_ms": 30, "num_database_calls": 2},
                {"response_time_ms": 33, "num_database_calls": 199},
                {"response_time_ms": 1111, "num_database_calls": 10},
            ]
        },
        True,
    )
