from typing import Callable
from datetime import datetime

from anthropic.types import ToolParam
from anthropic.types import MessageParam

import fn


def get_current_datetime(date_format: str = "%Y-%m-%d %H:%M:%S") -> str:
    if not date_format:
        raise ValueError("date_format cannot be empty")

    return datetime.now().strftime(date_format)


get_current_datetime_schema = ToolParam(
    {
        "name": "get_current_datetime",
        "description": "Get the current date and time formatted as a string. Returns the current datetime using Python's strftime formatting.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date_format": {
                    "type": "string",
                    "description": 'A strftime-compatible format string that controls the output format, e.g. "%Y-%m-%d %H:%M:%S" for "2026-03-25 14:30:00" or "%d/%m/%Y" for "25/03/2026". Must not be empty.',
                }
            },
        },
    }
)

TOOLS: dict[str, Callable] = {"get_current_datetime": get_current_datetime}


def call_claude():
    # create a list of messages
    messages: list[MessageParam] = []

    # create the user message as normal
    fn.add_user_message(
        messages,
        content="What is the exact time formatted as HH:MM:SS? When you've figured it out, give me the time and a quick poem related to the time",
    )

    # call with the schema - response has the tool use block
    response = fn.chat2(messages, tools=[get_current_datetime_schema])

    # add the message into the messages
    messages.append({"role": "assistant", "content": response.content})

    # get text blocks and tool blocks
    text = fn.get_text(response)
    tool_calls = fn.get_tool_calls(response)

    # if there was text, print it.
    if text is not None:
        print(text)

    # call the tools and append the messages
    for tool_call in tool_calls:
        result = TOOLS[tool_call.name](**tool_call.input)

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": result,
                        "is_error": False,
                    }
                ],
            }
        )

    # now send the final message and we should get the final response back
    # use chat and not chat2
    result = fn.chat(messages, tools=[get_current_datetime_schema])
    print(result)
