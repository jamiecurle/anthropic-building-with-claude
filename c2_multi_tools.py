from typing import Callable
from datetime import datetime
from anthropic.types import MessageParam, ToolParam

import fn2


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
    fn2.add_user_message(
        messages,
        message="What is the exact time formatted as HH:MM:SS? When you've figured it out, give me the time and a quick poem related to the time",
    )

    # call with the schema - response has the tool use block
    response = fn2.chat(messages, tools=[get_current_datetime_schema])

    # add the response into the messages
    fn2.add_assistant_message(messages, response)

    print(response)
