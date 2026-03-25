from anthropic.types import ToolParam
from datetime import datetime


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
