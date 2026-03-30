from typing import cast, Any
from dotenv import load_dotenv
from anthropic import Anthropic
from anthropic.types import (
    MessageParam,
    Message,
    ToolParam,
    ContentBlockParam,
    TextBlock,
)


load_dotenv()
client = Anthropic()


def inspect(content: str, label: str | None = None) -> None:
    if label:
        print(label, ":\n")
    print(content, "\n----------\n\n")


def text_from_message(message: Message) -> str:
    return "\n".join(
        [block.text for block in message.content if isinstance(block, TextBlock)]
    )


def add_user_message(messages: list[MessageParam], message: Message | str) -> None:
    """Peasant context simulation v2"""
    user_message: MessageParam = {
        "role": "user",
        "content": cast(list[ContentBlockParam], message.content)
        if isinstance(message, Message)
        else message,
    }
    messages.append(user_message)


def add_assistant_message(messages: list[MessageParam], message: Message | str) -> None:
    """Peasant application management v2"""
    assistant_message: MessageParam = {
        "role": "assistant",
        "content": cast(list[ContentBlockParam], message.content)
        if isinstance(message, Message)
        else message,
    }
    messages.append(assistant_message)


def chat(
    messages: list[MessageParam],
    system: str | None = None,
    temperature: float = 0.2,
    stop_sequences: list[str] = [],
    tools: list[ToolParam] | None = None,
) -> Message:
    """Basic text response handling"""

    # define params
    params: dict[str, Any] = {
        "model": "claude-haiku-4-5",
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature,
        "stop_sequences": stop_sequences,
    }

    if system:
        params["system"] = system

    if tools:
        params["tools"] = tools

    return client.messages.create(**params)
