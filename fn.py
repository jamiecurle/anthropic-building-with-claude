from dotenv import load_dotenv
from anthropic import Anthropic
from anthropic.types import MessageParam, TextBlock, Message


load_dotenv()
client = Anthropic()


def inspect(content: str, label: str | None = None) -> None:
    if label:
        print(label, ":\n")
    print(content, "\n----------\n\n")


def add_user_message(messages: list[MessageParam], content: str) -> None:
    """Peasant context simulation"""
    user_message: MessageParam = {"role": "user", "content": content}
    messages.append(user_message)


def add_assistant_message(messages: list[MessageParam], content: str) -> None:
    """Peasant application management"""
    assistant_message: MessageParam = {"role": "assistant", "content": content}
    messages.append(assistant_message)


def chat(
    messages: list[MessageParam],
    system: str | None = None,
    temperature: float = 0.2,
    stop_sequences: list[str] = [],
) -> str:
    """Basic text response handling"""
    message: Message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1000,
        messages=messages,
        system=system or "",
        temperature=0.3,
        stop_sequences=stop_sequences,
    )

    # make sure we're returning something capable of having text type
    block = message.content[0]
    assert isinstance(block, TextBlock)
    return block.text
