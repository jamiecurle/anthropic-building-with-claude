import fn
from anthropic.types import MessageParam

# create a list of messages
messages: list[MessageParam] = []

# make the message
fn.add_user_message(
    messages, "Write a python function that checks a string for duplicate characters"
)

# make the system prompt
system = """
You are an expert python programmer. Your answers should be very short.
"""

response: str = fn.chat(messages, system=system)
print(response)
