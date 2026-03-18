"""
Use message prefilling and stop sequences ONLY to get three different commands in a single response.
There shouldn't be any comments or explaination
HINT: message prefilling isn't limited to just characters like ```
"""

import fn
from anthropic.types import MessageParam

# create a list of messages
messages: list[MessageParam] = []


# make the prompt
prompt: str = (
    "Generate three different sample AWS CLI commands. Each should be very short."
)

# make the message
fn.add_user_message(messages, prompt)

# Add the assistant message.
# This is where I had to stop and think. Old school deterministic "do it in code" mind kicked in
# and I couldn't get the answer by myself. This non-deterministic mindset is going to require some
# attention and development.
fn.add_assistant_message(
    messages,
    "Here are all three commands in a single block without any comments```bash",
)

# get the response
response: str = fn.chat(messages, stop_sequences=["```"])

# and inspect it
fn.inspect(response)
