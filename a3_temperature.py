import fn
from anthropic.types import MessageParam

# create a list of messages
messages: list[MessageParam] = []

# make the message
fn.add_user_message(
    messages,
    "Write me a one sentence pitch for a book",
)

# get the response for 0
response: str = fn.chat(
    messages,
    temperature=0,
)
print("temperature is 0: ", response)

# get the response for 1.0
response: str = fn.chat(
    messages,
    temperature=1.0,
)
print("---------")
print("temperature is 1: ", response)
