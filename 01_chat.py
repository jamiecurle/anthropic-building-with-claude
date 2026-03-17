import fn
from anthropic.types import MessageParam

# create a list of messages
messages: list[MessageParam] = []

# some basic UX
print("What do you want to ask?")

# enter into a loop of question, response, update peasant context
# sidenote: obviously this is untrused, unsanitised input.
while True:
    user_input = input(">")
    fn.add_user_message(messages, user_input)
    response: str = fn.chat(messages)
    fn.add_assistant_message(messages, response)
    print(response)
