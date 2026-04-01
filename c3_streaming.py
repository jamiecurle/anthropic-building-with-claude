from anthropic.types import MessageParam

import fn2

from anthropic.types import MessageParam

import fn2


def yo_claude(message):
    # create a list of messages
    messages: list[MessageParam] = []

    # create the user message as normal
    fn2.add_user_message(messages, message=message)

    # run the conversation
    response = fn2.run_conversation(messages)

    # extract the text
    fn2.text_from_message(response)
