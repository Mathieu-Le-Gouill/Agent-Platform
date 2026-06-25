from models.message import Message

def to_mistral(message: Message) -> ChatCompletionRequestMessage:
    ...


def from_mistral(message: ChatCompletionRequestMessage) -> Message: