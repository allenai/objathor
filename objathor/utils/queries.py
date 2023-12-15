from abc import ABC, abstractmethod
from typing import Any, Literal
from io import BytesIO
import base64


Role = Literal["system", "user", "assistant"]


class Message(ABC):
    def __init__(self, content: Any, role: Role = "user"):
        self.content = content
        self.role = role

    @abstractmethod
    def gpt(self):
        raise NotImplementedError

    @abstractmethod
    def gemini(self):
        raise NotImplementedError


def encode_image_base64_and_wrap(bytes_like: BytesIO) -> str:
    # Ensure we are at the beginning of the buffer
    bytes_like.seek(0)
    # Encode the image
    base64_image = base64.b64encode(bytes_like.read()).decode("utf-8")
    # Return wrapped encoded image
    return f"data:image/jpeg;base64,{base64_image}"


class Image(Message):
    def gpt(self):
        if isinstance(self.content, str):
            url = self.content
        elif isinstance(self.content, BytesIO):
            url = encode_image_base64_and_wrap(self.content)
        else:
            raise NotImplementedError

        return dict(type="image_url", image_url=dict(url=url, detail="low"))

    def gemini(self):
        raise NotImplementedError


class Text(Message):
    def gpt(self):
        assert isinstance(self.content, str), f"Text message content must be a string"
        return dict(type="text", text=self.content)

    def gemini(self):
        raise NotImplementedError


class ComposedMessage(Message):
    def check_contents(self):
        assert isinstance(self.content, (list, tuple)) and all(
            isinstance(msg, Message) for msg in self.content
        ), f"ComposedMessage contents must be a sequence of messages"

    def gpt(self):
        self.check_contents()
        return [msg.gpt() for msg in self.content]

    def gemini(self):
        self.check_contents()
        return [msg.gemini() for msg in self.content]
