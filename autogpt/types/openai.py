"""Type helpers for working with the OpenAI library"""
from __future__ import annotations

from datetime import datetime
from uuid import uuid4


class Message:
    """OpenAI Message object containing a role and the message content"""

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
        self.timestamp = datetime.now()
        self.id = uuid4().hex

    def to_dict(self) -> dict:
        """Convert the message to a dictionary"""
        return {
            "role": self.role,
            "content": self.content,
        }


def ensure_messages(messages: list[Message | dict]) -> list[Message]:
    """Ensure that the given messages are Message objects

    Args:
        messages: The messages to ensure

    Returns:
        The messages as Message objects
    """
    message_objs = []
    for message in messages:
        if not isinstance(message, Message):
            message_objs.append(
                Message(message["role"], message["content"])
            )
        else:
            message_objs.append(message)
    return message_objs
