class BaseMessage:
    def __init__(self, content: str = "", name: str | None = None):
        self.content = content
        self.name = name
        self.type = "base"


class HumanMessage(BaseMessage):
    def __init__(self, content: str = "", name: str | None = None):
        super().__init__(content, name)
        self.type = "human"


class SystemMessage(BaseMessage):
    def __init__(self, content: str = "", name: str | None = None):
        super().__init__(content, name)
        self.type = "system"


class AIMessage(BaseMessage):
    def __init__(self, content: str = "", name: str | None = None):
        super().__init__(content, name)
        self.type = "ai"


def messages_to_dict(messages):
    result = []
    for msg in messages:
        result.append(
            {
                "type": getattr(msg, "type", "human"),
                "data": {"content": getattr(msg, "content", None), "name": getattr(msg, "name", None)},
            }
        )
    return result
