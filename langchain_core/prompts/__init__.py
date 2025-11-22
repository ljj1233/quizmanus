class ChatPromptTemplate:
    @classmethod
    def from_template(cls, template: str):
        return cls(template)

    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs):  # pragma: no cover - helper stub
        return self.template.format(**kwargs)
