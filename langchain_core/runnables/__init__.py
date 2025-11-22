class RunnableLambda:
    def __init__(self, func):
        self.func = func

    def invoke(self, *args, **kwargs):  # pragma: no cover - helper stub
        return self.func(*args, **kwargs)
