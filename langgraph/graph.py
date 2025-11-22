END = "__end__"


class StateGraph:
    def __init__(self, *_, **__):
        self.nodes = {}

    def add_node(self, name, node):  # pragma: no cover - helper stub
        self.nodes[name] = node
        return self

    def add_edge(self, *_, **__):  # pragma: no cover - helper stub
        return self

    def add_conditional_edges(self, *_, **__):  # pragma: no cover - helper stub
        return self

    def compile(self, *_, **__):  # pragma: no cover - helper stub
        return self

    def invoke(self, state):
        if callable(state):
            return state()
        return state
