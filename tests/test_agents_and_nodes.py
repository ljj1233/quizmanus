import pytest
from langchain_core.messages import HumanMessage

from src.graph.agents import agents as agents_module
from src.graph.nodes import nodes
from src.graph.nodes import rag_nodes


class DummyResult:
    def __init__(self, content):
        self.content = content


class DummyLLM:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = 0

    def invoke(self, _messages):
        content = self.contents[min(self.calls, len(self.contents) - 1)]
        self.calls += 1
        return DummyResult(content)

    def stream(self, _messages):
        yield from (DummyResult(c) for c in self.contents)


class DummyRagGraph:
    def __init__(self):
        self.calls = 0

    def invoke(self, _state):
        self.calls += 1
        return {"existed_qa": [f"generated-{self.calls}"]}


def test_extract_question_fingerprint_basic():
    state = {
        "next_work": "title: Optics\ndescription: Lens behavior",
        "rag": {"type": "单选题"},
        "meta_history": [],
    }

    fingerprint, duplicate = rag_nodes.extract_question_fingerprint(state, "题干；...参考答案")

    assert duplicate is False
    assert fingerprint["id"] == 1
    assert fingerprint["topic"] == "Optics"
    assert fingerprint["focus"] == "Lens behavior"
    assert fingerprint["question_type"] == "单选题"


def test_extract_question_fingerprint_detects_duplicates():
    existing = [{"id": 3, "topic": "Optics", "focus": "Lens behavior", "question_type": "单选题"}]
    state = {
        "next_work": "title: Optics\ndescription: Lens behavior",
        "rag": {"type": "单选题"},
        "meta_history": existing,
    }

    fingerprint, duplicate = rag_nodes.extract_question_fingerprint(state, "题干；...参考答案")

    assert duplicate is True
    assert fingerprint == existing[0]


def test_rag_generator_appends_meta_history(monkeypatch):
    monkeypatch.setattr(rag_nodes, "generator_model", "gemini")
    fake_llm = DummyLLM(["题干；测试题目\n参考答案：A"])
    monkeypatch.setattr(rag_nodes, "get_llm_by_type", lambda **_: fake_llm)

    state = {
        "existed_qa": [],
        "next_work": "title: Gravity\ndescription: Free fall",
        "meta_history": [],
        "rag": {
            "reranked_docs": ["doc"],
            "retrieved_docs": ["doc"],
            "outer_knowledge": "",
            "type": "单选题",
            "subject": "physics",
            "hyde_query": "q",
            "reranker_model": object(),
            "embedding_model": object(),
            "enable_browser": False,
            "get_input": False,
        },
    }

    command = rag_nodes.rag_generator(state)

    assert command.update["meta_history"][0]["topic"] == "Gravity"
    assert command.update["meta_history"][0]["id"] == 1
    assert command.update["latest_fingerprint"]["focus"] == "Free fall"


def test_rag_generator_skips_duplicate_meta(monkeypatch):
    monkeypatch.setattr(rag_nodes, "generator_model", "gemini")
    fake_llm = DummyLLM(["题干；测试题目\n参考答案：A"])
    monkeypatch.setattr(rag_nodes, "get_llm_by_type", lambda **_: fake_llm)

    meta_entry = {"id": 2, "topic": "Gravity", "focus": "Free fall", "question_type": "单选题"}
    state = {
        "existed_qa": [],
        "next_work": "title: Gravity\ndescription: Free fall",
        "meta_history": [meta_entry],
        "rag": {
            "reranked_docs": ["doc"],
            "retrieved_docs": ["doc"],
            "outer_knowledge": "",
            "type": "单选题",
            "subject": "physics",
            "hyde_query": "q",
            "reranker_model": object(),
            "embedding_model": object(),
            "enable_browser": False,
            "get_input": False,
        },
    }

    command = rag_nodes.rag_generator(state)

    assert command.update["meta_history"] == []
    assert command.update["latest_fingerprint"] == meta_entry



def test_create_agent_uses_factory(monkeypatch):
    captured = {}

    def fake_prompt_template(agent_type, state):
        return f"prompt for {agent_type}: {state['value']}"

    def fake_react_agent(llm, tools, prompt):
        captured["llm"] = llm
        captured["tools"] = tools
        captured["prompt_output"] = prompt({"value": "state"})
        return "created-agent"

    monkeypatch.setattr(agents_module, "apply_prompt_template", fake_prompt_template)
    monkeypatch.setattr(agents_module, "create_react_agent", fake_react_agent)
    monkeypatch.setattr(agents_module, "get_llm_by_type", lambda agent_llm_type: f"llm-{agent_llm_type}")

    created = agents_module.create_agent("planner", "browser_generator", ["tool"], "template")

    assert created == "created-agent"
    assert captured["llm"] == "llm-planner"
    assert captured["tools"] == ["tool"]
    assert captured["prompt_output"] == "prompt for browser_generator: state"


def test_rag_hyde_updates_state(monkeypatch):
    fake_llm = DummyLLM(['{"hyde_query": "expanded"}'])
    monkeypatch.setattr(rag_nodes, "get_llm_by_type", lambda **_: fake_llm)

    command = rag_nodes.rag_hyde({"next_work": "question", "rag": {}})

    assert command.goto == "router"
    assert command.update["rag"]["hyde_query"] == "expanded"


def test_fill_missing_questions_generates_from_planned_steps(monkeypatch):
    monkeypatch.setattr(nodes, "generator_model", "mock-model")
    fake_rag_graph = DummyRagGraph()

    state = {
        "rag_graph": fake_rag_graph,
        "rag": {"reranked_docs": ["doc"], "retrieved_docs": ["doc"], "outer_knowledge": "", "type": "单选题", "subject": "math"},
        "planned_generator_steps": [
            {"agent_name": "rag_er", "title": "Q1", "description": "desc1", "note": ""},
            {"agent_name": "rag_er", "title": "Q2", "description": "desc2", "note": ""},
        ],
        "planned_question_count": 2,
        "existed_qa": ["already"],
        "next_work": "placeholder",
    }

    questions, messages, failed = nodes.fill_missing_questions(state)

    assert questions == ["generated-1"]
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].name == "rag_er"
    assert "Q2" in messages[0].content
    assert failed == []
    assert fake_rag_graph.calls == 1


def test_main_supervisor_writes_fallback_when_no_report(monkeypatch, tmp_path):
    dummy_llm = DummyLLM(['{"next": "FINISH", "next_step_content": ""}'])
    monkeypatch.setattr(nodes, "get_llm_by_type", lambda *_args, **_kwargs: dummy_llm)
    monkeypatch.setattr(nodes, "fill_missing_questions", lambda _state: ([], [], []))
    monkeypatch.setattr(
        nodes,
        "apply_prompt_template",
        lambda *_args, **_kwargs: [HumanMessage(content="summary", name="planner")],
    )

    quiz_file = tmp_path / "quiz.md"
    state = {
        "messages": [HumanMessage(content="fallback report", name="planner")],
        "rag": {},
        "quiz_url": str(quiz_file),
        "existed_qa": [],
        "pending_generator_steps": [],
    }

    command = nodes.main_supervisor(state)

    assert command.goto == "__end__"
    assert quiz_file.exists()
    assert quiz_file.read_text() == "fallback report"


def test_rag_reranker_survives_browser_failure(monkeypatch):
    monkeypatch.setattr(rag_nodes, "rerank", lambda **_: ["doc1"])

    class FailingBrowser:
        def invoke(self, _state):
            raise RuntimeError("boom")

    monkeypatch.setattr(rag_nodes, "knowledge_based_browser", FailingBrowser())

    state = {
        "rag": {
            "hyde_query": "query",
            "retrieved_docs": ["doc"],
            "reranker_model": object(),
            "enable_browser": True,
        }
    }

    command = rag_nodes.rag_reranker(state)

    assert command.goto == "generator"
    assert command.update["rag"]["outer_knowledge"] == ""
