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


def test_main_critic_passes_and_tracks_fingerprint():
    state = {
        "rag": {"enable_browser": False},
        "existed_qa": ["question"],
        "current_generator_step": {
            "agent_name": "rag_er",
            "fingerprint": "fp-1",
            "title": "Q1",
            "difficulty": "easy",
        },
        "question_fingerprints": [],
    }

    command = nodes.main_critic(state)

    assert command.goto == "supervisor"
    assert command.update["critic_result"] == {"status": "passed", "feedback": ""}
    assert "fp-1" in command.update["question_fingerprints"]


def test_main_critic_rejects_and_reroutes_to_generator():
    state = {
        "rag": {"enable_browser": False},
        "existed_qa": ["another question"],
        "current_generator_step": {
            "agent_name": "rag_and_browser",
            "fingerprint": "dup",
            "description": "redo with sources",
            "title": "Q2",
        },
        "question_fingerprints": ["dup"],
        "pending_generator_steps": [],
        "generator_retry_counts": {},
    }

    command = nodes.main_critic(state)

    assert command.goto == "rag_and_browser"
    assert command.update["critic_result"]["status"] == "rejected"
    assert command.update["pending_generator_steps"][0]["feedback"].startswith("题目指纹重复")
    assert command.update["next_work"] == "redo with sources"
    assert command.update["rag"]["enable_browser"] is True


def test_main_critic_stops_after_max_retries():
    state = {
        "rag": {},
        "existed_qa": ["question"],
        "current_generator_step": {
            "agent_name": "rag_er",
            "fingerprint": "dup",
            "title": "Q1",
        },
        "question_fingerprints": ["dup"],
        "pending_generator_steps": [],
        "generator_retry_counts": {"Q1": nodes.MAX_CRITIC_RETRIES - 1},
    }

    command = nodes.main_critic(state)

    assert command.goto == "supervisor"
    assert command.update["critic_result"]["status"] == "rejected"
    assert command.update["generator_retry_counts"]["Q1"] == nodes.MAX_CRITIC_RETRIES
    assert command.update.get("pending_generator_steps", []) == []


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
