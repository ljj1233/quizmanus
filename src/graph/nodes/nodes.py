import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, TypedDict, Literal, Optional
from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage,messages_to_dict
from .quiz_types import State
from langgraph.types import Command
from ..prompts.prompts import get_prompt_template,apply_prompt_template
from ..llms.llms import get_llm_by_type
from ...config.nodes import TEAM_MEMBERS
from ..agents.agents import browser_generator, knowledge_searcher
from ..tools.search import tavily_tool
from ...config.llms import llm_type,generator_model,reporter_llm_type,planner_llm_type,supervisor_llm_type
from copy import deepcopy
import json
import logging
from ...utils import get_json_result,call_Hkust_api
from ...config.rag import SUBJECTS
import re
# Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
# )

logger = logging.getLogger(__name__)


MAX_CRITIC_RETRIES = 2


@dataclass
class PlanStep:
    agent_name: Literal["rag_er", "rag_and_browser"]
    title: str
    description: str
    note: str = ""
    subject: Optional[str] = None
    question_type: Optional[str] = None


@dataclass
class QuizPlan:
    subject: str
    steps: List[PlanStep] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict) -> "QuizPlan":
        if data.get("subject") not in SUBJECTS:
            raise ValueError("Planner subject is not in configured SUBJECTS")
        steps = []
        for step in data.get("steps", []):
            steps.append(PlanStep(**step))
        if not steps:
            raise ValueError("Planner returned empty steps")
        return cls(subject=data["subject"], steps=steps)

    def to_json(self) -> str:
        serializable = {
            "subject": self.subject,
            "steps": [step.__dict__ for step in self.steps],
        }
        return json.dumps(serializable, ensure_ascii=False, indent=2)


class SupervisorDecisionDict(TypedDict):
    next_action: str
    missing_points: List[str]
    instruction: str


class SupervisorDecision(BaseModel):
    next_action: str = Field(..., alias="next_action")
    missing_points: List[str] = Field(default_factory=list)
    instruction: str = ""

    class Config:
        populate_by_name = True
        extra = "ignore"


DEFAULT_SUPERVISOR_DECISION = SupervisorDecision(next_action="planner")

def _summarize_fingerprint(fingerprint: Optional[Dict], fallback: str) -> str:
    if not fingerprint:
        return f"题目概述：{fallback[:100]}"

    focus = str(fingerprint.get("focus", ""))[:80]
    topic = fingerprint.get("topic", "")
    question_type = fingerprint.get("question_type", "")
    return f"题目指纹#{fingerprint.get('id', '?')}: {topic} | {focus} | 题型：{question_type}"

def main_coordinator(state: State) -> Command[Literal["planner", "__end__"]]:
    """Coordinator node that communicate with customers."""
    logger.info("Coordinator talking.")
    system_message = get_prompt_template("coordinator")
    messages = [
        SystemMessage(
            content = system_message
        ),
        HumanMessage(content=f'''当前查询：{state["ori_query"]}''')
    ]
    
    response_content = re.sub(r'<think>.*?</think>', '', get_llm_by_type(type = llm_type).invoke(messages).content, flags=re.DOTALL).strip()
    logger.info(f"Current state messages: {state['messages']}")
    # 尝试修复可能的JSON输出
    logger.info(f"Coordinator response: {response_content}")

    goto = "__end__"
    if "handoff_to_planner" in response_content:
        goto = "planner"

    # 更新response.content为修复后的内容
    # response.content = response_content

    return Command(
        goto=goto,
    )
def main_planner(state: State):
    system_message = get_prompt_template("planner", SUBJECT="，".join(SUBJECTS))
    messages = [
        SystemMessage(
            content=system_message
        ),
        HumanMessage(content=f'''当前查询：{state["ori_query"]}''')
    ]
    llm = get_llm_by_type(type=planner_llm_type)

    if state.get("search_before_planning"):
        searched_content = str(knowledge_searcher.invoke(state)["messages"][-1].content)
        messages = deepcopy(messages)
        messages[
            -1
        ].content += f"\n\n# 相关搜索结果\n\n{searched_content}"

    parser = JsonOutputParser()
    parsed_plan: QuizPlan
    for i in range(3):
        try:
            raw_plan = re.sub(
                r'<think>.*?</think>', '', llm.invoke(messages).content, flags=re.DOTALL
            ).strip()
            if raw_plan.startswith("```json"):
                raw_plan = raw_plan.removeprefix("```json")
            if raw_plan.endswith("```"):
                raw_plan = raw_plan.removesuffix("```")
            parsed_dict = parser.parse(raw_plan)
            parsed_plan = QuizPlan.from_dict(parsed_dict)
            break
        except Exception as e:
            logger.warning(f"plan生成报错：{e}，重试第{i+1}次。")
            if i == 2:
                raise

    logger.info("Planner response: %s", parsed_plan.to_json())

    generator_agents = {"rag_er", "rag_and_browser"}
    need_to_generate = [
        step.dict()
        for step in parsed_plan.steps
        if step.agent_name in generator_agents
    ]

    full_response = parsed_plan.to_json()
    existed_qa, new_messages, failed_steps = run_generator_concurrently(state, need_to_generate)

    planner_message = HumanMessage(content=full_response, name="planner")
    all_messages = [planner_message]
    all_messages.extend(new_messages)

    return Command(
        update={
            "messages": all_messages,
            "full_plan": full_response,
            "planned_generator_steps": need_to_generate,
            "planned_question_count": len(need_to_generate),
            "pending_generator_steps": failed_steps,
            "existed_qa": state.get("existed_qa", []) + existed_qa,
            "rag": {**state.get("rag", {}), "subject": parsed_plan.subject},
        },
        goto="supervisor",
    )

async def _generate_single(needi, state: State):
    updated_rag = {
        **state["rag"],
        "enable_browser": False if needi["agent_name"] == "rag_er" else True,
    }
    updated_rag["get_input"] = generator_model == "qwen"
    next_step_content = f"title: {needi['title']}\ndescription: {needi['description']}"
    if needi.get("note") and needi["note"].strip():
        next_step_content += f"\nnote:{needi['note']}"

    needi_state = {**state}
    needi_state["next_work"] = next_step_content
    needi_state["rag"] = updated_rag

    try:
        if hasattr(state["rag_graph"], "ainvoke"):
            rag_state = await state["rag_graph"].ainvoke(needi_state)
        else:
            rag_state = await asyncio.to_thread(state["rag_graph"].invoke, needi_state)
        qa_payload = rag_state["existed_qa"][-1]
        fingerprint = rag_state.get("latest_fingerprint")
        if not fingerprint:
            history = rag_state.get("meta_history") or []
            fingerprint = history[-1] if history else None
        message = HumanMessage(
            content=_summarize_fingerprint(fingerprint, next_step_content),
            name=needi["agent_name"],
        )
        return qa_payload, message, None
    except Exception as exc:  # pragma: no cover - logged and surfaced via failed_steps
        logger.error("asyncio_generator error: %s", exc)
        return None, None, needi


def run_generator_concurrently(state: State, need_to_generate: List):
    """Use asyncio gather to parallelize question generation."""

    async def _runner():
        tasks = [asyncio.create_task(_generate_single(needi, state)) for needi in need_to_generate]
        return await asyncio.gather(*tasks)

    results = asyncio.run(_runner()) if need_to_generate else []

    existed_qa: List = []
    messages: List = []
    failed_steps: List = []
    inputs: List = []

    for qa_payload, message, failed in results:
        if failed is not None:
            failed_steps.append(failed)
            continue
        if generator_model == "qwen":
            inputs.append(qa_payload)
        else:
            existed_qa.append(qa_payload)
        messages.append(message)

    if generator_model == "qwen" and inputs:
        existed_qa = get_llm_by_type(
            type="qwen", model=state["generate_model"], tokenizer=state["generate_tokenizer"]
        ).invoke(inputs)

    return existed_qa, messages, failed_steps


def _clean_supervisor_response(raw_response: str) -> str:
    cleaned = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json")
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```")
    if cleaned.endswith("```"):
        cleaned = cleaned.removesuffix("```")
    return cleaned.strip()


def _parse_supervisor_decision(
    response_content: str, parser: PydanticOutputParser
) -> SupervisorDecision:
    cleaned = _clean_supervisor_response(response_content)
    parsed = parser.parse(cleaned)
    if not getattr(parsed, "next_action", None) or getattr(parsed, "next_action") is Ellipsis:
        raise ValidationError("missing next_action")
    if not hasattr(parsed, "missing_points") or parsed.missing_points is Ellipsis:
        parsed.missing_points = []
    if not hasattr(parsed, "instruction") or parsed.instruction is Ellipsis:
        parsed.instruction = ""
    return parsed


def _merge_missing_points(state: State, decision_missing: List[str]) -> List[str]:
    required_points = state.get("required_knowledge_points") or []
    covered_points = set(state.get("covered_knowledge_points") or [])
    remaining_required = [p for p in required_points if p not in covered_points]
    combined: List[str] = []
    for point in [*decision_missing, *remaining_required]:
        if point and point not in combined:
            combined.append(point)
    return combined


def _build_next_work(
    instruction: str, missing_points: List[str], pending_step: Optional[Dict] = None
) -> str:
    parts: List[str] = []
    if pending_step:
        parts.append(pending_step.get("description", ""))
        note = pending_step.get("note") or ""
        if note.strip():
            parts.append(f"note:{note}")
    if instruction:
        parts.append(instruction)
    if missing_points:
        parts.append(f"请优先覆盖以下缺失的知识点：{', '.join(missing_points)}")
    return "\n".join([part for part in parts if part])


def _choose_generator_target(pending_steps: List[Dict]) -> Dict:
    if pending_steps:
        return pending_steps[0]
    return {"agent_name": "rag_er", "description": "补充生成一道覆盖缺失知识点的题目"}


def fill_missing_questions(state: State):
    planned_steps = state.get("planned_generator_steps", [])
    produced = len(state.get("existed_qa", []))
    planned_count = state.get("planned_question_count") or len(planned_steps)

    pending_steps = state.get("pending_generator_steps", [])
    if not pending_steps and planned_count and produced < planned_count:
        pending_steps = planned_steps[produced:]

    if not pending_steps:
        return [], [], []

    new_questions, new_messages, failed_steps = run_generator_concurrently(state, pending_steps)
    return new_questions, new_messages, failed_steps


def _step_key(step: Dict) -> str:
    """Generate a stable key for tracking retries of a generator step."""

    return step.get("title") or step.get("description") or step.get("agent_name") or "unknown"


def main_critic(state: State) -> Command[Literal["supervisor", "rag_er", "rag_and_browser"]]:
    """Critic node to validate generated questions before reporting.

    Performs deduplication via fingerprints, checks for empty questions,
    and validates the declared difficulty if provided. Returns feedback and
    reroutes to the generator when remediation is possible.
    """

    current_step = state.get("current_generator_step", {}) or {}
    fingerprint = state.get("latest_fingerprint") or current_step.get("fingerprint")
    question_fingerprints = list(state.get("question_fingerprints", []))
    retry_counts = dict(state.get("generator_retry_counts", {}))
    question = state.get("existed_qa", [])[-1] if state.get("existed_qa") else ""

    status: Literal["passed", "rejected"] = "passed"
    feedback = ""

    if not question:
        status = "rejected"
        feedback = "生成的题目内容为空，无法通过审核。"
    elif fingerprint and fingerprint in question_fingerprints:
        status = "rejected"
        feedback = "题目指纹重复，疑似生成了重复题目。"
    elif current_step.get("difficulty") and current_step["difficulty"] not in {"easy", "medium", "hard"}:
        status = "rejected"
        feedback = f"题目难度{current_step['difficulty']}不符合要求。"

    if status == "passed":
        if fingerprint:
            question_fingerprints.append(fingerprint)
        return Command(
            goto="supervisor",
            update={
                "critic_result": {"status": status, "feedback": feedback},
                "question_fingerprints": question_fingerprints,
            },
        )

    step_identifier = _step_key(current_step)
    retry_counts[step_identifier] = retry_counts.get(step_identifier, 0) + 1

    updates: Dict = {
        "critic_result": {"status": status, "feedback": feedback},
        "generator_retry_counts": retry_counts,
    }

    if retry_counts[step_identifier] >= MAX_CRITIC_RETRIES:
        # Stop retrying and hand control back to supervisor for a decision.
        return Command(goto="supervisor", update=updates)

    pending_steps = list(state.get("pending_generator_steps", []))
    retry_step = {**current_step, "feedback": feedback}
    pending_steps.insert(0, retry_step)

    updates["pending_generator_steps"] = pending_steps

    target = current_step.get("agent_name", "supervisor")
    if target in {"rag_er", "rag_and_browser"}:
        updates.update(
            {
                "next": target,
                "next_work": retry_step.get("description", ""),
                "rag": {
                    **state["rag"],
                    "enable_browser": target == "rag_and_browser",
                },
            }
        )

    return Command(goto=target, update=updates)

    
RESPONSE_FORMAT = "{}的回复:\n\n<response>\n{}\n</response>\n\n*请执行下一步.*"
def main_supervisor(state: State) -> Command[Literal[*TEAM_MEMBERS, "__end__"]]:
    """Supervisor node that decides which agent should act next."""
    logger.info("Supervisor evaluating next action")
    parser = PydanticOutputParser(pydantic_object=SupervisorDecision)
    trimmed_state = {**state, "messages": state.get("messages", [])[-10:]}
    messages = apply_prompt_template("supervisor", trimmed_state)
    # preprocess messages to make supervisor execute better.
    messages = deepcopy(messages)
    reports = []
    response_content = ""
    for message in messages:
        if isinstance(message, BaseMessage) and message.name in TEAM_MEMBERS:
            if message.name == "reporter":
                reports.append(message.content)
            message.content = RESPONSE_FORMAT.format(message.name, message.content)
    goto = "__end__"
    next_step_content = ""
    missing_points: List[str] = []
    decision: Optional[SupervisorDecision] = None
    for i in range(3):
        try:
            if len(messages)>119:
                dict_messages = messages_to_dict(messages)
                role_mapping = {
                    "system": "system",
                    "human": "user",
                    "ai": "assistant"
                }
                openai_format = [
                    {"role": role_mapping[msg["type"]], "content": msg['data']["content"],"name":msg['data']['name']}
                    for msg in dict_messages
                ]
                logger.info("使用hkust-deepseek-r1")
                response = call_Hkust_api(prompt = "",messages = openai_format)
                response_content = json.dumps(response, ensure_ascii=False) if isinstance(response, dict) else str(response)
            else:
                response = get_llm_by_type(supervisor_llm_type).invoke(messages).content
                response_content = _clean_supervisor_response(response)
            decision = _parse_supervisor_decision(response_content, parser)
            goto = decision.next_action
            next_step_content = decision.instruction
            missing_points = decision.missing_points
            break
        except (ValidationError, Exception) as e:
            logger.warning(f"supervisor出错了：{e}")
            decision = None
    logger.info(f"Current state messages: {state['messages']}")
    logger.info(f"Supervisor response: {response_content}")

    if decision is None:
        decision = DEFAULT_SUPERVISOR_DECISION
        goto = decision.next_action
        next_step_content = decision.instruction
        missing_points = decision.missing_points

    updates = {}
    if goto == "FINISH":
        extra_questions, extra_messages, failed_steps = fill_missing_questions(state)
        if extra_questions or extra_messages:
            updates["existed_qa"] = state.get("existed_qa", []) + extra_questions
            updates["messages"] = state.get("messages", []) + extra_messages
            updates["pending_generator_steps"] = failed_steps

        planned_count = state.get("planned_question_count", 0)
        produced_count = len(updates.get("existed_qa", state.get("existed_qa", [])))
        pending_steps = updates.get("pending_generator_steps") or state.get("pending_generator_steps", [])
        merged_missing = _merge_missing_points(state, missing_points)

        needs_more_questions = bool(planned_count and produced_count < planned_count)
        has_missing_points = bool(merged_missing)

        if needs_more_questions or has_missing_points:
            if needs_more_questions:
                logger.info(
                    "Not enough questions generated (have %s / target %s); routing to generator.",
                    produced_count,
                    planned_count,
                )
            if has_missing_points:
                logger.info("Missing required knowledge points: %s", merged_missing)
            next_pending = _choose_generator_target(pending_steps)
            goto = next_pending["agent_name"]
            next_step_content = _build_next_work(next_step_content, merged_missing, next_pending)
            missing_points = merged_missing
        else:
            goto = "__end__"
            report_content = reports[-1] if reports else ""
            if not report_content and state.get("messages"):
                last_message = state["messages"][-1]
                if isinstance(last_message, dict):
                    report_content = last_message.get("content", "")
                else:
                    report_content = getattr(last_message, "content", "")
            if report_content:
                with open(state['quiz_url'], "w", encoding="utf-8") as f:
                    f.write(report_content)
            else:
                logger.warning("No reporter output available; skipping report write for %s", state.get("quiz_url"))
            logger.info("Workflow completed")
    else:
        logger.info(f"Supervisor delegating to: {goto}")
    if goto == "rag_er":
        updated_rag = {
            **state['rag'],
            "enable_browser": False
        }
    elif goto == "rag_and_browser":
        updated_rag = {
            **state['rag'],
            "enable_browser": True
        }
    else:
        updated_rag = {
            **state['rag']
        }
    if missing_points:
        updates["missing_points"] = missing_points

    updates.update({"next": goto, "next_work": next_step_content, "rag":updated_rag})
    return Command(goto=goto, update=updates)


def main_browser_generator(state: State) -> Command[Literal["supervisor"]]:
    """Node for the browser agent that performs web browsing tasks."""
    logger.info("Browser agent starting task")
    for i in range(3):
        try: 
            result = browser_generator.invoke(state)
            logger.info("Browser agent completed task")
            response_content = result["messages"][-1].content
            break
        except Exception as e:
            logger.error(f"Browser agent failed with error: {e}")
            response_content = ""
            return Command(goto="__end__")
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response_content,
                    name="browser_generator",
                )
            ]
        },
        goto="supervisor",
    )


def main_rag(state: State) -> Command[Literal["critic"]]:
    """Node for the RAG that performs RAG tasks."""
    logger.info("Browser agent starting task")
    rag_state = state['rag_graph'].invoke(state)
    new_qa = str(rag_state['existed_qa'][-1])
    rag_fingerprint = rag_state.get("latest_fingerprint")
    if not rag_fingerprint:
        history = rag_state.get("meta_history") or []
        rag_fingerprint = history[-1] if history else None
    new_q = _summarize_fingerprint(rag_fingerprint, state['next_work'])
    # if "参考答案" in new_qa:
    #     new_q = new_qa.split("参考答案")[0].strip()
    # elif "答案" in new_qa:
    #     new_q = new_qa.split("答案")[0].strip()
    # else:
    #     new_q = new_qa
    logger.info("RAG agent completed task")
    # 尝试修复可能的JSON输出
    # response_content = repair_json_output(response_content)
    logger.info(f"RAG agent response: {new_qa}")
    return Command(
        update={
            "existed_qa": [new_qa],
            "messages": [
                HumanMessage(
                    content=new_q,
                    name="rag_er",
                )
            ],
            "latest_fingerprint": rag_fingerprint,
        },
        goto="critic",
    )

def main_rag_browser(state: State) -> Command[Literal["critic"]]:
    """Node for the RAG that performs RAG tasks."""
    logger.info("Browser agent starting task")
    rag_state = state['rag_graph'].invoke(state)
    new_qa = str(rag_state['existed_qa'][-1])
    # if "参考答案" in new_qa:
    #     new_q = new_qa.split("参考答案")[0].strip()
    # elif "答案" in new_qa:
    #     new_q = new_qa.split("答案")[0].strip()
    # else:
    #     new_q = new_qa
    rag_fingerprint = rag_state.get("latest_fingerprint")
    if not rag_fingerprint:
        history = rag_state.get("meta_history") or []
        rag_fingerprint = history[-1] if history else None
    new_q = _summarize_fingerprint(rag_fingerprint, state['next_work'])
    logger.info("RAG agent completed task")
    # 尝试修复可能的JSON输出
    # response_content = repair_json_output(response_content)
    logger.info(f"RAG agent response: {new_qa}")
    return Command(
        update={
            "existed_qa": [new_qa],
            "messages": [
                HumanMessage(
                    content=new_q,
                    name="rag_and_browser",
                )
            ],
            "latest_fingerprint": rag_fingerprint,
        },
        goto="critic",
    )


def main_reporter(state: State) -> Command[Literal["supervisor"]]:
    """Reporter node that write a final report."""
    logger.info("Reporter write final report")
    meta_history = state.get("meta_history", [])
    if meta_history:
        summary_lines = [_summarize_fingerprint(meta, meta.get("topic", "")) for meta in meta_history]
        payload = "\n".join(summary_lines)
    else:
        payload = '\n\n\n\n'.join(state.get('existed_qa', []))
    tmp_state = {
        "messages":[
            state['messages'][0],
            state['messages'][1],
            HumanMessage(content = payload)
        ]
    }
    messages = apply_prompt_template("reporter", tmp_state)
    response_content = re.sub(r'<think>.*?</think>', '', get_llm_by_type(reporter_llm_type).invoke(messages).content, flags=re.DOTALL).strip()
    logger.info(f"Current state messages: {state['messages']}")
    # 尝试修复可能的JSON输出
    # response_content = repair_json_output(response_content)
    logger.info(f"reporter response: {response_content}")

    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response_content,
                    name="reporter",
                )
            ]
        },
        goto="supervisor",
    )