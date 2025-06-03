from typing import List, Dict, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage,messages_to_dict
from .quiz_types import State
from langgraph.types import Command
from ..prompts.prompts import get_prompt_template,apply_prompt_template
from ..llms.llms import get_llm_by_type
from ...config.nodes import TEAM_MEMBERS
from ..agents.agents import browser_generator, knowledge_searcher
from ..tools.search import tavily_tool
from copy import deepcopy
import json
import logging
import json_repair
from ...utils import get_json_result,call_Hkust_api
from ...config.llms import llm_type
from ...config.rag import SUBJECTS
# Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
# )

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def main_coordinator(state: State) -> Command[Literal["planner", "__end__"]]:
    """Coordinator node that communicate with customers."""
    logger.info("Coordinator开始工作")
    system_message = get_prompt_template("coordinator")
    messages = [
        SystemMessage(
            content = system_message
        ),
        HumanMessage(content=f'''当前查询：{state["ori_query"]}''')
    ]
    logger.info(f"Coordinator使用模型: {llm_type}")
    try:
        logger.info("Coordinator调用LLM")
        response_content = get_llm_by_type(type = llm_type).invoke(messages).content
        logger.info("Coordinator成功获取LLM响应")
    except Exception as e:
        logger.error(f"Coordinator调用LLM失败: {str(e)}", exc_info=True)
        return Command(goto="__end__")
        
    logger.debug(f"当前状态消息: {state['messages']}")
    logger.debug(f"Coordinator响应: {response_content}")

    goto = "__end__"
    if "handoff_to_planner" in response_content:
        logger.info("Coordinator决定转交给Planner")
        goto = "planner"
    else:
        logger.info("Coordinator决定结束流程")

    return Command(
        goto=goto,
    )
def main_planner(state: State):
    logger.info("Planner开始工作")
    parser = JsonOutputParser()
    def inner_router():
        logger.info("开始知识库路由")
        for i in range(3):
            try:
                logger.info(f"路由尝试 #{i+1}")
                system_message = get_prompt_template("knowledge_store_router")
                messages = [
                    SystemMessage(
                        content = system_message
                    ),
                    HumanMessage(content=f'''当前查询：{state["ori_query"]}''')
                ]
                logger.info("路由器调用LLM")
                response = get_llm_by_type(type = llm_type).invoke(messages).content
                logger.debug(f"路由器LLM响应: {response}")
                
                # 5. 解析JSON输出
                parser = JsonOutputParser()
                result = parser.parse(response)
                logger.info(f"解析的路由结果: {result}")
                
                # 6. 验证结果是否在可用知识库中
                # valid_sources = {t["name"] for t in VECTORSTORES}
                if result["subject"] not in SUBJECTS:
                    logger.warning(f"第{i+1}次尝试：选择的知识库不存在: {result['subject']}")
                    print(f"第{i+1}次尝试：选择的知识库不存在: {result['subject']}")
                    continue
                logger.info(f"成功选择知识库: {result['subject']}")
                print("router",result["subject"])
                return result["subject"]
            except Exception as e:
                logger.error(f"路由器错误: {str(e)}", exc_info=True)
                print(f"模型返回非法JSON: {response} {e}")
            except KeyError as e:
                logger.error(f"路由器缺少必要字段: {str(e)}", exc_info=True)
                print(f"模型返回缺少必要字段: {response} {e}")
        logger.warning("所有路由尝试失败，返回默认值")
        return "无可用知识库"
    # 1. 定义 JSON 输出解析器
    subject = inner_router()
    logger.info(f"使用主题: {subject} 创建Planner提示")
    system_message = get_prompt_template("planner",SUBJECT=subject)
    messages = [
        SystemMessage(
            content = system_message
        ),
        HumanMessage(content=f'''当前查询：{state["ori_query"]}''')
    ]
    llm = get_llm_by_type(type = llm_type)
    if state.get("search_before_planning"):
        logger.info("执行规划前搜索")
        # search_system_prompt = '''
        # 查阅"{{query}}"会涉及的知识点，比如查百度百科、对应书本的目录等，需要得到详细的具体的知识点，比如'组成细胞的分子'。
        # '''
        # # searched_content = tavily_tool.invoke({"query": search_system_prompt.replace("{{query}}",state["ori_query"])})
        try:
            logger.info("调用知识搜索器")
            searched_content = str(knowledge_searcher.invoke(state)["messages"][-1].content)
            logger.info("搜索成功完成")
            messages = deepcopy(messages)
            messages[
                -1
            ].content += f"\n\n# 相关搜索结果\n\n{searched_content}"
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}", exc_info=True)
    
    logger.info("Planner开始流式调用LLM")
    try:
        stream = llm.stream(messages)
        full_response = ""
        for chunk in stream:
            full_response += chunk.content
        logger.info("Planner成功获取完整响应")
    except Exception as e:
        logger.error(f"Planner LLM调用失败: {str(e)}", exc_info=True)
        return Command(goto="__end__")
        
    logger.debug(f"当前状态消息: {state['messages']}")
    logger.debug(f"Planner响应: {full_response}")

    if full_response.startswith("```json"):
        logger.debug("移除JSON代码块标记")
        full_response = full_response.removeprefix("```json")

    if full_response.endswith("```"):
        full_response = full_response.removesuffix("```")

    goto = "supervisor"
    try:
        logger.info("尝试修复和解析JSON响应")
        repaired_response = json_repair.loads(full_response)
        full_response = json.dumps(repaired_response, ensure_ascii=False, indent=2)
        logger.info("JSON修复和解析成功")
    except json.JSONDecodeError:
        logger.warning("Planner响应不是有效的JSON，流程将结束")
        goto = "__end__"
    print(full_response)
    logger.info(f"Planner完成，下一步转到: {goto}")
    return Command(
        update={
            "messages": [HumanMessage(content=full_response,name="planner")],
            "full_plan": full_response,
        },
        goto=goto,
    )
    
RESPONSE_FORMAT = "{}的回复:\n\n<response>\n{}\n</response>\n\n*请执行下一步.*"
def main_supervisor(state: State) -> Command[Literal[*TEAM_MEMBERS, "__end__"]]:
    """Supervisor node that decides which agent should act next."""
    logger.info("Supervisor开始评估下一步行动")
    parser = JsonOutputParser()
    logger.info("应用Supervisor提示模板")
    messages = apply_prompt_template("supervisor",state)
    # preprocess messages to make supervisor execute better.
    messages = deepcopy(messages)
    reports = []
    for message in messages:
        if isinstance(message, BaseMessage) and message.name in TEAM_MEMBERS:
            if message.name == "reporter":
                reports.append(message.content)
            message.content = RESPONSE_FORMAT.format(message.name, message.content)
    
    logger.info(f"Supervisor处理了 {len(messages)} 条消息")
    goto = "__end__"  # 默认值
    next_step_content = ""
    
    for i in range(3):
        try:
            logger.info(f"Supervisor尝试 #{i+1}")
            if len(messages)>9:
                logger.info("消息过长，使用特殊处理方式")
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
                logger.info("使用hkust-deepseek-r1 API")
                response = call_Hkust_api(prompt = "",messages = openai_format)
                logger.info("成功获取API响应")
                parsed_response = get_json_result(response)
            else:
                logger.info(f"使用标准LLM: {llm_type}")
                response = (
                    get_llm_by_type(llm_type)
                    # .with_structured_output(schema=Router, method="json_mode")
                    .invoke(messages)
                )
                logger.info("成功获取LLM响应")
                parsed_response = get_json_result(response.content)
            
            logger.info(f"解析的响应: {parsed_response}")
            goto = parsed_response["next"]
            next_step_content = parsed_response["next_step_content"]
            logger.info(f"Supervisor决定下一步: {goto}")
            break
        except Exception as e:
            logger.error(f"Supervisor错误(尝试 #{i+1}): {str(e)}", exc_info=True)
            print(f"supervisor出错了：{e}")
    
    logger.debug(f"当前状态消息: {state['messages']}")
    logger.debug(f"Supervisor响应: {response}")

    if goto == "FINISH":
        logger.info("Supervisor决定完成工作流")
        goto = "__end__"
        
        try:
            logger.info(f"将结果写入文件: {state['quiz_url']}")
            with open(state['quiz_url'], "w", encoding="utf-8") as f:
                f.write(reports[-1])
            logger.info("成功写入结果文件")
        except Exception as e:
            logger.error(f"写入结果文件失败: {str(e)}", exc_info=True)
            
        logger.info("工作流完成")
    else:
        logger.info(f"Supervisor委派给: {goto}")
    
    if goto == "rag_er":
        logger.info("配置RAG(不启用浏览器)")
        updated_rag = {
            **state['rag'],
            "enable_browser": False
        }
    elif goto == "rag_and_browser":
        logger.info("配置RAG(启用浏览器)")
        updated_rag = {
            **state['rag'],
            "enable_browser": True
        }
    else:
        logger.info("使用默认RAG配置")
        updated_rag = {
            **state['rag']
        }
    return Command(goto=goto, update={"next": goto, "next_work": next_step_content, "rag":updated_rag})


def main_browser_generator(state: State) -> Command[Literal["supervisor"]]:
    """Node for the browser agent that performs web browsing tasks."""
    logger.info("浏览器代理开始任务")
    for i in range(3):
        try: 
            logger.info(f"浏览器代理尝试 #{i+1}")
            result = browser_generator.invoke(state)
            logger.info("浏览器代理成功完成任务")
            response_content = result["messages"][-1].content
            logger.debug(f"浏览器代理响应: {response_content[:100]}...")
            break
        except Exception as e:
            logger.error(f"浏览器代理失败(尝试 #{i+1}): {str(e)}", exc_info=True)
            response_content = ""
            return Command(goto="__end__")
    # 尝试修复可能的JSON输出
    # response_content = repair_json_output(response_content)
    # logger.debug(f"Browser agent response: {response_content}")
    # print(f"Browser agent response: {response_content}")
    logger.info("浏览器代理任务完成，返回到Supervisor")
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


def main_rag(state: State) -> Command[Literal["supervisor"]]:
    """Node for the RAG that performs RAG tasks."""
    logger.info("RAG代理开始任务")
    try:
        logger.info("调用RAG图")
        rag_state = state['rag_graph'].invoke(state)
        logger.info("RAG图调用成功")
        new_qa = str(rag_state['existed_qa'][-1])
        logger.debug(f"RAG生成的QA: {new_qa[:100]}...")
        new_q = f"题目内容已省略，概括内容为{state['next_work']}"
        logger.info("RAG代理成功完成任务")
    except Exception as e:
        logger.error(f"RAG处理失败: {str(e)}", exc_info=True)
        return Command(goto="__end__")
    # if "参考答案" in new_qa:
    #     new_q = new_qa.split("参考答案")[0].strip()
    # elif "答案" in new_qa:
    #     new_q = new_qa.split("答案")[0].strip()
    # else:
    #     new_q = new_qa
    # 尝试修复可能的JSON输出
    # response_content = repair_json_output(response_content)
    logger.debug(f"RAG代理响应: {new_qa[:100]}...")
    return Command(
        update={
            "existed_qa": [new_qa],
            "messages": [
                HumanMessage(
                    content=new_q,
                    name="rag_er",
                )
            ]
        },
        goto="supervisor",
    )


def main_rag_browser(state: State) -> Command[Literal["supervisor"]]:
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
    new_q = f"题目内容已省略，概括内容为{state['next_work']}"
    logger.info("RAG agent completed task")
    # 尝试修复可能的JSON输出
    # response_content = repair_json_output(response_content)
    logger.debug(f"RAG agent response: {new_qa}")
    return Command(
        update={
            "existed_qa": [new_qa],
            "messages": [
                HumanMessage(
                    content=new_q,
                    name="rag_er",
                )
            ]
        },
        goto="supervisor",
    )


def main_reporter(state: State) -> Command[Literal["supervisor"]]:
    """Reporter node that write a final report."""
    logger.info("Reporter write final report")
    tmp_state = {
        "messages":[
            state['messages'][0],
            state['messages'][1],
            HumanMessage(content = '\n\n\n\n'.join(state['existed_qa']))
        ]
    }
    messages = apply_prompt_template("reporter", tmp_state)
    response = get_llm_by_type(llm_type).invoke(messages)
    logger.debug(f"Current state messages: {state['messages']}")
    response_content = response.content
    # 尝试修复可能的JSON输出
    # response_content = repair_json_output(response_content)
    logger.debug(f"reporter response: {response_content}")

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