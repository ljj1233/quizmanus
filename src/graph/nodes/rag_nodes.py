from typing import List, Dict, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from jinja2 import Template
# import sys
# import ollama
# from ..config.rag import VECTORSTORES
# from ..config.rag import reranker, embedding_model
from .quiz_types import State
from langgraph.types import Command
# from langchain_community.vectorstores import FAISS
# 定义状态机结构
from ..llms.llms import get_llm_by_type

from ..agents.agents import knowledge_based_browser

from ...config.rag import DB_URI, COLLECTION_NAME, SUBJECTS
from ...RAG.vector_store_utils import get_collection
from ...RAG.retrieval import hybrid_search
from ...RAG.reranker import rerank
from langchain_core.messages import HumanMessage, SystemMessage
from ...config.llms import llm_type,generator_model, api_max_retries
from ...config.nodes import QUESTION_TYPES
from ...utils import get_json_result
import logging
import time
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@retry(
    wait=wait_exponential(multiplier=2, min=1, max=60),
    stop=stop_after_attempt(api_max_retries),
    retry=retry_if_exception_type((httpx.ReadTimeout, httpx.ConnectTimeout, httpx.TimeoutException)),
    reraise=True
)
def call_llm_with_retry(llm, messages):
    """使用重试机制调用LLM"""
    try:
        logger.info("调用LLM开始")
        response = llm.invoke(messages)
        logger.info("调用LLM成功")
        return response.content
    except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.TimeoutException) as e:
        logger.warning(f"LLM调用超时，准备重试: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"LLM调用失败: {str(e)}", exc_info=True)
        raise

def rag_hyde(state: State):
    # 1. 定义 JSON 输出解析器
    parser = JsonOutputParser()
    messages = [
        SystemMessage(
            content='''
            [角色] 查询的假设性文档生成器
            [任务] 用自身知识对当前查询进行改写，改写成和查询相关的课本知识，文档要为陈述句，不要直接生成题目。
            再次强调，不要直接生成题目！！！
            严格返回以下 JSON 格式：
            {
                "hyde_query": "改写后的查询内容"
            }
            '''
        ),
        HumanMessage(content=f'''当前查询：{state["next_work"]}''')
    ]
    
    logger.info("开始执行HyDE查询改写")
    try:
        # 使用带重试机制的LLM调用
        llm = get_llm_by_type(type=llm_type)
        rewrite_res = call_llm_with_retry(llm, messages)
        logger.info("HyDE查询改写成功")
        
        # 5. 用 JsonOutputParser 解析结果
        try:
            parsed_output = parser.parse(rewrite_res)
            logger.info(f"解析HyDE结果: {parsed_output}")
            print("hyde", parsed_output)
            updated_rag = {
                **state['rag'],
                "hyde_query": parsed_output["hyde_query"]
            }
            return Command(
                update = {
                    "rag": updated_rag
                },
                goto = "router"
            )
        except Exception as e:
            logger.error(f"解析HyDE结果失败: {str(e)}", exc_info=True)
            print(f"解析失败: {rewrite_res} 错误: {e}")
    except Exception as e:
        logger.error(f"HyDE查询改写失败: {str(e)}", exc_info=True)
    
    # 如果所有尝试都失败，使用原始查询
    logger.warning("所有HyDE尝试失败，使用原始查询")
    updated_rag = {
        **state['rag'],
        "hyde_query": state["next_work"]  # 使用原始查询作为备选
    }
    return Command(
        update = {
            "rag": updated_rag
        },
        goto = "router"
    )


def rag_router(state: State):
    messages = [
        SystemMessage(
            content='''
            [任务] 学科与题型选择决策
    选择标准：
    - 选择与查询语义最相关的1个学科
    - 判断查询是想要生成说明类型的题目
    - 只能返回上述JSON格式，不要包含额外内容
    请严格按以下JSON格式返回结果：
    {
        "subject": "学科名称",
        "question_type": "题型"
    }
            '''
        ),
        HumanMessage(content=f'''可选学科：
    {SUBJECTS}
    可选题型：
    单选题、多选题、主观题

    当前查询："{state["next_work"]}"
    当前扩展查询："{state["rag"]["hyde_query"]}"
    回答：''')
    ]
    
    logger.info("开始执行路由选择")
    for i in range(3):  # 尝试3次
        try:
            logger.info(f"路由尝试 #{i+1}")
            llm = get_llm_by_type(type=llm_type)
            response = call_llm_with_retry(llm, messages)
            logger.info("路由选择成功")
            
            # 5. 解析JSON输出
            parser = JsonOutputParser()
            result = parser.parse(response)
            logger.info(f"解析路由结果: {result}")
            
            # 6. 验证结果是否在可用知识库中
            if result["subject"] not in SUBJECTS:
                logger.warning(f"选择的知识库不存在: {result['subject']}")
                print(f"第{i+1}次尝试：选择的知识库不存在: {result['subject']}")
                continue
                
            logger.info(f"选择学科: {result['subject']}, 题型: {result['question_type']}")
            print("router", result["subject"])
            print("type", result["question_type"])
            updated_rag = {
                **state['rag'],
                "subject": result["subject"],
                "type": result["question_type"],
            }
            return Command(
                update = {
                    "rag": updated_rag
                },
                goto = "retriever"
            )
            
        except Exception as e:
            logger.error(f"路由选择失败(尝试 #{i+1}): {str(e)}", exc_info=True)
            print(f"模型返回非法JSON或缺少字段: {e}")
            time.sleep(2)  # 失败后短暂等待
    
    # 如果所有尝试都失败，使用默认值
    logger.warning("所有路由尝试失败，使用默认值")
    default_subject = list(SUBJECTS)[0] if SUBJECTS else "生物"
    updated_rag = {
        **state['rag'],
        "subject": default_subject,
        "type": "单选题",
    }
    return Command(
        update = {
            "rag": updated_rag
        },
        goto = "retriever"
    )


# 3. 检索执行组件
def rag_retrieve(state:State):
    logger.info("开始执行检索")
    try:
        # 使用BGE模型进行嵌入
        query = state["rag"]['hyde_query']
        logger.info(f"检索查询: {query}")
        
        # BGE-M3模型会同时返回dense和sparse嵌入
        query_embeddings = state["rag"]['embedding_model']([query])
        logger.info("嵌入生成成功")
        
        col = get_collection(DB_URI, COLLECTION_NAME)
        logger.info(f"使用学科过滤: {state['rag']['subject']}")
        
        hybrid_results = hybrid_search(
            col,
            query_embeddings["dense"][0],
            query_embeddings["sparse"]._getrow(0),
            subject_value=state["rag"]['subject'],  # 指定 subject 值
            sparse_weight=0.7,
            dense_weight=1.0,
            limit = 10
        )
        logger.info(f"检索到 {len(hybrid_results)} 条结果")
        
        updated_rag = {
            **state['rag'],
            "retrieved_docs": hybrid_results
        }
        return Command(
            update = {
                "rag": updated_rag
            },
            goto = "reranker"
        )
    except Exception as e:
        logger.error(f"检索过程失败: {str(e)}", exc_info=True)
        # 如果检索失败，返回空结果并继续流程
        updated_rag = {
            **state['rag'],
            "retrieved_docs": []
        }
        return Command(
            update = {
                "rag": updated_rag
            },
            goto = "reranker"
        )


def rag_reranker(state: State):
    logger.info("开始执行重排序")
    try:
        reranked_docs = rerank(
            query_text = state["rag"]['hyde_query'], 
            search_results = state["rag"]['retrieved_docs'], 
            reranker = state["rag"]['reranker_model'],
            topk = 1)
        logger.info("重排序成功完成")
    except Exception as e:
        logger.error(f"重排序失败: {str(e)}", exc_info=True)
        # 如果重排序失败，直接使用原始检索结果
        reranked_docs = state["rag"]['retrieved_docs'][:1] if state["rag"]['retrieved_docs'] else []
        logger.warning("使用原始检索结果作为备选")
    
    # print("reranked_docs",reranked_docs)
    updated_rag = {
        **state['rag'], 
        "reranked_docs": reranked_docs,
        'outer_knowledge':""
    }
    
    if state["rag"]['enable_browser']:
        logger.info("启用浏览器搜索补充知识")
        for i in range(3):  # 尝试3次
            try: 
                logger.info(f"浏览器搜索尝试 #{i+1}")
                message_state = {
                    "messages":[
                        HumanMessage(content=f'''课本知识：{''.join(reranked_docs)}''')
                    ]
                }
                result = knowledge_based_browser.invoke(message_state)
                logger.info("浏览器搜索成功完成")
                response_content = result["messages"][-1].content
                try:
                    outer_knowledge = get_json_result(response_content)['课外知识']
                    logger.info("成功解析课外知识")
                    updated_rag = {
                        **updated_rag, 
                        "outer_knowledge": outer_knowledge
                    }
                    break
                except Exception as e:
                    logger.error(f"解析课外知识失败: {str(e)}", exc_info=True)
                    if i == 2:  # 最后一次尝试
                        updated_rag = {
                            **updated_rag, 
                            "outer_knowledge": response_content  # 使用原始响应作为备选
                        }
            except Exception as e:
                logger.error(f"浏览器搜索失败(尝试 #{i+1}): {str(e)}", exc_info=True)
                time.sleep(2)  # 失败后短暂等待
                if i == 2:  # 最后一次尝试失败
                    logger.warning("所有浏览器搜索尝试失败")
    
    return Command(
        update = {
            "rag": updated_rag
        },
        goto = "generator"
    )


# 6. 生成组件
def rag_generator(state: State):
    logger.info("开始生成题目")
    try:
        # 准备提示模板
        template_str = '''
        [角色] 你是一个专业的教育工作者，擅长命题
        [任务] 根据提供的课本知识和课外知识，生成一道高质量的{{type}}。
        [要求]
        - 题目应该基于课本知识，但可以融合课外知识使题目更加丰富
        - 题目难度适中，既要考察基础知识，也要有一定的思考深度
        - 题目表述清晰，无歧义
        - 提供详细的参考答案和解析
        - 题目内容为：{{next_work}}

        [课本知识]
        {{reranked_docs}}

        [课外知识]
        {{outer_knowledge}}

        [输出格式]
        请按照以下格式输出：

        题目：（题干内容）

        {% if type == "单选题" or type == "多选题" %}
        A. 选项A
        B. 选项B
        C. 选项C
        D. 选项D

        {% endif %}
        参考答案：

        解析：
        '''

        # 使用Jinja2渲染模板
        template = Template(template_str)
        prompt = template.render(
            type=state["rag"]["type"],
            next_work=state["next_work"],
            reranked_docs=''.join(state["rag"]["reranked_docs"]),
            outer_knowledge=state["rag"]["outer_knowledge"]
        )
        
        # 构建消息
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content="请根据上述要求生成题目。")
        ]
        
        # 使用带重试机制的LLM调用
        logger.info("调用LLM生成题目")
        for i in range(3):  # 尝试3次
            try:
                llm = get_llm_by_type(type=generator_model)
                response = call_llm_with_retry(llm, messages)
                logger.info("题目生成成功")
                
                # 验证生成的内容
                if len(response) < 50:
                    logger.warning(f"生成的题目内容过短: {response}")
                    if i < 2:  # 不是最后一次尝试
                        continue
                
                return Command(
                    update = {
                        "existed_qa": state.get("existed_qa", []) + [response]
                    },
                    goto = "__end__"
                )
            except Exception as e:
                logger.error(f"题目生成失败(尝试 #{i+1}): {str(e)}", exc_info=True)
                time.sleep(3)  # 失败后等待时间更长
        
        # 如果所有尝试都失败，返回一个基本的错误信息
        logger.critical("所有题目生成尝试均失败")
        error_message = f"抱歉，由于技术原因无法生成关于"{state['next_work']}"的题目。请稍后再试。"
        
        return Command(
            update = {
                "existed_qa": state.get("existed_qa", []) + [error_message]
            },
            goto = "__end__"
        )
    except Exception as e:
        logger.critical(f"题目生成过程中发生严重错误: {str(e)}", exc_info=True)
        error_message = "生成题目时发生错误，请检查日志获取详细信息。"
        return Command(
            update = {
                "existed_qa": state.get("existed_qa", []) + [error_message]
            },
            goto = "__end__"
        )

