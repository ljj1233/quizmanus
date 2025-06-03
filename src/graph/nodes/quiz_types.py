# from typing_extensions import TypedDict
from langgraph.graph import MessagesState, StateGraph
from typing import List, Dict, TypedDict, Literal, Optional, Annotated, Union
from operator import add
from langgraph.graph import StateGraph
from openai import OpenAI
from src.config.llms import openai_api_key, openai_api_base
from pymilvus.model.embedding import BGEM3EmbeddingFunction
from pymilvus.model.reranker import BGERerankFunction

# 初始化OpenAI客户端
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base
)

# 初始化BGE模型 - 使用本地模型进行文本嵌入
embeddings = BGEM3EmbeddingFunction(
    model_name = "/hpc2hdd/home/fye374/models/BAAI/bge-m3",
    use_fp16=False, 
    device="cuda"
)

# 初始化BGE重排序模型 - 使用本地模型进行文档重排序
reranker = BGERerankFunction(
    model_name="/hpc2hdd/home/fye374/models/BAAI/bge-reranker-v2-m3",  
    device="cuda",
    use_fp16=False
)

class RAGState(MessagesState):
    
    hyde_query: str
    selected_subject: str
    retrieved_docs: List[str]
    reranked_docs: List[str]
    embedding_model: BGEM3EmbeddingFunction
    reranker_model: BGERerankFunction
    enable_browser: bool
    outer_knowledge: str


class State(MessagesState):
    """State for the agent system, extends MessagesState with next field."""

    # Runtime Variables
    ori_query: str
    rag_graph: StateGraph
    existed_qa: Annotated[List,add]
    next: str
    full_plan: str
    deep_thinking_mode: bool
    search_before_planning: bool
    next_work: str
    rag: RAGState
    quiz_url: str


