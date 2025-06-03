from openai import OpenAI
from ...config.llms import (
    openai_model, openai_api_key, openai_api_base, 
    llm_type, embedding_model,
    api_timeout, api_max_retries, api_retry_interval, api_batch_size
)

from langchain_openai import ChatOpenAI
import os
import time
import logging
import httpx
from langchain.schema.runnable import RunnableLambda
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

logger = logging.getLogger(__name__)

def getClient()->OpenAI:
    client = OpenAI(
        base_url=openai_api_base, 
        api_key=openai_api_key,
        timeout=api_timeout,
        max_retries=api_max_retries
    )
    return client

@retry(
    wait=wait_exponential(multiplier=api_retry_interval, min=1, max=60),
    stop=stop_after_attempt(api_max_retries),
    retry=retry_if_exception_type((httpx.ReadTimeout, httpx.ConnectTimeout, httpx.TimeoutException)),
    reraise=True
)
def call_api(prompt, model=openai_model):
    logger.info(f"调用API，模型: {model}")
    try:
        client = getClient()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024
        )
        logger.info("API调用成功")
        return response.choices[0].message.content
    except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.TimeoutException) as e:
        logger.warning(f"API超时，准备重试: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"API调用失败: {str(e)}")
        raise

def get_llm_response(prompt, model, model_type=llm_type, options={"format": "json", "num_ctx": 8192}):
    '''
    model="gpt-4o-mini"
    model_type = "openai"
    '''
    response = call_api(prompt, model)
    return response

def get_llm_by_type(type, api_config=None):
    '''
    # paramter:
    type: "openai"
    api_config: 包含API配置的字典，包括model, api_key, api_base, llm_type等
    # usage:
    from langchain_core.messages import HumanMessage, SystemMessage
    llm = get_llm_by_type("openai")
    # 构建消息
    messages = [
        SystemMessage(content="你是一个物理学教授"),
        HumanMessage(content="用简单的比喻解释量子隧穿效应")
    ]
    # 调用模型
    response = llm.invoke(messages)
    print("回答：", response.content)
    '''
    config = api_config or {}
    llm = ChatOpenAI(
        model=config.get("model", openai_model),
        api_key=config.get("api_key", openai_api_key),
        base_url=config.get("api_base", openai_api_base),
        temperature=0.7,
        max_retries=api_max_retries,
        timeout=api_timeout,
        request_timeout=api_timeout,  # 明确设置请求超时
        streaming=False  # 关闭流式处理，减少超时风险
    )
    return llm
    

