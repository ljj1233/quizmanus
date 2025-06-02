from openai import OpenAI
from ...config.llms import (
    openai_model, openai_api_key, openai_api_base, 
    llm_type, embedding_model,
    api_timeout, api_max_retries, api_batch_size
)

from langchain_openai import ChatOpenAI
import os
from langchain.schema.runnable import RunnableLambda

def getClient()->OpenAI:
    client = OpenAI(
        base_url=openai_api_base, 
        api_key=openai_api_key,
        timeout=api_timeout,
        max_retries=api_max_retries
    )
    return client

def call_api(prompt, model=openai_model):
    client = getClient()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024
    )
    return response.choices[0].message.content

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
        timeout=api_timeout
    )
    return llm
    

