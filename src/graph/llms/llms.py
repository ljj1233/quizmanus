from openai import OpenAI
import ollama
import sys
from ...config.llms import openai_model, openai_api_key, openai_api_base, ollama_model

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import os
from langchain.schema.runnable import RunnableLambda

import torch
def getClient()->OpenAI:
    client = OpenAI(
        base_url=openai_api_base, 
        api_key=openai_api_key,# gpt35  
        http_client=httpx.Client(
            base_url=openai_api_base,
            follow_redirects=True,
        ),
    )
    return client

def call_api(prompt, model):
    try:
        response = getClient().chat.completions.create(
            model=model,
            # temperature=float(temperature),
            # max_tokens=int(max_tokens),
            # top_p=float(top_p),
            messages=[
                # {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

def get_llm_response(prompt, model, model_type = "ollama", options={"format": "json","num_ctx": 8192,"device": "cuda:0"}):
    '''
    model="qwen2.5:14b"
    model_type = "ollama"
    ======
    model="gpt-4o-mini"
    model_type = "openai"
    '''
    if model_type == "ollama":
        response = ollama.generate(
            model="qwen2.5:14b",
            prompt=prompt,
            options=options,  # 强制JSON输出
        )["response"]
    elif model_type == "openai":
        response = call_api(prompt,model)
    return response
    
    



def get_llm_by_type(type, model=None, tokenizer=None, api_config=None):
    '''
    # paramter:
    type: "ollama","openai","qwen2.5-7b","qwen2.5-3b"
    api_config: 包含API配置的字典，包括model, api_key, api_base, llm_type等
    # usage:
    from langchain_core.messages import HumanMessage, SystemMessage
    llm = get_llm_by_type("ollama")
    # 构建消息
    messages = [
        SystemMessage(content="你是一个物理学教授"),
        HumanMessage(content="用简单的比喻解释量子隧穿效应")
    ]
    # 调用模型
    response = llm.invoke(messages)
    print("回答：", response.content)
    '''
    
    if type == "openai" or (api_config and api_config.get("llm_type") == "openai"):
        config = api_config or {}
        llm = ChatOpenAI(
            model=config.get("model", openai_model),
            api_key=config.get("api_key", openai_api_key),
            base_url=config.get("api_base", openai_api_base),
            temperature=0.7,
            max_retries=3
        )
    elif type == "ollama" or (api_config and api_config.get("llm_type") == "ollama"):
        config = api_config or {}
        llm = ChatOllama(
            model=config.get("model", ollama_model),
            num_ctx=25600,
            temperature=0.7,
            stream=False
        )
    elif "qwen" in type.lower():
        if api_config and api_config.get("llm_type") == "openai":
            # 如果配置了API，使用API调用
            config = api_config
            llm = ChatOpenAI(
                model=config.get("model", openai_model),
                api_key=config.get("api_key", openai_api_key),
                base_url=config.get("api_base", openai_api_base),
                temperature=0.7,
                max_retries=3
            )
        else:
            # 使用本地模型
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            def prepare_input(messages, tokenizer):
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                return prompt.replace("\n<|im_end|>\n",'')+"\n"
            
            def single_generate(x, model, tokenizer):
                model.to(device)
                inputs = tokenizer(
                    prepare_input(x, tokenizer), 
                    return_tensors="pt", 
                    padding=True,
                    truncation=True
                ).to(device)
                
                generated_ids = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                    repetition_penalty=1.1
                )
                
                completion_ids = generated_ids[0][len(inputs.input_ids[0]):].cpu()
                return tokenizer.decode(completion_ids, skip_special_tokens=True)
            
            llm = RunnableLambda(lambda x: single_generate(x, model, tokenizer))
    return llm
    

