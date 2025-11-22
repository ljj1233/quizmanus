import json
import sys
import types


# Provide lightweight stubs when optional dependencies are unavailable
if "langchain_core" not in sys.modules:
    langchain_core = types.ModuleType("langchain_core")

    messages = types.ModuleType("langchain_core.messages")

    class BaseMessage:
        def __init__(self, content=None, name=None):
            self.content = content
            self.name = name

    class HumanMessage(BaseMessage):
        pass

    class SystemMessage(BaseMessage):
        pass

    class AIMessage(BaseMessage):
        pass

    def messages_to_dict(msgs):
        result = []
        for msg in msgs:
            entry = {"type": "human", "data": {"content": getattr(msg, "content", None), "name": getattr(msg, "name", None)}}
            result.append(entry)
        return result

    messages.BaseMessage = BaseMessage
    messages.HumanMessage = HumanMessage
    messages.SystemMessage = SystemMessage
    messages.AIMessage = AIMessage
    messages.messages_to_dict = messages_to_dict

    prompts = types.ModuleType("langchain_core.prompts")

    class ChatPromptTemplate:
        def __init__(self, *_, **__):
            pass

    prompts.ChatPromptTemplate = ChatPromptTemplate

    runnables = types.ModuleType("langchain_core.runnables")

    class RunnableLambda:
        def __init__(self, func):
            self.func = func

        def __call__(self, *args, **kwargs):
            return self.func(*args, **kwargs)

    runnables.RunnableLambda = RunnableLambda

    output_parsers = types.ModuleType("langchain_core.output_parsers")

    class JsonOutputParser:
        def parse(self, text):
            return json.loads(text)

    output_parsers.JsonOutputParser = JsonOutputParser

    tools_mod = types.ModuleType("langchain_core.tools")

    def tool(func=None, *args, **kwargs):
        if func is None:
            return lambda f: f
        return func

    tools_mod.tool = tool

    sys.modules["langchain_core"] = langchain_core
    sys.modules["langchain_core.messages"] = messages
    sys.modules["langchain_core.prompts"] = prompts
    sys.modules["langchain_core.runnables"] = runnables
    sys.modules["langchain_core.output_parsers"] = output_parsers
    sys.modules["langchain_core.tools"] = tools_mod

if "langgraph" not in sys.modules:
    langgraph = types.ModuleType("langgraph")

    graph = types.ModuleType("langgraph.graph")
    graph.StateGraph = type("StateGraph", (), {})
    graph.END = "__end__"
    graph.MessagesState = dict

    types_mod = types.ModuleType("langgraph.types")

    class Command:
        def __init__(self, goto=None, update=None):
            self.goto = goto
            self.update = update or {}

        @classmethod
        def __class_getitem__(cls, _key):
            return cls

    types_mod.Command = Command

    prebuilt = types.ModuleType("langgraph.prebuilt")

    def create_react_agent(*args, **kwargs):
        return {"args": args, "kwargs": kwargs}

    prebuilt.create_react_agent = create_react_agent

    chat_agent_executor = types.ModuleType("langgraph.prebuilt.chat_agent_executor")
    chat_agent_executor.AgentState = dict

    sys.modules["langgraph"] = langgraph
    sys.modules["langgraph.graph"] = graph
    sys.modules["langgraph.types"] = types_mod
    sys.modules["langgraph.prebuilt"] = prebuilt
    sys.modules["langgraph.prebuilt.chat_agent_executor"] = chat_agent_executor

if "milvus_model" not in sys.modules:
    milvus_model = types.ModuleType("milvus_model")
    hybrid = types.ModuleType("milvus_model.hybrid")

    class BGEM3EmbeddingFunction:
        def __init__(self, *_, **__):
            pass

        def __call__(self, texts):
            return {"dense": [[0.0 for _ in texts]], "sparse": types.SimpleNamespace(_getrow=lambda _i: None)}

    hybrid.BGEM3EmbeddingFunction = BGEM3EmbeddingFunction

    sys.modules["milvus_model"] = milvus_model
    sys.modules["milvus_model.hybrid"] = hybrid

if "pymilvus" not in sys.modules:
    pymilvus = types.ModuleType("pymilvus")
    model_mod = types.ModuleType("pymilvus.model")
    reranker_mod = types.ModuleType("pymilvus.model.reranker")

    class BGERerankFunction:
        def __init__(self, *_, **__):
            pass

        def __call__(self, *args, **kwargs):
            return []

    reranker_mod.BGERerankFunction = BGERerankFunction

    class DummyCollection:
        def __init__(self, *_, **__):
            self.data = []

        def drop(self):
            self.data.clear()

        def create_index(self, *_, **__):
            return None

        def load(self):
            return None

        def insert(self, _data):
            self.data.append(_data)

    class DummyConnections:
        def connect(self, *_, **__):
            return None

    class DummyUtility:
        def has_collection(self, *_args, **_kwargs):
            return False

    class DummyField:
        def __init__(self, *_, **__):
            pass

    class DummyCollectionSchema:
        def __init__(self, *_, **__):
            pass

    class DummyDataType:
        VARCHAR = "VARCHAR"
        SPARSE_FLOAT_VECTOR = "SPARSE_FLOAT_VECTOR"
        FLOAT_VECTOR = "FLOAT_VECTOR"

    pymilvus.connections = DummyConnections()
    pymilvus.utility = DummyUtility()
    pymilvus.FieldSchema = DummyField
    pymilvus.CollectionSchema = DummyCollectionSchema
    pymilvus.DataType = DummyDataType
    pymilvus.Collection = DummyCollection
    pymilvus.AnnSearchRequest = type("AnnSearchRequest", (), {})
    pymilvus.HybridSearchResult = type("HybridSearchResult", (), {})
    pymilvus.WeightedRanker = lambda *_, **__: None

    sys.modules["pymilvus"] = pymilvus
    sys.modules["pymilvus.model"] = model_mod
    sys.modules["pymilvus.model.reranker"] = reranker_mod

if "modelscope" not in sys.modules:
    modelscope = types.ModuleType("modelscope")
    modelscope.AutoModelForCausalLM = type("AutoModelForCausalLM", (), {})
    modelscope.AutoTokenizer = type("AutoTokenizer", (), {})
    sys.modules["modelscope"] = modelscope

if "peft" not in sys.modules:
    peft = types.ModuleType("peft")
    peft.PeftModel = type("PeftModel", (), {})
    sys.modules["peft"] = peft

if "jinja2" not in sys.modules:
    jinja2 = types.ModuleType("jinja2")

    class Template:
        def __init__(self, template_str=""):
            self.template_str = template_str

        def render(self, **kwargs):
            return self.template_str.format(**kwargs)

    class Environment:
        def __init__(self, *_, **__):
            pass

        def get_template(self, _name):
            return Template("")

    def FileSystemLoader(*_args, **_kwargs):
        return None

    jinja2.Template = Template
    jinja2.Environment = Environment
    jinja2.FileSystemLoader = FileSystemLoader
    sys.modules["jinja2"] = jinja2

if "httpx" not in sys.modules:
    httpx = types.ModuleType("httpx")

    class Client:
        def __init__(self, *_, **__):
            pass

    httpx.Client = Client
    sys.modules["httpx"] = httpx

if "ALL_KEYS" not in sys.modules:
    ALL_KEYS = types.SimpleNamespace(
        hkust_openai_base_url="",
        hkust_openai_key="",
        Authorization_hkust_key="",
        common_openai_base_url="",
        common_openai_key="",
    )
    sys.modules["ALL_KEYS"] = ALL_KEYS

if "openai" not in sys.modules:
    openai = types.ModuleType("openai")
    openai.OpenAI = lambda *args, **kwargs: None
    sys.modules["openai"] = openai

if "requests" not in sys.modules:
    requests = types.ModuleType("requests")

    class Response:
        def json(self):
            return {}

    def post(*_args, **_kwargs):
        return Response()

    requests.post = post
    sys.modules["requests"] = requests

if "torch" not in sys.modules:
    torch = types.ModuleType("torch")

    def device(name):
        return name

    def cuda():
        return False

    torch.device = device
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = torch

if "tqdm" not in sys.modules:
    tqdm = types.ModuleType("tqdm")

    def tqdm_func(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

    tqdm.tqdm = tqdm_func
    sys.modules["tqdm"] = tqdm

if "json_repair" not in sys.modules:
    json_repair = types.ModuleType("json_repair")

    def loads(text):
        import json as _json

        try:
            return _json.loads(text)
        except Exception:
            return {}

    json_repair.loads = loads
    sys.modules["json_repair"] = json_repair

if "markdownify" not in sys.modules:
    markdownify = types.ModuleType("markdownify")

    def markdownify_func(text, *args, **kwargs):
        return text

    markdownify.markdownify = markdownify_func
    sys.modules["markdownify"] = markdownify

if "readabilipy" not in sys.modules:
    readabilipy = types.ModuleType("readabilipy")

    def simple_json_from_html_string(*_args, **_kwargs):
        return {"content": ""}

    readabilipy.simple_json_from_html_string = simple_json_from_html_string
    sys.modules["readabilipy"] = readabilipy

if "langchain_community" not in sys.modules:
    lc_community = types.ModuleType("langchain_community")
    tools_pkg = types.ModuleType("langchain_community.tools")
    tavily_search = types.ModuleType("langchain_community.tools.tavily_search")

    class TavilySearchResults:
        def __init__(self, *_, **__):
            pass

        def _run(self, *args, **kwargs):
            return []

    tavily_search.TavilySearchResults = TavilySearchResults

    sys.modules["langchain_community"] = lc_community
    sys.modules["langchain_community.tools"] = tools_pkg
    sys.modules["langchain_community.tools.tavily_search"] = tavily_search

if "dotenv" not in sys.modules:
    dotenv = types.ModuleType("dotenv")

    def load_dotenv(*_args, **_kwargs):
        return True

    dotenv.load_dotenv = load_dotenv
    sys.modules["dotenv"] = dotenv

if "ollama" not in sys.modules:
    sys.modules["ollama"] = types.ModuleType("ollama")

if "langchain_openai" not in sys.modules:
    langchain_openai = types.ModuleType("langchain_openai")

    class ChatOpenAI:
        def __init__(self, *_, **__):
            pass

        def invoke(self, messages):
            return types.SimpleNamespace(content="")

        def stream(self, messages):
            return []

    langchain_openai.ChatOpenAI = ChatOpenAI
    sys.modules["langchain_openai"] = langchain_openai

if "langchain_ollama" not in sys.modules:
    langchain_ollama = types.ModuleType("langchain_ollama")

    class ChatOllama:
        def __init__(self, *_, **__):
            pass

        def invoke(self, messages):
            return types.SimpleNamespace(content="")

        def stream(self, messages):
            return []

    langchain_ollama.ChatOllama = ChatOllama
    sys.modules["langchain_ollama"] = langchain_ollama

if "vllm" not in sys.modules:
    vllm = types.ModuleType("vllm")
    class SamplingParams:
        def __init__(self, *_, **__):
            pass

    vllm.SamplingParams = SamplingParams
    sys.modules["vllm"] = vllm

if "langchain" not in sys.modules:
    langchain = types.ModuleType("langchain")
    schema_mod = types.ModuleType("langchain.schema")
    runnable_mod = types.ModuleType("langchain.schema.runnable")

    class RunnableLambda:
        def __init__(self, func):
            self.func = func

        def __call__(self, *args, **kwargs):
            return self.func(*args, **kwargs)

    runnable_mod.RunnableLambda = RunnableLambda
    schema_mod.runnable = runnable_mod

    sys.modules["langchain"] = langchain
    sys.modules["langchain.schema"] = schema_mod
    sys.modules["langchain.schema.runnable"] = runnable_mod

# Ensure repository modules can be imported without installation
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
