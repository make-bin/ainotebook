from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, AgentExecutor
from langchain.agents import create_react_agent
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------- 初始化 Ollama 模型 ---------------------
llm = ChatOllama(
  model= "deepseek-r1:14b",
  base_url="http://10.230.66.59:11434",
  temperature=0.7
)

# --------------------- 步骤 1: RAG 检索增强 ---------------------
# 1.1 加载文档（示例使用网页内容）
loader = WebBaseLoader(["https://python.langchain.com/docs/get_started/introduction"])
docs = loader.load()

# 1.2 分割文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 1.3 创建向量存储（使用 Ollama 的嵌入模型）
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OllamaEmbeddings(base_url="http://10.230.66.59:11434", model="deepseek-r1:14b")
)

# 1.4 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --------------------- 步骤 2: 定义自定义工具 ---------------------
# 工具 1: 检索增强问答
def rag_qa(query: str) -> str:
    """用于回答关于 LangChain 框架的问题"""
    print("calling rag_qa")
    docs = retriever.invoke(query)
    content = "\n\n".join([d.page_content for d in docs])
    return f"Observation: {content}" 

# 工具 2: 计算器
def calculator(expression: str) -> str:
    """用于执行数学计算，输入必须是数学表达式如 '2 + 3 * 4'"""
    try:
        return str(eval(expression))
    except:
        return "无效的数学表达式"

# 工具 3: 模拟搜索引擎
def search_engine(query: str) -> str:
    """用于搜索最新信息，输入为搜索关键词"""
    # 此处可替换为真实搜索引擎 API（如 Serper、Google 等）
    return f"模拟搜索结果: {query}"

# 工具列表
tools = [
    Tool(
        name="LangChain 文档检索",
        func=rag_qa,
        description="当需要回答关于 LangChain 框架的问题时使用"
    ),
    Tool(
        name="计算器",
        func=calculator,
        description="当需要执行数学计算时使用"
    ),
    Tool(
        name="网络搜索",
        func=search_engine,
        description="当需要获取最新信息或实时数据时使用"
    )
]

# --------------------- 步骤 3: 创建提示词模板 ---------------------
custom_prompt = PromptTemplate.from_template("""
你是一个全能助手，可以回答问题、执行计算和搜索信息。请遵循以下步骤：

1. 仔细分析用户问题
2. 根据问题类型选择合适工具
3. 使用工具处理问题
4. 用简洁的中文给出最终答案

请始终使用以下工具：
{tools}

历史对话记录：
{history}

问题：{input}
思考过程：""")

# --------------------- 步骤 4: 创建 Agent ---------------------
# 从 Hub 加载 ReAct 提示词（可自定义）
react_prompt = hub.pull("hwchase17/react")

# 创建 Agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt  # 可替换为 custom_prompt
)

# --------------------- 步骤 5: 执行 Agent ---------------------
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 显示详细执行过程
    handle_parsing_errors=True
)

# --------------------- 测试案例 ---------------------
questions = [
    "LangChain 的核心功能是什么？",
    "计算 (25 * 4) + (18 / 3) 的结果",
    "最新的人工智能发展动态是什么？"
]

for question in questions:
    print(f"\n用户问题：{question}")
    response = agent_executor.invoke({"input": question})
    print(f"最终答案：{response['output']}")