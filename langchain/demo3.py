from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
import math

# --------------------- 1. 初始化模型 ---------------------
llm = ChatOllama(
    model="llama3",  # 或 deepseek-r1:14b
    base_url="http://localhost:11434",
    temperature=0.3,
    system="""你是一个严谨的助手，必须严格按以下格式响应：
Thought: <思考过程>
Action: <工具名>
Action Input: <输入>
Observation: <工具返回结果>
Final Answer: <最终答案>"""
)

# --------------------- 2. 定义工具 ---------------------
# 工具1: 维基百科搜索
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))

# 工具2: Arxiv论文搜索
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())

# 工具3: 高级计算器
def advanced_calculator(input_str: str) -> str:
    """执行复杂数学计算（支持sin/cos/log等）"""
    try:
        # 安全评估数学表达式
        allowed_names = {"math": math}
        code = compile(input_str, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                raise ValueError(f"禁止使用的函数: {name}")
        result = eval(code, {"__builtins__": {}}, allowed_names)
        return f"Observation: {result}"
    except Exception as e:
        return f"Observation: 计算错误 - {str(e)}"

tools = [
    Tool(
        name="Wikipedia",
        func=lambda q: f"Observation: {wikipedia.run(q)}",
        description="用于查询事实性信息，输入应为具体问题"
    ),
    Tool(
        name="Arxiv",
        func=lambda q: f"Observation: {arxiv.run(q)}",
        description="用于搜索学术论文，输入应为技术关键词"
    ),
    Tool(
        name="Calculator",
        func=advanced_calculator,
        description="用于数学计算，输入应为表达式如 'sqrt(2)*sin(0.5)'"
    )
]

# --------------------- 3. 配置ReAct提示词 ---------------------
react_prompt = PromptTemplate.from_template("""
你是一个专业研究员，可以访问以下工具：

{tools}

请严格按以下格式响应：
Question: {input}
Thought: 分析问题并决定使用哪个工具
Action: 工具名（必须是[{tool_names}]之一）
Action Input: 工具需要的输入
Observation: 工具返回的结果
...（重复直到解决问题）
Final Answer: 最终答案

当前问题：{input}""")

# --------------------- 4. 创建Agent执行器 ---------------------
agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=lambda e: str(e)[:50]  # 截断错误信息
)

# --------------------- 5. 测试案例 ---------------------
queries = [
    "量子纠缠的最新理论研究进展是什么？",  # 需要Arxiv
    "爱因斯坦的出生地在哪里？",          # 需要Wikipedia
    "计算 sin(π/4) + log(100,10)",     # 需要Calculator
    "比较Python和Rust在并发编程上的差异"  # 需要多工具协作
]

for query in queries:
    print(f"\n\033[1;36m用户问题: {query}\033[0m")
    try:
        result = agent_executor.invoke({"input": query})
        print(f"\033[1;32m最终答案: {result['output']}\033[0m")
    except Exception as e:
        print(f"\033[1;31m执行错误: {str(e)}\033[0m")

# --------------------- 预期输出示例 ---------------------
"""
用户问题: 量子纠缠的最新理论研究进展是什么？
Thought: 我需要查询最新的量子纠缠理论研究论文
Action: Arxiv
Action Input: "quantum entanglement latest theory"
Observation: 1. Title: Entanglement Dynamics in... (2024)...
Thought: 找到3篇相关论文
Final Answer: 最新研究包括：1) Entanglement Dynamics in...(2024)...

用户问题: 计算 sin(π/4) + log(100,10)
Thought: 需要分解计算两个部分
Action: Calculator
Action Input: "math.sin(math.pi/4)"
Observation: 0.7071067811865475
Action: Calculator
Action Input: "math.log(100,10)"
Observation: 2.0
Thought: 将两部分相加
Action: Calculator
Action Input: "0.7071067811865475 + 2.0"
Observation: 2.7071067811865475
Final Answer: sin(π/4) + log(100,10) = 2.7071
"""