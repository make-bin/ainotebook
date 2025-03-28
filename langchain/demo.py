from langchain_ollama import ChatOllama


llm = ChatOllama(
  model= "deepseek-r1:14b",
  base_url="http://10.230.66.59:11434",
 # format="json"
)

res = llm.invoke("你是谁？")

print(res)