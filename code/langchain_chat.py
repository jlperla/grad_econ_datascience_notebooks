from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:11434/v1", api_key="ollama", model="qwen3.5:2b"
)
question = "Explain comparative advantage in two sentences."
response = llm.invoke(
    [
        SystemMessage("You are a concise economics tutor."),
        HumanMessage(question),
    ]
)
print(response.content)
