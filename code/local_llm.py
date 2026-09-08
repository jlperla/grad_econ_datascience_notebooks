import os

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.environ.get("LLM_API_KEY", "ollama"),
)
question = "Explain comparative advantage in two sentences."
response = client.chat.completions.create(
    model=os.environ.get("LLM_MODEL", "qwen3.5:2b"),
    messages=[
        {"role": "system", "content": "You are a concise economics tutor."},
        {"role": "user", "content": question},
    ],
)
print(response.choices[0].message.content)
