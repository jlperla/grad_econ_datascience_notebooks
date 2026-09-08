import os

import numpy as np
from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.environ.get("LLM_API_KEY", "ollama"),
)
words = ["bank", "banks", "river", "money"]
response = client.embeddings.create(
    model=os.environ.get("EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
    input=words,
)
vectors = np.array([item.embedding for item in response.data])
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

for i, j in [(0, 1), (0, 2), (0, 3), (2, 3)]:
    print(f"sim({words[i]}, {words[j]}) = {vectors[i] @ vectors[j]:.4f}")
