# Cached comparison of local Qwen and OpenAI text embeddings.
# Rebuild: uv run python utilities/build_embedding_comparison.py
import os

import numpy as np
import pandas as pd
from openai import OpenAI

words = ["bank", "banks", "river", "money"]
pairs = [(0, 1), (0, 2), (0, 3), (2, 3)]
models = [
    (
        "Local Qwen",
        "qwen3-embedding:0.6b",
        OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    ),
    (
        "OpenAI",
        "text-embedding-3-large",
        OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
    ),
]

rows = []
for provider, model, client in models:
    response = client.embeddings.create(model=model, input=words)
    vectors = np.array([item.embedding for item in response.data])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    for pair_order, (i, j) in enumerate(pairs):
        rows.append(
            {
                "pair_order": pair_order,
                "word_1": words[i],
                "word_2": words[j],
                "provider": provider,
                "model": model,
                "dimensions": vectors.shape[1],
                "cosine_similarity": vectors[i] @ vectors[j],
            }
        )

comparison = pd.DataFrame(rows)
comparison.to_csv(
    "slides/data/embedding_bank_comparison.csv",
    index=False,
    float_format="%.8f",
)
print(comparison.to_string(index=False))
