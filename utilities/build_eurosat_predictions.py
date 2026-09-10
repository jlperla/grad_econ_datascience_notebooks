# Cached EuroSAT predictions from local vision-language models and a fine-tuned ViT.
# Rebuild: uv run python utilities/build_eurosat_predictions.py
import base64
import io
import os
import subprocess
import time

import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from transformers import pipeline

chips = load_dataset("cm93/eurosat", split="test[:100]")
labels = chips.features["label"].names
truth = [labels[i] for i in chips["label"]]
local = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
hosted = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=10)
question = (
    "Classify this satellite image as exactly one of: "
    + ", ".join(labels)
    + ". Answer with the label only."
)


def image_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def classify(image, model, client, examples=(), **options):
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    *examples,
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_url(image)}},
                ],
            }
        ],
        **options,
    )
    return response.choices[0].message.content.strip()


pool = load_dataset("cm93/eurosat", split="test[100:300]")
pool_truth = [labels[i] for i in pool["label"]]
examples = [{"type": "text", "text": "One labelled example of each class:"}]
for label in labels:
    url = image_url(pool[pool_truth.index(label)]["image"])
    examples += [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": label},
    ]

rows = []
for model in ["qwen3.5:2b", "qwen3.5:4b", "qwen3.8:27b-mlx", "qwen3-vl:4b-instruct"]:
    for n_examples, prefix in [(0, ()), (10, examples)]:
        for i, row in enumerate(chips):
            answer = classify(
                row["image"], model, local, prefix, reasoning_effort="none"
            )
            rows.append(
                {
                    "index": i,
                    "truth": truth[i],
                    "model": model,
                    "examples": n_examples,
                    "answer": answer,
                }
            )
    subprocess.run(["ollama", "stop", model], check=True)  # three loaded exhaust memory
for i, row in enumerate(chips):
    rows.append(
        {
            "index": i,
            "truth": truth[i],
            "model": "gpt-4o-mini",
            "examples": 0,
            "answer": classify(row["image"], "gpt-4o-mini", hosted),
        }
    )
    time.sleep(3)  # a chip counts as 8,500 tokens against a 200K per minute limit
vit = pipeline("image-classification", model="Haifald/vit-eurosat")
for i, out in enumerate(vit(list(chips["image"]), top_k=1, batch_size=16)):
    rows.append(
        {
            "index": i,
            "truth": truth[i],
            "model": "Haifald/vit-eurosat",
            "examples": 0,
            "answer": out[0]["label"],
        }
    )

predictions = pd.DataFrame(rows)
predictions["predicted"] = predictions["answer"].where(
    predictions["answer"].isin(labels), ""
)
predictions.to_csv("slides/data/eurosat_predictions.csv", index=False)
correct = predictions["predicted"] == predictions["truth"]
print(
    correct.groupby([predictions["model"], predictions["examples"]]).mean().to_string()
)
