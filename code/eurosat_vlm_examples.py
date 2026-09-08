import base64
import io
import os

from datasets import load_dataset
from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.environ.get("LLM_API_KEY", "ollama"),
)
chips = load_dataset("cm93/eurosat", split="test[:10]")
pool = load_dataset("cm93/eurosat", split="test[100:300]")
labels = chips.features["label"].names
question = (
    "Classify this satellite image as exactly one of: "
    + ", ".join(labels)
    + ". Answer with the label only."
)


def image_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


pool_truth = [labels[i] for i in pool["label"]]
examples = [{"type": "text", "text": "One labelled example of each class:"}]
for label in labels:
    url = image_url(pool[pool_truth.index(label)]["image"])
    examples += [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": label},
    ]


def classify(image):
    response = client.chat.completions.create(
        model=os.environ.get("LLM_MODEL", "qwen3.5:4b"),
        temperature=0,
        reasoning_effort="none",
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
    )
    return response.choices[0].message.content.strip()


predicted = [classify(row["image"]) for row in chips]
truth = [labels[i] for i in chips["label"]]
for t, p in zip(truth, predicted):
    print(f"{t:22s} {p}")
print("accuracy:", sum(p == t for p, t in zip(predicted, truth)) / len(truth))
