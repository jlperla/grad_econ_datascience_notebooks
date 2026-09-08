from datasets import load_dataset
from transformers import pipeline

chips = load_dataset("cm93/eurosat", split="test[:10]")
labels = chips.features["label"].names
vit = pipeline("image-classification", model="Haifald/vit-eurosat")
predicted = [vit(row["image"], top_k=1)[0]["label"] for row in chips]
truth = [labels[i] for i in chips["label"]]
for t, p in zip(truth, predicted):
    print(f"{t:22s} {p}")
print("accuracy:", sum(p == t for p, t in zip(predicted, truth)) / len(truth))
