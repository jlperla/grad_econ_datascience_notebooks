# Financial PhraseBank v1.0, sentences on which all annotators agreed.
# Malo, Sinha, Korhonen, Wallenius, Takala (2014), "Good debt or bad debt:
# Detecting semantic orientations in economic texts", JASIST 65(4).
# CC BY-NC-SA 3.0. Archive hosted on the Hugging Face Hub as
# takala/financial_phrasebank.
# Rebuild: uv run python utilities/build_phrasebank.py
import zipfile
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download("takala/financial_phrasebank",
                       "data/FinancialPhraseBank-v1.0.zip", repo_type="dataset")
with zipfile.ZipFile(path) as z:
    df = pd.read_csv(z.open("FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"),
                     sep="@", header=None, names=["sentence", "label"],
                     encoding="latin-1")
df.to_csv("slides/data/financial_phrasebank.csv.gz", index=False)
print(df.shape, df["label"].value_counts().to_dict())
