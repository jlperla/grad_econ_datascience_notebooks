import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE_MODEL = "datalab-to/chandra"
BASE_REVISION = "bd40c21576564d31ba99f204e28ccd5c4c1751fa"
ADAPTERS = {
    "original": ("Saucystream/fine_tuned_chandra",
                 "d94b4ca422dd87e06b55abdd955bb199879a545a", "argentina-model"),
    "blur": ("Saucystream/fine_tuned_chandra_blur",
             "6a103e674149b2aed54e824fe41ab09838443e7e", "argentina-blur-model"),
}
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", choices=ADAPTERS, default="original")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    adapter, revision, directory = ADAPTERS[args.adapter]
    output = args.output or ROOT / "build/handwriting" / directory
    torch.set_num_threads(8)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, dtype=torch.bfloat16,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(
        model, adapter, revision=revision,
    )
    lora_modules = sum(hasattr(m, "lora_A") for m in model.modules())
    print("LoRA modules:", lora_modules)
    model = model.merge_and_unload()
    model.save_pretrained(output)
    # MLX uses the base model's rope_theta/rope_scaling configuration.
    shutil.copyfile(hf_hub_download(
        BASE_MODEL, "config.json", revision=BASE_REVISION,
    ), output / "config.json")
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION,
    )
    processor.save_pretrained(output)
    (output / "merge_source.json").write_text(json.dumps({
        "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
        "adapter": adapter, "adapter_revision": revision,
        "dtype": "bfloat16", "method": "PEFT merge_and_unload",
        "lora_modules": lora_modules,
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
