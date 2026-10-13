import argparse
import json
from pathlib import Path

from chandra.prompts import OCR_PROMPT
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", default="build/handwriting/model")
    parser.add_argument("--prompt-source", type=Path)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    messages = OCR_PROMPT
    if args.prompt_source is not None:
        source = json.loads(args.prompt_source.read_text())
        messages = [
            {"role": "system", "content": source["system_prompt"]},
            {"role": "user", "content": source["user_prompt"]},
        ]
    model, processor = load(args.model)
    prompt = apply_chat_template(
        processor, load_config(args.model), messages, num_images=1,
    )
    result = generate(
        model, processor, prompt, image=[str(args.image)],
        max_tokens=args.max_tokens, temperature=0.0, verbose=False,
    )
    print(result.text)


if __name__ == "__main__":
    main()
