import argparse
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen

PROMPT = (
    'Extract every row of this census table as JSON. Return only an object '
    'with key "rows", a list of objects with string fields "line", "name", '
    '"age", "occupation", "earnings". Read only the image. Preserve spelling '
    'and abbreviations, including ditto marks. Do not fill blanks or infer '
    'missing values. Use an empty string for an empty cell. Keep every row in '
    'printed order. No commentary.'
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", default="qwen3.5:2b")
    args = parser.parse_args()

    encoded = base64.b64encode(args.image.read_bytes()).decode()
    payload = {
        "model": args.model, "stream": False, "think": False,
        "messages": [{"role": "user", "content": PROMPT,
                      "images": [encoded]}],
        "options": {"temperature": 0, "num_predict": 4096,
                    "num_ctx": 16384},
        "keep_alive": "10m",
    }
    request = Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=600) as response:
        reply = json.load(response)
    print(reply["message"]["content"])


if __name__ == "__main__":
    main()
