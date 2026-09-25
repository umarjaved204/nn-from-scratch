"""Download the Tiny Shakespeare corpus (~1 MB) into part2_gpt/data/input.txt."""

import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "input.txt"


def main() -> None:
    if OUT_PATH.exists():
        print(f"Already downloaded: {OUT_PATH}")
        return
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Tiny Shakespeare to {OUT_PATH}...")
    urllib.request.urlretrieve(URL, OUT_PATH)
    text = OUT_PATH.read_text(encoding="utf-8")
    print(f"Done: {len(text):,} characters, {len(set(text))} unique")


if __name__ == "__main__":
    main()
