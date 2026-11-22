import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""
OpenAI chat completion wrapper for ncm-probe.
"""

import argparse
import json
import urllib.request
import urllib.error
from typing import Any, Optional
from src.config import OPENAI_API_KEY, get_required_env

def chat(
    messages: list[dict[str, str]],
    model: str = "gpt-4o-mini",
    response_format: Optional[dict[str, Any]] = None,
    temperature: float = 0.0,
) -> str:
    """
    Sends chat completion request to OpenAI API and returns string content.
    """
    api_key = OPENAI_API_KEY or get_required_env("OPENAI_API_KEY", "OpenAI API Key")
    url = "https://api.openai.com/v1/chat/completions"

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format:
        payload["response_format"] = response_format

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "ncm-probe/0.1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError(f"OpenAI returned no choices: {data}")
            return choices[0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API HTTPError ({e.code}): {err_msg}")
    except Exception as e:
        raise RuntimeError(f"OpenAI chat request failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="OpenAI chat wrapper CLI.")
    parser.add_argument("--prompt", default=None, help="Prompt text to test")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model identifier")
    args = parser.parse_args()

    if args.prompt:
        resp = chat([{"role": "user", "content": args.prompt}], model=args.model)
        print(resp)

if __name__ == "__main__":
    main()
