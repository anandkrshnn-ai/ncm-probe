import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""
Tavily search wrapper for ncm-probe.
"""

import argparse
import json
import urllib.request
import urllib.error
from src.config import TAVILY_API_KEY, get_required_env

def search(query: str, max_results: int = 5) -> list[str]:
    """
    Calls Tavily Search API and returns a list of snippet strings.
    """
    api_key = TAVILY_API_KEY or get_required_env("TAVILY_API_KEY", "Tavily Search API Key")
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "include_answer": False,
        "max_results": max_results,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "ncm-probe/0.1.0"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            snippets = []
            for r in results:
                content = r.get("content") or r.get("snippet") or ""
                if content:
                    snippets.append(content.strip())
            return snippets
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Tavily API HTTPError ({e.code}): {err_msg}")
    except Exception as e:
        raise RuntimeError(f"Tavily search request failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Tavily search wrapper CLI.")
    parser.add_argument("--query", default=None, help="Query string to search")
    parser.add_argument("--limit", type=int, default=5, help="Maximum snippets to return")
    args = parser.parse_args()

    if args.query:
        snippets = search(args.query, max_results=args.limit)
        for i, s in enumerate(snippets, 1):
            print(f"[{i}] {s}\n")

if __name__ == "__main__":
    main()
