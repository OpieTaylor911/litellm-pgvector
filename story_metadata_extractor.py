#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import litellm
from dotenv import load_dotenv
from pydantic import ValidationError

from models import StoryMetadata


SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")

DEFAULT_API_BASE = os.environ.get("STORY_METADATA_API_BASE", "http://127.0.0.1:18001")
DEFAULT_API_KEY = os.environ.get("STORY_METADATA_API_KEY", os.environ.get("SERVER_API_KEY", ""))
DEFAULT_LLM_MODEL = os.environ.get("STORY_METADATA_LLM_MODEL", "custom_openai/qwen/qwen3.6-27b")
DEFAULT_LLM_API_BASE = os.environ.get("STORY_METADATA_LLM_API_BASE", os.environ.get("EMBEDDING__BASE_URL", "http://192.168.254.66:4000"))
DEFAULT_LLM_API_KEY = os.environ.get("STORY_METADATA_LLM_API_KEY", os.environ.get("EMBEDDING__API_KEY", "sk-oW1ZHol1wBXa0oRR0OQBlA"))


PROMPT_TEMPLATE = """You are extracting structured ROMANCE FICTION metadata for ONE COMPLETE STORY.

Return ONLY valid JSON matching this schema exactly. No markdown. No explanation.

Rules:
- Analyze the ENTIRE story, not just one scene.
- Use only evidence present in the text. If unknown, leave arrays empty, booleans false, strings null, and heat_level=1.
- Normalize tags to short lowercase phrases except fixed enums like HEA/HFN.
- Keep arrays deduplicated.
- search_keywords should contain the best retrieval phrases a reader would search for.
- tone may contain multiple values from: wholesome, emotional, erotic, suspenseful, humorous.
- pov must be one of: dual, first person, third person, or null.
- explicitness must be one of: kissing, fade to black, explicit, very explicit, or null.
- relationship_structure must be one of: monogamous, open, poly, or null.
- ending must be one of: HEA, HFN, or null.
- heat_level must be an integer from 1 to 5.

JSON schema:
{schema_json}

Story filename: {filename}

Story text:
<<<STORY
{story_text}
STORY
>>>
"""


def api_request(method: str, url: str, api_key: str, json_body: dict[str, Any] | None = None) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    body = None
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if json_body is not None:
        body = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url, data=body, method=method, headers=headers)
    try:
        with urlopen(request) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload) if payload else {}
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: HTTP {exc.code} - {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def build_prompt(filename: str, story_text: str) -> str:
    schema_json = json.dumps(StoryMetadata.model_json_schema(), indent=2)
    return PROMPT_TEMPLATE.format(
        schema_json=schema_json,
        filename=filename,
        story_text=story_text,
    )


async def extract_story_metadata(filename: str, story_text: str, model: str, api_base: str, api_key: str) -> StoryMetadata:
    prompt = build_prompt(filename, story_text)
    response = await litellm.acompletion(
        model=model,
        api_base=api_base,
        api_key=api_key,
        messages=[
            {"role": "system", "content": "Return only valid JSON matching the requested schema."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    content = response.choices[0].message.content or "{}"
    parsed = json.loads(strip_code_fences(content))
    return StoryMetadata.model_validate(parsed)


async def extract_with_retry(filename: str, story_text: str, model: str, api_base: str, api_key: str, retries: int = 2) -> StoryMetadata:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return await extract_story_metadata(filename, story_text, model, api_base, api_key)
        except (json.JSONDecodeError, ValidationError, KeyError, TypeError, ValueError) as exc:
            last_error = exc
            if attempt >= retries:
                raise RuntimeError(f"Structured extraction failed after {retries + 1} attempts: {exc}") from exc
            time.sleep(1.0)
    raise RuntimeError(f"Structured extraction failed: {last_error}")


def upload_story_metadata(api_base: str, api_key: str, vector_store_id: str, filename: str, metadata: StoryMetadata) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}/v1/vector_stores/{vector_store_id}/stories/{filename}/metadata"
    return api_request("PUT", url, api_key, json_body=metadata.model_dump())


async def main() -> int:
    parser = argparse.ArgumentParser(description="Single-pass whole-story metadata extractor prototype")
    parser.add_argument("story_file", help="Path to a .txt story file")
    parser.add_argument("--vector-store-id", help="If provided with --upload, upload tags into this vector store record")
    parser.add_argument("--upload", action="store_true", help="Upload extracted metadata into the API")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-api-base", default=DEFAULT_LLM_API_BASE)
    parser.add_argument("--llm-api-key", default=DEFAULT_LLM_API_KEY)
    parser.add_argument("--output", help="Optional path to write extracted JSON")
    args = parser.parse_args()

    story_path = Path(args.story_file)
    if not story_path.is_file():
        print(f"ERROR: story file not found: {story_path}", file=sys.stderr)
        return 1

    story_text = story_path.read_text(encoding="utf-8", errors="replace")
    metadata = await extract_with_retry(
        filename=story_path.name,
        story_text=story_text,
        model=args.model,
        api_base=args.llm_api_base,
        api_key=args.llm_api_key,
    )

    payload = metadata.model_dump()
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.upload:
        if not args.vector_store_id:
            print("ERROR: --vector-store-id is required with --upload", file=sys.stderr)
            return 2
        result = upload_story_metadata(args.api_base, args.api_key, args.vector_store_id, story_path.name, metadata)
        print("\nUploaded:\n" + json.dumps(result, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(main()))
