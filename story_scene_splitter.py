#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import List, Optional

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel


load_dotenv(Path(__file__).resolve().parent / ".env")


class SceneSummary(BaseModel):
    scene_number: int
    title: Optional[str] = None
    summary: str
    participants: List[str] = []
    location: Optional[str] = None
    time_marker: Optional[str] = None
    emotional_beat: Optional[str] = None
    conflict_beat: Optional[str] = None
    keywords: List[str] = []
    text_excerpt: str


class SceneSplitResult(BaseModel):
    scenes: List[SceneSummary] = []


PROMPT = """Split the COMPLETE story into scenes and return only valid JSON matching this schema exactly.
No markdown. No explanation.

Rules:
- A scene is a meaningful shift in place, time, or dramatic beat.
- Keep summaries concise.
- text_excerpt should be a representative excerpt for the scene, not the full scene.
- Use whole-story understanding for participants and conflict.

Schema:
{schema}

Story text:
<<<STORY
{story}
STORY
>>>
"""


async def main() -> int:
    parser = argparse.ArgumentParser(description="Scene splitter + summary extractor")
    parser.add_argument("story_file")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--api-base", default="http://127.0.0.1:4000")
    parser.add_argument("--api-key", default="sk-1234")
    args = parser.parse_args()

    story_path = Path(args.story_file)
    story_text = story_path.read_text(encoding="utf-8", errors="replace")
    schema = json.dumps(SceneSplitResult.model_json_schema(), indent=2)

    response = await litellm.acompletion(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        messages=[
            {"role": "system", "content": "Return only valid JSON matching the requested schema."},
            {"role": "user", "content": PROMPT.format(schema=schema, story=story_text)},
        ],
        temperature=0.1,
    )
    content = (response.choices[0].message.content or "{}").strip()
    if content.startswith("```"):
        lines = content.splitlines()[1:-1]
        content = "\n".join(lines).strip()
    result = SceneSplitResult.model_validate(json.loads(content))
    print(json.dumps(result.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
