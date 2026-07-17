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


class CharacterNode(BaseModel):
    name: str
    role: Optional[str] = None
    archetypes: List[str] = []
    occupations: List[str] = []
    traits: List[str] = []


class RelationshipEdge(BaseModel):
    source: str
    target: str
    relationship_type: str
    status: Optional[str] = None
    notes: List[str] = []


class TimelineBeat(BaseModel):
    order: int
    label: str
    summary: str
    location: Optional[str] = None
    emotional_turn: Optional[str] = None


class StoryGraph(BaseModel):
    characters: List[CharacterNode] = []
    relationships: List[RelationshipEdge] = []
    timeline: List[TimelineBeat] = []


PROMPT = """Analyze the COMPLETE story and return only valid JSON matching this schema exactly.
No markdown. No explanation.

Goals:
- Extract the main characters.
- Extract relationship edges between them.
- Extract a compact ordered timeline of major beats.
- Use whole-story context, not isolated scenes.

Schema:
{schema}

Story text:
<<<STORY
{story}
STORY
>>>
"""


async def main() -> int:
    parser = argparse.ArgumentParser(description="Whole-story character/relationship/timeline extractor")
    parser.add_argument("story_file")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--api-base", default="http://127.0.0.1:4000")
    parser.add_argument("--api-key", default="sk-1234")
    args = parser.parse_args()

    story_path = Path(args.story_file)
    story_text = story_path.read_text(encoding="utf-8", errors="replace")
    schema = json.dumps(StoryGraph.model_json_schema(), indent=2)

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
    graph = StoryGraph.model_validate(json.loads(content))
    print(json.dumps(graph.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
