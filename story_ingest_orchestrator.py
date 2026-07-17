#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from story_metadata_extractor import extract_with_retry, upload_story_metadata
from story_graph_extractor import StoryGraph, PROMPT as GRAPH_PROMPT
from story_scene_splitter import SceneSplitResult, PROMPT as SCENE_PROMPT
import litellm


async def run_scene_splitter(story_text: str, model: str, api_base: str, api_key: str) -> dict[str, Any]:
    schema = json.dumps(SceneSplitResult.model_json_schema(), indent=2)
    response = await litellm.acompletion(
        model=model,
        api_base=api_base,
        api_key=api_key,
        messages=[
            {"role": "system", "content": "Return only valid JSON matching the requested schema."},
            {"role": "user", "content": SCENE_PROMPT.format(schema=schema, story=story_text)},
        ],
        temperature=0.1,
    )
    content = (response.choices[0].message.content or "{}").strip()
    if content.startswith("```"):
        content = "\n".join(content.splitlines()[1:-1]).strip()
    return SceneSplitResult.model_validate(json.loads(content)).model_dump()


async def run_story_graph(story_text: str, model: str, api_base: str, api_key: str) -> dict[str, Any]:
    schema = json.dumps(StoryGraph.model_json_schema(), indent=2)
    response = await litellm.acompletion(
        model=model,
        api_base=api_base,
        api_key=api_key,
        messages=[
            {"role": "system", "content": "Return only valid JSON matching the requested schema."},
            {"role": "user", "content": GRAPH_PROMPT.format(schema=schema, story=story_text)},
        ],
        temperature=0.1,
    )
    content = (response.choices[0].message.content or "{}").strip()
    if content.startswith("```"):
        content = "\n".join(content.splitlines()[1:-1]).strip()
    return StoryGraph.model_validate(json.loads(content)).model_dump()


async def process_story(
    story_path: Path,
    *,
    model: str,
    llm_api_base: str,
    llm_api_key: str,
    output_dir: Path,
    upload: bool,
    api_base: str,
    api_key: str,
    vector_store_id: str | None,
) -> dict[str, Any]:
    story_text = story_path.read_text(encoding="utf-8", errors="replace")

    metadata = await extract_with_retry(
        filename=story_path.name,
        story_text=story_text,
        model=model,
        api_base=llm_api_base,
        api_key=llm_api_key,
    )
    scenes = await run_scene_splitter(story_text, model, llm_api_base, llm_api_key)
    graph = await run_story_graph(story_text, model, llm_api_base, llm_api_key)

    story_dir = output_dir / story_path.stem
    story_dir.mkdir(parents=True, exist_ok=True)

    (story_dir / "metadata.json").write_text(
        json.dumps(metadata.model_dump(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (story_dir / "scenes.json").write_text(
        json.dumps(scenes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (story_dir / "graph.json").write_text(
        json.dumps(graph, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if upload:
        if not vector_store_id:
            raise RuntimeError("--vector-store-id is required with --upload")
        upload_story_metadata(api_base, api_key, vector_store_id, story_path.name, metadata)

    return {
        "filename": story_path.name,
        "output_dir": str(story_dir),
        "scene_count": len(scenes.get("scenes", [])),
        "character_count": len(graph.get("characters", [])),
        "uploaded": upload,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full 3-part ingest prototype")
    parser.add_argument("input_path", help="Path to a .txt story file or a directory of .txt files")
    parser.add_argument("--output-dir", default="story-ingest-output")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--llm-api-base", default="http://127.0.0.1:4000")
    parser.add_argument("--llm-api-key", default="sk-1234")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--api-base", default="http://127.0.0.1:18001")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--vector-store-id")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() == ".txt")
    else:
        raise SystemExit(f"input_path not found: {input_path}")

    if args.limit > 0:
        files = files[: args.limit]

    results = []
    for story_path in files:
        print(f"Processing {story_path.name}...")
        result = await process_story(
            story_path,
            model=args.model,
            llm_api_base=args.llm_api_base,
            llm_api_key=args.llm_api_key,
            output_dir=output_dir,
            upload=args.upload,
            api_base=args.api_base,
            api_key=args.api_key,
            vector_store_id=args.vector_store_id,
        )
        results.append(result)

    print(json.dumps({"count": len(results), "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
