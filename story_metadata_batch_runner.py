#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from story_metadata_extractor import extract_with_retry, upload_story_metadata


async def process_story(
    story_path: Path,
    *,
    model: str,
    llm_api_base: str,
    llm_api_key: str,
    output_dir: Path | None,
    upload: bool,
    api_base: str,
    api_key: str,
    vector_store_id: str | None,
) -> dict[str, Any]:
    text = story_path.read_text(encoding="utf-8", errors="replace")
    metadata = await extract_with_retry(
        filename=story_path.name,
        story_text=text,
        model=model,
        api_base=llm_api_base,
        api_key=llm_api_key,
    )

    payload = metadata.model_dump()
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{story_path.name}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    upload_result = None
    if upload:
        if not vector_store_id:
            raise RuntimeError("vector_store_id is required for upload")
        upload_result = upload_story_metadata(api_base, api_key, vector_store_id, story_path.name, metadata)

    return {
        "filename": story_path.name,
        "metadata": payload,
        "uploaded": bool(upload_result),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Batch whole-story metadata extractor")
    parser.add_argument("input_dir", help="Directory containing .txt stories")
    parser.add_argument("--output-dir", default="extracted-story-metadata")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--llm-api-base", default="http://127.0.0.1:4000")
    parser.add_argument("--llm-api-key", default="sk-1234")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--api-base", default="http://127.0.0.1:18001")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--vector-store-id")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"input_dir not found: {input_dir}")

    files = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt")
    if args.limit > 0:
        files = files[: args.limit]

    output_dir = Path(args.output_dir) if args.output_dir else None
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

    print(json.dumps({"count": len(results), "files": [r["filename"] for r in results]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
