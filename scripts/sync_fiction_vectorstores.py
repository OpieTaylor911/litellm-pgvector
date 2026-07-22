#!/usr/bin/env python3

import argparse
import json
import mimetypes
import os
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

load_dotenv(PROJECT_DIR / ".env")

RESET_FILE_COUNTS = {
    "in_progress": 0,
    "completed": 0,
    "failed": 0,
    "cancelled": 0,
    "total": 0,
}


def api_request(
    method: str,
    url: str,
    api_key: str,
    *,
    json_body: dict[str, Any] | None = None,
    raw_body: bytes | None = None,
    content_type: str | None = None,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    body = None
    if json_body is not None:
        body = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif raw_body is not None:
        body = raw_body
        if content_type:
            headers["Content-Type"] = content_type

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


def build_multipart_body(
    fields: dict[str, str],
    files: list[tuple[str, Path]],
) -> tuple[bytes, str]:
    boundary = f"----fiction-sync-{uuid.uuid4().hex}"
    parts: list[bytes] = []

    for name, value in fields.items():
        parts.extend(
            [
                f"--{boundary}".encode(),
                f'Content-Disposition: form-data; name="{name}"'.encode(),
                b"",
                str(value).encode("utf-8"),
            ]
        )

    for field_name, file_path in files:
        content_type = mimetypes.guess_type(file_path.name)[0] or "text/plain"
        file_bytes = file_path.read_bytes()
        parts.extend(
            [
                f"--{boundary}".encode(),
                (
                    f'Content-Disposition: form-data; name="{field_name}"; '
                    f'filename="{file_path.name}"'
                ).encode(),
                f"Content-Type: {content_type}".encode(),
                b"",
                file_bytes,
            ]
        )

    parts.append(f"--{boundary}--".encode())
    parts.append(b"")

    body = b"\r\n".join(parts)
    return body, f"multipart/form-data; boundary={boundary}"


def list_vector_stores(api_base: str, api_key: str) -> list[dict[str, Any]]:
    stores: list[dict[str, Any]] = []
    after: str | None = None

    while True:
        params = {"limit": 100}
        if after:
            params["after"] = after

        url = f"{api_base.rstrip('/')}/v1/vector_stores?{urlencode(params)}"
        payload = api_request("GET", url, api_key)

        batch = payload.get("data", [])
        stores.extend(batch)

        if not payload.get("has_more") or not batch:
            break

        after = payload.get("last_id")
        if not after:
            break

    return stores


def choose_preferred_store(matches: list[dict[str, Any]]) -> dict[str, Any]:
    def sort_key(store: dict[str, Any]) -> tuple[int, int, int]:
        file_counts = store.get("file_counts") or {}
        total = int(file_counts.get("total", 0) or 0)
        usage_bytes = int(store.get("usage_bytes", 0) or 0)
        created_at = int(store.get("created_at", 0) or 0)
        return (total, usage_bytes, created_at)

    return max(matches, key=sort_key)


def normalize_store_name(name: str) -> str:
    return " ".join((name or "").split()).lower()


def create_vector_store(api_base: str, api_key: str, topic: str) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}/v1/vector_stores"
    return api_request(
        "POST",
        url,
        api_key,
        json_body={
            "name": topic,
            "metadata": {
                "topic": topic,
                "source": "fiction-sync-script",
            },
        },
    )


def upload_text_files(
    api_base: str,
    api_key: str,
    vector_store_id: str,
    txt_files: list[Path],
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}/v1/vector_stores/upload-text-files"

    fields = {
        "vector_store_id": vector_store_id,
        "chunk_size": str(chunk_size),
        "chunk_overlap": str(chunk_overlap),
    }
    files = [("files", path) for path in txt_files]

    body, content_type = build_multipart_body(fields, files)
    return api_request(
        "POST",
        url,
        api_key,
        raw_body=body,
        content_type=content_type,
    )


def normalize_database_url(database_url: str) -> str:
    parts = urlsplit(database_url)
    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() != "schema"
    ]
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(filtered_query), parts.fragment)
    )


def clear_vector_store(database_url: str, vector_store_id: str) -> None:
    import psycopg

    normalized_url = normalize_database_url(database_url)

    with psycopg.connect(normalized_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM embeddings
                WHERE vector_store_id = %s
                """,
                (vector_store_id,),
            )
            # story_metadata (added after this script was originally written)
            # is keyed by vector_store_id + filename and is not touched by the
            # embeddings delete above, so it must be cleared explicitly or
            # stale per-file tags for removed/renamed files would linger.
            cur.execute(
                """
                DELETE FROM story_metadata
                WHERE vector_store_id = %s
                """,
                (vector_store_id,),
            )
            cur.execute(
                """
                UPDATE vector_stores
                SET
                    file_counts = %s::jsonb,
                    usage_bytes = 0,
                    status = 'completed',
                    last_active_at = NOW()
                WHERE id = %s
                """,
                (json.dumps(RESET_FILE_COUNTS), vector_store_id),
            )
        conn.commit()


def gather_topics(fiction_root: Path) -> list[tuple[str, Path, list[Path]]]:
    topics: list[tuple[str, Path, list[Path]]] = []

    for topic_dir in sorted(p for p in fiction_root.iterdir() if p.is_dir()):
        source_dir = topic_dir / "source"
        if not source_dir.is_dir():
            continue

        txt_files = sorted(
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".txt"
        )
        topics.append((topic_dir.name, source_dir, txt_files))

    return topics


def write_mapping_file(mapping_file: Path, mapping: dict[str, str]) -> None:
    mapping_file.parent.mkdir(parents=True, exist_ok=True)
    mapping_file.write_text(
        json.dumps(mapping, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync fiction/<topic>/source/*.txt into one vector store per topic."
    )
    parser.add_argument(
        "--fiction-root",
        default="/mnt/commandjobs/fiction",
        help="Root directory containing topic folders",
    )
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:18001",
        help="Base URL for the vector store API",
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="Bearer token for the vector store API",
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help=(
            "Postgres connection string used to clear existing vector stores in place "
            "(defaults to DATABASE_URL from .env)"
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1600,
        help="Chunk size for upload-text-files",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap for upload-text-files",
    )
    parser.add_argument(
        "--mapping-file",
        default="/mnt/commandjobs/fiction/.vectorstore_ids.json",
        help="Where to write topic -> vector_store_id mapping",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show actions without changing vector stores",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Skip topics whose source folder has no .txt files",
    )
    args = parser.parse_args()

    fiction_root = Path(args.fiction_root)
    mapping_file = Path(args.mapping_file)

    print(f"fiction_root={fiction_root}")
    print(f"api_base={args.api_base}")
    print(f"mapping_file={mapping_file}")
    print(f"dry_run={args.dry_run}")

    if not fiction_root.is_dir():
        print(f"ERROR: fiction root not found: {fiction_root}", file=sys.stderr)
        return 1

    if args.chunk_overlap >= args.chunk_size:
        print("ERROR: --chunk-overlap must be smaller than --chunk-size", file=sys.stderr)
        return 1

    if not args.dry_run and not args.database_url:
        print(
            "ERROR: --database-url not provided and DATABASE_URL was not found in .env",
            file=sys.stderr,
        )
        return 1

    topics = gather_topics(fiction_root)
    print(f"Detected {len(topics)} topic directories with a source/ subdirectory.")

    if not topics:
        print("No topic/source directories found.")
        return 1

    print("Loading vector stores from API...")
    stores = list_vector_stores(args.api_base, args.api_key)
    print(f"Loaded {len(stores)} existing vector stores from API.\n")

    stores_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for store in stores:
        stores_by_name[normalize_store_name(store["name"])].append(store)

    topic_to_store_id: dict[str, str] = {}
    failed_uploads: list[dict[str, str]] = []

    for topic, source_dir, txt_files in topics:
        print(f"=== {topic} ===")
        print(f"source: {source_dir}")
        print(f"txt_files_found={len(txt_files)}")

        if not txt_files and args.skip_empty:
            print("No .txt files found, skipping due to --skip-empty.\n")
            continue

        topic_key = normalize_store_name(topic)
        matches = stores_by_name.get(topic_key, [])
        if len(matches) > 1:
            match_ids = ", ".join(store["id"] for store in matches)
            print(f"WARNING: multiple vector stores named '{topic}': {match_ids}")

        if matches:
            selected_store = choose_preferred_store(matches)
            vector_store_id = selected_store["id"]
            print(f"Using existing vector store: {vector_store_id}")
        else:
            if args.dry_run:
                vector_store_id = f"DRY_RUN_{topic}"
                print("Would create new vector store.")
            else:
                created_store = create_vector_store(args.api_base, args.api_key, topic)
                vector_store_id = created_store["id"]
                print(f"Created vector store: {vector_store_id}")

        topic_to_store_id[topic] = vector_store_id

        if args.dry_run:
            print(f"Would refresh store {vector_store_id} with {len(txt_files)} .txt file(s).")
            for path in txt_files:
                print(f"  - {path.name}")
            print()
            continue

        print(f"Clearing existing embeddings for vector_store_id={vector_store_id} ...")
        clear_vector_store(args.database_url, vector_store_id)

        if not txt_files:
            print("No .txt files found; store cleared and left empty.\n")
            continue

        successful_files = 0
        failed_files = 0
        total_embeddings_created = 0

        print(f"Uploading {len(txt_files)} .txt file(s) one at a time...")
        for path in txt_files:
            print(f"  -> {path.name}")
            try:
                result = upload_text_files(
                    args.api_base,
                    args.api_key,
                    vector_store_id,
                    [path],
                    args.chunk_size,
                    args.chunk_overlap,
                )
                embeddings_created = int(result.get("embeddings_created", 0) or 0)
                total_embeddings_created += embeddings_created
                successful_files += 1
                print(f"     OK embeddings_created={embeddings_created}")
            except Exception as exc:
                failed_files += 1
                failed_uploads.append(
                    {
                        "topic": topic,
                        "vector_store_id": vector_store_id,
                        "file": path.name,
                        "error": str(exc),
                    }
                )
                print(f"     ERROR {path.name}: {exc}", file=sys.stderr)

        print(
            "Topic complete: "
            f"successful_files={successful_files}, "
            f"failed_files={failed_files}, "
            f"embeddings_created={total_embeddings_created}\n"
        )

    if args.dry_run:
        print("Dry run complete. Mapping file not written.")
    else:
        write_mapping_file(mapping_file, topic_to_store_id)
        print(f"Mapping written to {mapping_file}")

    print("\nTopic -> vector_store_id")
    for topic in sorted(topic_to_store_id):
        print(f"{topic}: {topic_to_store_id[topic]}")

    if failed_uploads:
        print("\nFailed uploads:")
        for item in failed_uploads:
            print(
                f"- topic={item['topic']} "
                f"store={item['vector_store_id']} "
                f"file={item['file']} "
                f"error={item['error']}"
            )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
