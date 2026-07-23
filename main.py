import os
import asyncio
import time
import json
import secrets
import re
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional
from fastapi import FastAPI, HTTPException, Depends, File, Form, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from prisma import Prisma
from dotenv import load_dotenv

from models import (
    VectorStoreCreateRequest,
    VectorStoreResponse,
    VectorStoreSearchRequest,
    VectorStoreSearchResponse,
    SearchResult,
    EmbeddingCreateRequest,
    EmbeddingResponse,
    EmbeddingBatchCreateRequest,
    EmbeddingBatchCreateResponse,
    VectorStoreListResponse,
    VectorStoreExposureUpdateRequest,
    ContentChunk,
    SceneSearchRequest,
    SceneSearchResponse,
    SceneSearchResult,
    StoryMetadata,
    StoryMetadataResponse,
    StoryMetadataBulkUpsertRequest,
    StoryMetadataExtractionRequest,
    StoryFileInfo,
)
from config import settings
from embedding_service import embedding_service
from story_metadata_extractor import extract_with_retry

load_dotenv()

UPLOAD_PAGE_PATH = Path(__file__).parent / "ui" / "index.html"

app = FastAPI(
    title="OpenAI Vector Stores API",
    description="OpenAI-compatible Vector Stores API using PGVector",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Prisma client
db = Prisma()
activity_events: deque[dict[str, Any]] = deque(maxlen=2000)
activity_event_id = 0
vector_store_ingest_jobs: dict[str, dict[str, Any]] = {}
ingest_job_semaphore = asyncio.Semaphore(2)

PROJECT_DIR = Path(__file__).parent
INGEST_SCRIPT_PATH = PROJECT_DIR / "scripts" / "sync_new_fiction_stories.sh"
INGEST_LOG_DIR = PROJECT_DIR / "logs" / "ingest-jobs"
STORY_METADATA_MAX_CHARS = max(2000, int(os.getenv("STORY_METADATA_MAX_CHARS", "20000")))
INGEST_WATCHDOG_RETRIES = max(0, int(os.getenv("INGEST_WATCHDOG_RETRIES", "3")))
INGEST_WATCHDOG_RETRY_DELAY_SECONDS = max(1, int(os.getenv("INGEST_WATCHDOG_RETRY_DELAY_SECONDS", "15")))

security = HTTPBearer()
ui_security = HTTPBasic()


async def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate API key from Authorization header"""
    expected_key = settings.server_api_key
    if credentials.credentials != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials


def record_activity(event_type: str, message: str, **data: Any) -> dict[str, Any]:
    """Record in-memory activity events for UI live-tail polling."""
    global activity_event_id
    activity_event_id += 1
    event = {
        "id": activity_event_id,
        "timestamp": int(time.time()),
        "type": event_type,
        "message": message,
    }
    if data:
        event["data"] = data
    activity_events.append(event)
    return event


def _new_ingest_job_id() -> str:
    return secrets.token_hex(8)


async def _run_ingest_job(job_id: str, vector_store_id: str, store_name: str) -> None:
    job = vector_store_ingest_jobs[job_id]
    await ingest_job_semaphore.acquire()
    process = None
    log_file = None
    try:
        job["status"] = "running"
        job["started_at"] = int(time.time())

        INGEST_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = INGEST_LOG_DIR / f"{job_id}.log"
        log_file = open(log_path, "ab")
        job["log_path"] = str(log_path)

        env = os.environ.copy()
        env["TOPIC_FILTER"] = store_name
        env["MAX_NEW_FILES_PER_TOPIC"] = env.get("MAX_NEW_FILES_PER_TOPIC", "20")
        # Manual UI-triggered ingest should avoid failing on model preload checks.
        env["PRECHECK_EMBEDDING"] = env.get("PRECHECK_EMBEDDING", "0")
        max_attempts = INGEST_WATCHDOG_RETRIES + 1
        for attempt in range(1, max_attempts + 1):
            job["attempt"] = attempt
            record_activity(
                "ingest_started",
                f"Background ingest started for {store_name} (attempt {attempt}/{max_attempts})",
                vector_store_id=vector_store_id,
                store_name=store_name,
                job_id=job_id,
                attempt=attempt,
                max_attempts=max_attempts,
            )

            process = await asyncio.create_subprocess_exec(
                "bash",
                str(INGEST_SCRIPT_PATH),
                cwd=str(PROJECT_DIR),
                env=env,
                stdout=log_file,
                stderr=asyncio.subprocess.STDOUT,
            )
            job["process_id"] = process.pid

            return_code = await process.wait()
            job["return_code"] = return_code
            process = None

            if return_code == 0:
                job["finished_at"] = int(time.time())
                job["status"] = "completed"
                record_activity(
                    "ingest_finished",
                    f"Background ingest completed for {store_name}",
                    vector_store_id=vector_store_id,
                    store_name=store_name,
                    job_id=job_id,
                    attempt=attempt,
                    return_code=return_code,
                )
                break

            if attempt < max_attempts:
                record_activity(
                    "ingest_retry_scheduled",
                    f"Background ingest failed for {store_name}; retrying in {INGEST_WATCHDOG_RETRY_DELAY_SECONDS}s",
                    vector_store_id=vector_store_id,
                    store_name=store_name,
                    job_id=job_id,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    return_code=return_code,
                )
                await asyncio.sleep(INGEST_WATCHDOG_RETRY_DELAY_SECONDS)
            else:
                job["finished_at"] = int(time.time())
                job["status"] = "failed"
                record_activity(
                    "ingest_finished",
                    f"Background ingest failed for {store_name}",
                    vector_store_id=vector_store_id,
                    store_name=store_name,
                    job_id=job_id,
                    attempt=attempt,
                    return_code=return_code,
                )
    except Exception as exc:
        job["status"] = "failed"
        job["finished_at"] = int(time.time())
        job["error"] = str(exc)
        record_activity(
            "ingest_failed",
            f"Background ingest failed for {store_name}",
            vector_store_id=vector_store_id,
            store_name=store_name,
            job_id=job_id,
            error=str(exc),
        )
    finally:
        if process and process.returncode is None:
            process.kill()
        if log_file is not None:
            log_file.close()
        ingest_job_semaphore.release()


async def require_ui_login(credentials: HTTPBasicCredentials = Depends(ui_security)):
    """Protect UI page with Basic auth credentials from settings."""
    username_ok = secrets.compare_digest(credentials.username, settings.ui_username)
    password_ok = secrets.compare_digest(credentials.password, settings.ui_password)
    if not (username_ok and password_ok):
        raise HTTPException(
            status_code=401,
            detail="Invalid UI username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


@app.on_event("startup")
async def startup():
    """Connect to database on startup"""
    await db.connect()
    await ensure_story_metadata_table()
    await ensure_ingest_file_stage_table()
    record_activity("system", "Server started")


@app.on_event("shutdown")
async def shutdown():
    """Disconnect from database on shutdown"""
    record_activity("system", "Server shutting down")
    await db.disconnect()


@app.get("/v1/activity")
async def list_activity(
    after_id: int = 0,
    limit: int = 100,
    api_key: str = Depends(get_api_key),
):
    safe_limit = min(max(limit, 1), 500)
    items = [item for item in activity_events if item["id"] > after_id]
    data = items[:safe_limit]
    last_id = data[-1]["id"] if data else after_id
    return {"object": "list", "data": data, "last_id": last_id}


@app.get("/v1/ingest/status")
async def list_ingest_status(
    vector_store_id: Optional[str] = None,
    status: Optional[str] = None,
    incomplete_only: bool = True,
    limit: int = 200,
    api_key: str = Depends(get_api_key),
):
    """Return latest tracked ingest stage per file, newest first."""
    safe_limit = min(max(limit, 1), 1000)
    where_clauses: list[str] = ["rn = 1"]
    params: list[Any] = []
    idx = 1

    if vector_store_id:
        where_clauses.append(f"vector_store_id = ${idx}")
        params.append(vector_store_id)
        idx += 1

    if status:
        where_clauses.append(f"status = ${idx}")
        params.append(status)
        idx += 1

    if incomplete_only:
        where_clauses.append("NOT (stage = 'completed' AND status = 'completed')")

    where_sql = " AND ".join(where_clauses)

    rows = await db.query_raw(
        f"""
        WITH latest AS (
            SELECT
                id,
                vector_store_id,
                folder,
                filename,
                stage,
                status,
                detail,
                error,
                created_at,
                EXTRACT(EPOCH FROM created_at)::bigint AS created_at_ts,
                ROW_NUMBER() OVER (
                    PARTITION BY vector_store_id, folder, filename
                    ORDER BY created_at DESC, id DESC
                ) AS rn
            FROM {INGEST_FILE_STAGE_TABLE}
        )
        SELECT
            id,
            vector_store_id,
            folder,
            filename,
            stage,
            status,
            detail,
            error,
            created_at_ts
        FROM latest
        WHERE {where_sql}
        ORDER BY created_at DESC, id DESC
        LIMIT {safe_limit}
        """,
        *params,
    )

    summary_rows = await db.query_raw(
        f"""
        WITH latest AS (
            SELECT
                vector_store_id,
                folder,
                filename,
                stage,
                status,
                ROW_NUMBER() OVER (
                    PARTITION BY vector_store_id, folder, filename
                    ORDER BY created_at DESC, id DESC
                ) AS rn
            FROM {INGEST_FILE_STAGE_TABLE}
        )
        SELECT status, COUNT(*)::bigint AS count
        FROM latest
        WHERE rn = 1
        GROUP BY status
        ORDER BY status ASC
        """
    )
    summary = {str(row["status"]): int(row["count"]) for row in summary_rows}

    return {
        "object": "list",
        "data": rows,
        "count": len(rows),
        "summary": summary,
        "incomplete_only": incomplete_only,
    }


async def generate_query_embedding(query: str) -> List[float]:
    """
    Generate an embedding for the query using LiteLLM
    """
    return await embedding_service.generate_embedding(query)


def to_unix_timestamp(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return int(value.timestamp())
    if isinstance(value, str):
        return int(datetime.fromisoformat(value).timestamp())
    return int(value)


def is_store_exposed(metadata: Optional[dict[str, Any]]) -> bool:
    """Opt-in flag: a vector store is only reachable via /v1/scenes/search
    (and the MCP server) once explicitly marked api_exposed=true. Absent or
    falsy values are treated as NOT exposed."""
    if not metadata:
        return False
    return bool(metadata.get("api_exposed", False))


def _search_content_keywords_sql(query: str, limit: int, target_ids: list[str]) -> tuple[str, list[Any]]:
    words = [word.lower() for word in re.findall(r"[a-z0-9]+", query.lower()) if len(word) >= 3]
    if not words:
        words = [query.lower().strip()]

    pattern = "|".join(re.escape(word) for word in words if word)
    if not pattern:
        pattern = re.escape(query.lower().strip())

    escaped_target_ids = ", ".join("'" + target_id.replace("'", "''") + "'" for target_id in target_ids)
    if not escaped_target_ids:
        escaped_target_ids = "''"

    vector_store_table = settings.table_names["vector_stores"]
    fields = settings.db_fields
    table_name = settings.table_names["embeddings"]
    sql = f"""
        SELECT
            e.{fields.id_field} as id,
            e.{fields.vector_store_id_field} as vector_store_id,
            e.{fields.content_field} as content,
            e.{fields.metadata_field} as metadata,
            0.0 as score,
            s.name as vector_store_name,
            sm.id as sm_id,
            sm.genres, sm.romance_subgenres, sm.tropes, sm.relationship_types,
            sm.character_archetypes, sm.occupations, sm.settings, sm.sports,
            sm.military, sm.kinks, sm.emotional_tone, sm.major_conflicts,
            sm.coming_out, sm.first_love, sm.forbidden_romance, sm.found_family,
            sm.slow_burn, sm.enemies_to_lovers, sm.hurt_comfort, sm.age_gap,
            sm.college, sm.military_romance, sm.heat_level, sm.content_warnings,
            sm.search_keywords, sm.pov, sm.tone, sm.explicitness,
            sm.relationship_structure, sm.ending
        FROM {table_name} e
        JOIN {vector_store_table} s ON s.id = e.{fields.vector_store_id_field}
        LEFT JOIN {STORY_METADATA_TABLE} sm
            ON sm.vector_store_id = e.{fields.vector_store_id_field}
           AND sm.filename = e.{fields.metadata_field}->>'filename'
        WHERE e.{fields.vector_store_id_field} IN ({escaped_target_ids})
          AND (
                e.{fields.content_field} ILIKE $1
             OR e.{fields.metadata_field}::text ILIKE $1
             OR sm.search_keywords::text ILIKE $1
          )
        ORDER BY s.name ASC, e.{fields.id_field} ASC
        LIMIT $2
    """
    return sql, [f"%{pattern}%", limit]


async def _safe_query_embedding(query: str) -> Optional[list[float]]:
    try:
        timeout_seconds = max(0.5, float(settings.embedding.query_timeout_seconds))
        return await asyncio.wait_for(generate_query_embedding(query), timeout=timeout_seconds)
    except Exception:
        return None


# ── Structured story metadata (genres, tropes, kinks, heat level, etc.) ─────
STORY_METADATA_TABLE = "story_metadata"
INGEST_FILE_STAGE_TABLE = "ingest_file_stages"

STORY_METADATA_ARRAY_FIELDS = [
    "genres",
    "romance_subgenres",
    "tropes",
    "relationship_types",
    "character_archetypes",
    "occupations",
    "settings",
    "sports",
    "military",
    "kinks",
    "emotional_tone",
    "major_conflicts",
    "content_warnings",
    "search_keywords",
    "tone",
]
STORY_METADATA_BOOL_FIELDS = [
    "coming_out",
    "first_love",
    "forbidden_romance",
    "found_family",
    "slow_burn",
    "enemies_to_lovers",
    "hurt_comfort",
    "age_gap",
    "college",
    "military_romance",
]
# pov, explicitness, relationship_structure, ending
STORY_METADATA_STRING_FIELDS = ["pov", "explicitness", "relationship_structure", "ending"]


async def ensure_story_metadata_table() -> None:
    """Idempotently create the story_metadata table (structured tags shared
    by every chunk of a given vector_store_id + filename)."""
    array_columns = ",\n            ".join(
        f"{field} JSONB NOT NULL DEFAULT '[]'::jsonb" for field in STORY_METADATA_ARRAY_FIELDS
    )
    bool_columns = ",\n            ".join(
        f"{field} BOOLEAN NOT NULL DEFAULT false" for field in STORY_METADATA_BOOL_FIELDS
    )
    string_columns = ",\n            ".join(f"{field} TEXT" for field in STORY_METADATA_STRING_FIELDS)
    vector_store_table = settings.table_names["vector_stores"]

    await db.execute_raw(
        f"""
        CREATE TABLE IF NOT EXISTS {STORY_METADATA_TABLE} (
            id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
            vector_store_id TEXT NOT NULL REFERENCES {vector_store_table}(id) ON DELETE CASCADE,
            filename TEXT NOT NULL,
            {array_columns},
            {bool_columns},
            heat_level SMALLINT NOT NULL DEFAULT 1 CHECK (heat_level BETWEEN 1 AND 5),
            {string_columns},
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (vector_store_id, filename)
        )
        """
    )
    await db.execute_raw(
        f"CREATE INDEX IF NOT EXISTS story_metadata_vector_store_idx "
        f"ON {STORY_METADATA_TABLE} (vector_store_id)"
    )


async def ensure_ingest_file_stage_table() -> None:
    """Create per-file ingest stage logs for audit and resume workflows."""
    vector_store_table = settings.table_names["vector_stores"]
    await db.execute_raw(
        f"""
        CREATE TABLE IF NOT EXISTS {INGEST_FILE_STAGE_TABLE} (
            id BIGSERIAL PRIMARY KEY,
            vector_store_id TEXT NOT NULL REFERENCES {vector_store_table}(id) ON DELETE CASCADE,
            folder TEXT NOT NULL,
            filename TEXT NOT NULL,
            stage TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'in_progress',
            detail JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            error TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    await db.execute_raw(
        f"CREATE INDEX IF NOT EXISTS ingest_file_stage_lookup_idx "
        f"ON {INGEST_FILE_STAGE_TABLE} (vector_store_id, folder, filename, created_at DESC)"
    )


async def record_ingest_file_stage(
    vector_store_id: str,
    folder: str,
    filename: str,
    stage: str,
    *,
    status: str = "in_progress",
    detail: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    if not filename:
        return
    await db.execute_raw(
        f"""
        INSERT INTO {INGEST_FILE_STAGE_TABLE}
            (vector_store_id, folder, filename, stage, status, detail, error, created_at)
        VALUES
            ($1, $2, $3, $4, $5, $6::jsonb, $7, NOW())
        """,
        vector_store_id,
        folder,
        filename,
        stage,
        status,
        json.dumps(detail or {}),
        error,
    )


def story_metadata_from_row(row: dict[str, Any]) -> StoryMetadata:
    kwargs: dict[str, Any] = {}
    for field in STORY_METADATA_ARRAY_FIELDS:
        kwargs[field] = row.get(field) or []
    for field in STORY_METADATA_BOOL_FIELDS:
        kwargs[field] = bool(row.get(field))
    for field in STORY_METADATA_STRING_FIELDS:
        kwargs[field] = row.get(field)
    kwargs["heat_level"] = row.get("heat_level") or 1
    return StoryMetadata(**kwargs)


def story_metadata_response_from_row(row: dict[str, Any]) -> StoryMetadataResponse:
    base = story_metadata_from_row(row)
    return StoryMetadataResponse(
        **base.model_dump(),
        vector_store_id=row["vector_store_id"],
        filename=row["filename"],
        created_at=int(row["created_at_ts"]),
        updated_at=int(row["updated_at_ts"]),
    )


async def upsert_story_metadata(vector_store_id: str, filename: str, metadata: StoryMetadata) -> dict[str, Any]:
    columns = (
        STORY_METADATA_ARRAY_FIELDS
        + STORY_METADATA_BOOL_FIELDS
        + ["heat_level"]
        + STORY_METADATA_STRING_FIELDS
    )
    values: list[Any] = [vector_store_id, filename]
    placeholders = ["$1", "$2"]
    idx = 3

    for field in STORY_METADATA_ARRAY_FIELDS:
        placeholders.append(f"${idx}::jsonb")
        values.append(json.dumps(getattr(metadata, field)))
        idx += 1
    for field in STORY_METADATA_BOOL_FIELDS:
        placeholders.append(f"${idx}")
        values.append(getattr(metadata, field))
        idx += 1
    placeholders.append(f"${idx}")
    values.append(metadata.heat_level)
    idx += 1
    for field in STORY_METADATA_STRING_FIELDS:
        placeholders.append(f"${idx}")
        values.append(getattr(metadata, field))
        idx += 1

    update_clauses = ", ".join(f"{col} = EXCLUDED.{col}" for col in columns) + ", updated_at = NOW()"

    result = await db.query_raw(
        f"""
        INSERT INTO {STORY_METADATA_TABLE} (id, vector_store_id, filename, {", ".join(columns)}, created_at, updated_at)
        VALUES (gen_random_uuid()::text, {", ".join(placeholders)}, NOW(), NOW())
        ON CONFLICT (vector_store_id, filename)
        DO UPDATE SET {update_clauses}
        RETURNING *, EXTRACT(EPOCH FROM created_at)::bigint as created_at_ts,
                 EXTRACT(EPOCH FROM updated_at)::bigint as updated_at_ts
        """,
        *values,
    )
    if not result:
        raise HTTPException(status_code=500, detail="Failed to upsert story metadata")
    return result[0]


async def upsert_default_story_metadata_rows(vector_store_id: str, filenames: list[str]) -> int:
    """Ensure each uploaded filename has a structured metadata row.

    Upload routes always write chunk-level embedding metadata, but callers also
    expect per-story rows in story_metadata to exist immediately after upload.
    """
    unique_filenames = sorted({name for name in filenames if name})
    for filename in unique_filenames:
        await upsert_story_metadata(vector_store_id, filename, StoryMetadata())
    return len(unique_filenames)


async def upsert_story_metadata_rows(vector_store_id: str, items: dict[str, StoryMetadata]) -> int:
    """Upsert one structured metadata payload per filename for a vector store."""
    count = 0
    for filename in sorted(items.keys()):
        if not filename:
            continue
        await upsert_story_metadata(vector_store_id, filename, items[filename])
        count += 1
    return count


def is_default_story_metadata(metadata: StoryMetadata) -> bool:
    return metadata.model_dump() == StoryMetadata().model_dump()


def infer_story_metadata_from_text(filename: str, text: str, vector_store_name: str | None = None) -> StoryMetadata:
    """Heuristic fallback when LLM extraction is unavailable.

    Produces lightweight but useful structured tags from filename/store context
    and a small text sample so story_metadata isn't left empty.
    """
    store_hint = (vector_store_name or "").replace("-", " ").strip().lower()
    filename_hint = Path(filename).stem.replace("-", " ").replace("_", " ").lower()
    sample = text[:20000].lower()
    scan = f"{store_hint}\n{filename_hint}\n{sample}"

    def has_any(terms: list[str]) -> bool:
        return any(term in scan for term in terms)

    genres: list[str] = []
    settings_tags: list[str] = []
    military_tags: list[str] = []
    relationship_types: list[str] = []
    search_keywords: list[str] = []

    if has_any(["romance", "love", "boyfriend", "relationship", "fall in love"]):
        genres.append("romance")

    if has_any(["gay", "m/m", "male male", "two men", "boyfriend", "husband"]):
        relationship_types.append("male/male")

    if has_any(["military", "army", "navy", "marine", "air force", "soldier", "barracks"]):
        settings_tags.append("military")

    branch_terms = {
        "army": ["army", "soldier", "infantry", "barracks"],
        "navy": ["navy", "sailor", "ship", "submarine"],
        "air force": ["air force", "pilot", "airman", "flight line"],
        "marines": ["marine", "marines", "corps"],
    }
    for label, terms in branch_terms.items():
        if has_any(terms):
            military_tags.append(label)

    explicit_hits = sum(
        1
        for token in ["sex", "cock", "cum", "blowjob", "fuck", "orgasm", "hardon", "anal"]
        if token in scan
    )
    if explicit_hits >= 5:
        heat_level = 5
        explicitness = "very explicit"
    elif explicit_hits >= 3:
        heat_level = 4
        explicitness = "explicit"
    elif explicit_hits >= 1:
        heat_level = 3
        explicitness = "explicit"
    else:
        heat_level = 2 if "kiss" in scan else 1
        explicitness = "kissing" if "kiss" in scan else None

    if store_hint:
        search_keywords.append(store_hint)
    if genres and settings_tags:
        search_keywords.append(f"{settings_tags[0]} {genres[0]}")
    if military_tags:
        search_keywords.extend(f"{branch} romance" for branch in military_tags[:2])
    if filename_hint:
        search_keywords.append(filename_hint)

    return StoryMetadata(
        genres=sorted(set(genres)),
        relationship_types=sorted(set(relationship_types)),
        settings=sorted(set(settings_tags)),
        military=sorted(set(military_tags)),
        military_romance=("romance" in genres and "military" in settings_tags),
        heat_level=heat_level,
        explicitness=explicitness,
        search_keywords=sorted(set(search_keywords)),
    )


def prepare_story_text_for_metadata(text: str) -> tuple[str, int]:
    """Clamp metadata extraction input to stay within model context limits."""
    if len(text) <= STORY_METADATA_MAX_CHARS:
        return text, 0
    return text[:STORY_METADATA_MAX_CHARS], len(text) - STORY_METADATA_MAX_CHARS


async def extract_story_metadata_for_upload(
    filename: str,
    text: str,
    vector_store_name: str | None = None,
) -> StoryMetadata:
    """Best-effort metadata extraction for upload flows.

    Falls back to default StoryMetadata on extraction failures so uploads can
    continue even when the metadata model is unavailable.
    """
    story_text, dropped_chars = prepare_story_text_for_metadata(text)
    if dropped_chars > 0:
        record_activity(
            "story_metadata_text_truncated",
            f"Metadata input truncated for {filename}",
            filename=filename,
            used_chars=len(story_text),
            dropped_chars=dropped_chars,
        )
    try:
        extracted = await extract_with_retry(
            filename=filename,
            story_text=story_text,
            model=settings.story_metadata_llm_model,
            api_base=settings.story_metadata_llm_api_base or settings.embedding.base_url,
            api_key=settings.story_metadata_llm_api_key or settings.embedding.api_key,
            retries=0,
        )
        if is_default_story_metadata(extracted):
            return infer_story_metadata_from_text(filename, text, vector_store_name)
        return extracted
    except Exception as exc:
        record_activity(
            "story_metadata_extraction_failed",
            f"Metadata extraction failed for {filename}; using heuristic fallback",
            filename=filename,
            error=str(exc),
        )
        return infer_story_metadata_from_text(filename, text, vector_store_name)


def build_tag_filter_sql(tag_filters: dict[str, Any], start_index: int) -> tuple[str, list[Any], int]:
    """Build a whitelisted SQL WHERE-clause fragment (ANDed conditions) for
    structured story-metadata filters against the `sm` alias. Raises
    HTTPException(400) on unrecognized keys so typos surface immediately."""
    clauses: list[str] = []
    params: list[Any] = []
    idx = start_index

    for key, value in tag_filters.items():
        if key in STORY_METADATA_ARRAY_FIELDS:
            if not isinstance(value, list) or not value:
                raise HTTPException(status_code=400, detail=f"tag_filters.{key} must be a non-empty list")
            clauses.append(f"sm.{key} ?| ${idx}::text[]")
            params.append([str(v) for v in value])
            idx += 1
        elif key in STORY_METADATA_BOOL_FIELDS:
            clauses.append(f"sm.{key} = ${idx}")
            params.append(bool(value))
            idx += 1
        elif key in STORY_METADATA_STRING_FIELDS:
            clauses.append(f"sm.{key} = ${idx}")
            params.append(str(value))
            idx += 1
        elif key == "heat_level_min":
            clauses.append(f"sm.heat_level >= ${idx}")
            params.append(int(value))
            idx += 1
        elif key == "heat_level_max":
            clauses.append(f"sm.heat_level <= ${idx}")
            params.append(int(value))
            idx += 1
        elif key == "any_keyword":
            if not isinstance(value, list) or not value:
                raise HTTPException(status_code=400, detail="tag_filters.any_keyword must be a non-empty list")
            or_clause = " OR ".join(f"sm.{field} ?| ${idx}::text[]" for field in STORY_METADATA_ARRAY_FIELDS)
            clauses.append(f"({or_clause})")
            params.append([str(v) for v in value])
            idx += 1
        else:
            raise HTTPException(status_code=400, detail=f"Unknown tag_filters key: {key}")

    return " AND ".join(clauses), params, idx


def chunk_text(text: str, max_chars: int = 1600, overlap: int = 200) -> list[str]:
    normalized_text = text.replace("\r\n", "\n").strip()
    if not normalized_text:
        return []

    cleaned_overlap = max(0, min(overlap, max_chars // 2))
    paragraphs = [part.strip() for part in normalized_text.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs or [normalized_text]:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""

            start = 0
            step = max_chars - cleaned_overlap or max_chars
            while start < len(paragraph):
                piece = paragraph[start:start + max_chars].strip()
                if piece:
                    chunks.append(piece)
                start += step
            continue

        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    return chunks


def canonicalize_vector_store_name(name: str) -> str:
    """Collapse whitespace and trim so equivalent names map consistently."""
    return " ".join((name or "").split())


def normalize_vector_store_name(name: str) -> str:
    return canonicalize_vector_store_name(name).lower()


async def create_vector_store_record(
    name: str,
    expires_after: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    canonical_name = canonicalize_vector_store_name(name)
    if not canonical_name:
        raise HTTPException(status_code=400, detail="Vector store name is required")

    normalized_name = normalize_vector_store_name(canonical_name)
    vector_store_table = settings.table_names["vector_stores"]
    result = await db.query_raw(
        f"""
        WITH lock_row AS (
            SELECT pg_advisory_lock(hashtext($1))
        ),
        existing AS (
            SELECT
                id,
                name,
                file_counts,
                status,
                usage_bytes,
                expires_after,
                expires_at,
                last_active_at,
                metadata,
                created_at
            FROM {vector_store_table}
            WHERE lower(regexp_replace(btrim(name), '\\s+', ' ', 'g')) = $1
            ORDER BY created_at ASC
            LIMIT 1
        ),
        inserted AS (
            INSERT INTO {vector_store_table} (id, name, file_counts, status, usage_bytes, expires_after, metadata, created_at)
            SELECT gen_random_uuid()::text, $2, $3, $4, $5, $6, $7, NOW()
            FROM lock_row
            WHERE NOT EXISTS (SELECT 1 FROM existing)
            RETURNING
                id,
                name,
                file_counts,
                status,
                usage_bytes,
                expires_after,
                expires_at,
                last_active_at,
                metadata,
                created_at
        ),
        chosen AS (
            SELECT
                id,
                name,
                file_counts,
                status,
                usage_bytes,
                expires_after,
                expires_at,
                last_active_at,
                metadata,
                created_at,
                FALSE AS created_new
            FROM existing
            UNION ALL
            SELECT
                id,
                name,
                file_counts,
                status,
                usage_bytes,
                expires_after,
                expires_at,
                last_active_at,
                metadata,
                created_at,
                TRUE AS created_new
            FROM inserted
        ),
        unlock_row AS (
            SELECT pg_advisory_unlock(hashtext($1))
            FROM lock_row
        )
        SELECT
            chosen.id,
            chosen.name,
            chosen.file_counts,
            chosen.status,
            chosen.usage_bytes,
            chosen.expires_after,
            chosen.expires_at,
            chosen.last_active_at,
            chosen.metadata,
            chosen.created_new,
            EXTRACT(EPOCH FROM chosen.created_at)::bigint as created_at_timestamp
        FROM chosen
        CROSS JOIN unlock_row
        LIMIT 1
        """,
        normalized_name,
        canonical_name,
        {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
        "completed",
        0,
        expires_after,
        metadata or {},
    )
    if not result:
        raise HTTPException(status_code=500, detail="Failed to create vector store")
    return result[0]


async def get_vector_store_id_or_create(
    vector_store_id: Optional[str],
    vector_store_name: Optional[str],
) -> tuple[str, str, bool]:
    vector_store_table = settings.table_names["vector_stores"]
    if vector_store_id:
        result = await db.query_raw(
            f"SELECT id, name FROM {vector_store_table} WHERE id = $1",
            vector_store_id,
        )
        if not result:
            raise HTTPException(status_code=404, detail="Vector store not found")
        return str(vector_store_id), result[0]["name"], False

    cleaned_name = canonicalize_vector_store_name(vector_store_name or "")
    if not cleaned_name:
        raise HTTPException(
            status_code=400,
            detail="Provide either vector_store_id or vector_store_name",
        )

    vector_store = await create_vector_store_record(
        cleaned_name,
        metadata={"created_via": "upload_ui"},
    )
    return str(vector_store["id"]), vector_store["name"], bool(vector_store.get("created_new", False))


def _sanitize_path_segment(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return cleaned or "untitled"


def backup_uploaded_text_file(store_name: str, filename: str, raw_bytes: bytes) -> Path:
    backup_root = Path(settings.backup_root_dir)
    store_dir = backup_root / _sanitize_path_segment(store_name)
    store_dir.mkdir(parents=True, exist_ok=True)

    source_name = Path(filename).name
    stem = _sanitize_path_segment(Path(source_name).stem)
    suffix = Path(source_name).suffix or ".txt"

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    backup_path = store_dir / f"{timestamp}_{stem}{suffix}"
    counter = 1
    while backup_path.exists():
        backup_path = store_dir / f"{timestamp}_{stem}_{counter}{suffix}"
        counter += 1

    backup_path.write_bytes(raw_bytes)
    return backup_path


async def insert_embeddings_records(
    vector_store_id: str,
    embeddings: list[EmbeddingCreateRequest],
) -> list[EmbeddingResponse]:
    if not embeddings:
        raise HTTPException(status_code=400, detail="No embeddings provided")

    vector_store_table = settings.table_names["vector_stores"]
    fields = settings.db_fields
    table_name = settings.table_names["embeddings"]

    # PostgreSQL prepared statements cap bind params at 32767.
    # Each row consumes 4 params in this INSERT, so we batch safely below the limit.
    max_bind_params = 30000
    params_per_row = 4
    rows_per_batch = max(1, max_bind_params // params_per_row)

    all_rows: list[dict[str, Any]] = []

    for batch_start in range(0, len(embeddings), rows_per_batch):
        batch_embeddings = embeddings[batch_start:batch_start + rows_per_batch]

        values_clauses = []
        params: list[Any] = []
        param_count = 1

        for embedding_req in batch_embeddings:
            embedding_vector_str = "[" + ",".join(map(str, embedding_req.embedding)) + "]"
            values_clauses.append(
                f"(gen_random_uuid(), ${param_count}, ${param_count + 1}, ${param_count + 2}::halfvec, ${param_count + 3}, NOW())"
            )
            params.extend([
                vector_store_id,
                embedding_req.content,
                embedding_vector_str,
                embedding_req.metadata or {},
            ])
            param_count += 4

        values_clause = ", ".join(values_clauses)
        batch_result = await db.query_raw(
            f"""
            INSERT INTO {table_name} ({fields.id_field}, {fields.vector_store_id_field}, {fields.content_field},
                                     {fields.embedding_field}, {fields.metadata_field}, {fields.created_at_field})
            VALUES {values_clause}
            RETURNING {fields.id_field}, {fields.vector_store_id_field}, {fields.content_field},
                     {fields.metadata_field}, EXTRACT(EPOCH FROM {fields.created_at_field})::bigint as created_at_timestamp
            """,
            *params,
        )

        if not batch_result:
            raise HTTPException(status_code=500, detail="Failed to create embeddings")

        all_rows.extend(batch_result)

    total_content_length = sum(len(embedding.content) for embedding in embeddings)
    await db.query_raw(
        f"""
        UPDATE {vector_store_table}
        SET
            file_counts = jsonb_set(
                jsonb_set(
                    COALESCE(file_counts, '{{"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0}}'::jsonb),
                    '{{completed}}',
                    (COALESCE(file_counts->>'completed', '0')::int + $2)::text::jsonb
                ),
                '{{total}}',
                (COALESCE(file_counts->>'total', '0')::int + $2)::text::jsonb
            ),
            usage_bytes = COALESCE(usage_bytes, 0) + $3,
            last_active_at = NOW()
        WHERE id = $1
        """,
        vector_store_id,
        len(embeddings),
        total_content_length,
    )

    return [
        EmbeddingResponse(
            id=row[fields.id_field],
            vector_store_id=row[fields.vector_store_id_field],
            content=row[fields.content_field],
            metadata=row[fields.metadata_field],
            created_at=int(row["created_at_timestamp"]),
        )
        for row in all_rows
    ]


@app.get("/", response_class=FileResponse, dependencies=[Depends(require_ui_login)])
@app.get("/ui", response_class=FileResponse, dependencies=[Depends(require_ui_login)])
async def upload_ui():
    return FileResponse(UPLOAD_PAGE_PATH)


@app.post("/v1/vector_stores", response_model=VectorStoreResponse)
async def create_vector_store(
    request: VectorStoreCreateRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Create a new vector store.
    """
    try:
        # Use raw SQL to insert the vector store with configurable table/field names
        vector_store = await create_vector_store_record(
            request.name,
            expires_after=request.expires_after,
            metadata=request.metadata,
        )
        
        # Convert to response format
        created_at = int(vector_store["created_at_timestamp"])
        expires_at = to_unix_timestamp(vector_store.get("expires_at"))
        last_active_at = to_unix_timestamp(vector_store.get("last_active_at"))
        
        return VectorStoreResponse(
            id=vector_store["id"],
            created_at=created_at,
            name=vector_store["name"],
            usage_bytes=vector_store["usage_bytes"] or 0,
            file_counts=vector_store["file_counts"] or {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
            status=vector_store["status"],
            expires_after=vector_store["expires_after"],
            expires_at=expires_at,
            last_active_at=last_active_at,
            metadata=vector_store["metadata"],
            api_exposed=is_store_exposed(vector_store["metadata"]),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create vector store: {str(e)}")


@app.get("/v1/vector_stores", response_model=VectorStoreListResponse)
async def list_vector_stores(
    limit: Optional[int] = 20,
    after: Optional[str] = None,
    before: Optional[str] = None,
    api_key: str = Depends(get_api_key)
):
    """
    List vector stores with optional pagination.
    """
    try:
        limit = min(limit or 20, 100)  # Cap at 100 results
        
        vector_store_table = settings.table_names["vector_stores"]
        
        # Build base query
        base_query = f"""
        SELECT id, name, file_counts, status, usage_bytes, expires_after, expires_at, last_active_at, metadata,
               EXTRACT(EPOCH FROM created_at)::bigint as created_at_timestamp
        FROM {vector_store_table}
        """
        
        # Add pagination conditions
        conditions = []
        params = []
        param_count = 1
        
        if after:
            conditions.append(f"id > ${param_count}")
            params.append(after)
            param_count += 1
            
        if before:
            conditions.append(f"id < ${param_count}")
            params.append(before)
            param_count += 1
        
        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)
        
        # Add ordering and limit
        final_query = base_query + f" ORDER BY created_at DESC LIMIT {limit + 1}"
        
        # Execute query
        results = await db.query_raw(final_query, *params)
        
        # Check if there are more results
        has_more = len(results) > limit
        if has_more:
            results = results[:limit]  # Remove extra result
        
        # Convert to response format
        vector_stores = []
        for row in results:
            created_at = int(row["created_at_timestamp"])
            expires_at = to_unix_timestamp(row.get("expires_at"))
            last_active_at = to_unix_timestamp(row.get("last_active_at"))
            
            vector_store = VectorStoreResponse(
                id=row["id"],
                created_at=created_at,
                name=row["name"],
                usage_bytes=row["usage_bytes"] or 0,
                file_counts=row["file_counts"] or {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
                status=row["status"],
                expires_after=row["expires_after"],
                expires_at=expires_at,
                last_active_at=last_active_at,
                metadata=row["metadata"],
                api_exposed=is_store_exposed(row["metadata"]),
            )
            vector_stores.append(vector_store)
        
        # Determine first_id and last_id
        first_id = vector_stores[0].id if vector_stores else None
        last_id = vector_stores[-1].id if vector_stores else None
        
        return VectorStoreListResponse(
            data=vector_stores,
            first_id=first_id,
            last_id=last_id,
            has_more=has_more
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to list vector stores: {str(e)}")


@app.patch("/v1/vector_stores/{vector_store_id}/exposure", response_model=VectorStoreResponse)
async def update_vector_store_exposure(
    vector_store_id: str,
    request: VectorStoreExposureUpdateRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Enable or disable a vector store for the public /v1/scenes/search API and
    the MCP server. Stores are opt-in: this must be explicitly set to true
    before a store is reachable through that surface.
    """
    vector_store_table = settings.table_names["vector_stores"]
    try:
        existing = await db.query_raw(
            f"SELECT id, name, metadata FROM {vector_store_table} WHERE id = $1",
            vector_store_id,
        )
        if not existing:
            raise HTTPException(status_code=404, detail="Vector store not found")

        merged_metadata = dict(existing[0]["metadata"] or {})
        merged_metadata["api_exposed"] = request.api_exposed

        result = await db.query_raw(
            f"""
            UPDATE {vector_store_table}
            SET metadata = $2::jsonb, last_active_at = NOW()
            WHERE id = $1
            RETURNING id, name, file_counts, status, usage_bytes, expires_after, expires_at, last_active_at, metadata,
                     EXTRACT(EPOCH FROM created_at)::bigint as created_at_timestamp
            """,
            vector_store_id,
            json.dumps(merged_metadata),
        )
        if not result:
            raise HTTPException(status_code=500, detail="Failed to update vector store exposure")

        row = result[0]
        record_activity(
            "exposure_updated",
            f"Store '{row['name']}' api_exposed set to {request.api_exposed}",
            vector_store_id=row["id"],
            store_name=row["name"],
            api_exposed=request.api_exposed,
        )

        return VectorStoreResponse(
            id=row["id"],
            created_at=int(row["created_at_timestamp"]),
            name=row["name"],
            usage_bytes=row["usage_bytes"] or 0,
            file_counts=row["file_counts"] or {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
            status=row["status"],
            expires_after=row["expires_after"],
            expires_at=to_unix_timestamp(row.get("expires_at")),
            last_active_at=to_unix_timestamp(row.get("last_active_at")),
            metadata=row["metadata"],
            api_exposed=is_store_exposed(row["metadata"]),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update vector store exposure: {str(e)}")


@app.get("/v1/vector_stores/{vector_store_id}/stories", response_model=List[StoryFileInfo])
async def list_story_files(
    vector_store_id: str,
    api_key: str = Depends(get_api_key),
):
    """
    List distinct story files uploaded to a vector store, with chunk counts
    and (if present) their structured metadata tags.
    """
    fields = settings.db_fields
    table_name = settings.table_names["embeddings"]

    file_rows = await db.query_raw(
        f"""
        SELECT {fields.metadata_field}->>'filename' as filename, COUNT(*) as chunk_count
        FROM {table_name}
        WHERE {fields.vector_store_id_field} = $1
        GROUP BY {fields.metadata_field}->>'filename'
        ORDER BY filename
        """,
        vector_store_id,
    )

    meta_rows = await db.query_raw(
        f"SELECT * FROM {STORY_METADATA_TABLE} WHERE vector_store_id = $1",
        vector_store_id,
    )
    meta_by_filename = {row["filename"]: row for row in meta_rows}

    return [
        StoryFileInfo(
            filename=row["filename"] or "document.txt",
            chunk_count=int(row["chunk_count"]),
            has_metadata=(row["filename"] in meta_by_filename),
            metadata=(
                story_metadata_from_row(meta_by_filename[row["filename"]])
                if row["filename"] in meta_by_filename
                else None
            ),
        )
        for row in file_rows
    ]


@app.post("/v1/vector_stores/{vector_store_id}/ingest-background")
async def launch_vector_store_ingest(
    vector_store_id: str,
    api_key: str = Depends(get_api_key),
):
    """Launch background ingest for one vector store topic."""
    vector_store_table = settings.table_names["vector_stores"]
    existing = await db.query_raw(
        f"SELECT id, name FROM {vector_store_table} WHERE id = $1",
        vector_store_id,
    )
    if not existing:
        raise HTTPException(status_code=404, detail="Vector store not found")

    store = existing[0]
    store_name = store["name"]

    for job in vector_store_ingest_jobs.values():
        if (
            job.get("vector_store_id") == vector_store_id
            and job.get("status") in {"queued", "running"}
        ):
            return {
                "started": False,
                "reason": "active_job_exists",
                "job": job,
            }

    if not INGEST_SCRIPT_PATH.is_file():
        raise HTTPException(status_code=500, detail=f"Ingest script not found: {INGEST_SCRIPT_PATH}")

    job_id = _new_ingest_job_id()
    job = {
        "job_id": job_id,
        "vector_store_id": vector_store_id,
        "store_name": store_name,
        "topic": store_name,
        "status": "queued",
        "process_id": None,
        "log_path": str(INGEST_LOG_DIR / f"{job_id}.log"),
        "queued_at": int(time.time()),
        "started_at": None,
        "finished_at": None,
        "return_code": None,
    }
    vector_store_ingest_jobs[job_id] = job

    asyncio.create_task(_run_ingest_job(job_id, vector_store_id, store_name))

    record_activity(
        "ingest_queued",
        f"Background ingest queued for {store_name}",
        vector_store_id=vector_store_id,
        store_name=store_name,
        job_id=job_id,
    )
    return {"started": True, "job": job}


@app.get("/v1/vector_stores/{vector_store_id}/ingest-background/{job_id}")
async def get_vector_store_ingest_status(
    vector_store_id: str,
    job_id: str,
    api_key: str = Depends(get_api_key),
):
    """Get status for a previously launched background ingest job."""
    job = vector_store_ingest_jobs.get(job_id)
    if not job or job.get("vector_store_id") != vector_store_id:
        raise HTTPException(status_code=404, detail="Ingest job not found")
    return job


@app.get("/v1/vector_stores/{vector_store_id}/stories/{filename}/metadata", response_model=StoryMetadataResponse)
async def get_story_metadata(
    vector_store_id: str,
    filename: str,
    api_key: str = Depends(get_api_key),
):
    """Fetch the structured tags for a single story."""
    result = await db.query_raw(
        f"""
        SELECT *, EXTRACT(EPOCH FROM created_at)::bigint as created_at_ts,
               EXTRACT(EPOCH FROM updated_at)::bigint as updated_at_ts
        FROM {STORY_METADATA_TABLE}
        WHERE vector_store_id = $1 AND filename = $2
        """,
        vector_store_id,
        filename,
    )
    if not result:
        raise HTTPException(status_code=404, detail="No structured metadata for this story yet")
    return story_metadata_response_from_row(result[0])


@app.put("/v1/vector_stores/{vector_store_id}/stories/{filename}/metadata", response_model=StoryMetadataResponse)
async def set_story_metadata(
    vector_store_id: str,
    filename: str,
    request: StoryMetadata,
    api_key: str = Depends(get_api_key),
):
    """Create or replace the structured tags (genres, tropes, kinks, heat
    level, POV, tone, explicitness, relationship structure, ending, etc.)
    for a single story."""
    vector_store_table = settings.table_names["vector_stores"]
    existing = await db.query_raw(f"SELECT id FROM {vector_store_table} WHERE id = $1", vector_store_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Vector store not found")

    row = await upsert_story_metadata(vector_store_id, filename, request)
    return story_metadata_response_from_row(row)


@app.post("/v1/vector_stores/{vector_store_id}/stories/metadata/bulk")
async def bulk_set_story_metadata(
    vector_store_id: str,
    request: StoryMetadataBulkUpsertRequest,
    api_key: str = Depends(get_api_key),
):
    """Bulk create/replace structured tags for many stories at once, e.g.
    output from an offline tagging pass. Body: {"items": {"<filename>": {...tags...}}}."""
    vector_store_table = settings.table_names["vector_stores"]
    existing = await db.query_raw(f"SELECT id FROM {vector_store_table} WHERE id = $1", vector_store_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Vector store not found")

    updated_files = []
    for filename, metadata in request.items.items():
        await upsert_story_metadata(vector_store_id, filename, metadata)
        updated_files.append(filename)

    return {"vector_store_id": vector_store_id, "updated_files": updated_files, "count": len(updated_files)}


@app.post("/v1/stories/metadata/extract", response_model=StoryMetadata)
async def extract_story_metadata_for_ui(
    request: StoryMetadataExtractionRequest,
    api_key: str = Depends(get_api_key),
):
    """Extract structured story metadata using server-side LiteLLM settings."""
    story_text, _ = prepare_story_text_for_metadata(request.text)
    try:
        return await extract_with_retry(
            filename=request.filename,
            story_text=story_text,
            model=settings.story_metadata_llm_model,
            api_base=settings.story_metadata_llm_api_base or settings.embedding.base_url,
            api_key=settings.story_metadata_llm_api_key or settings.embedding.api_key,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Metadata extraction failed: {str(exc)}")


@app.post("/v1/scenes/search", response_model=SceneSearchResponse)
async def search_scenes(
    request: SceneSearchRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Cross-store natural-language scene search, e.g. "a scene with a military
    guy in it". Only searches vector stores explicitly opted in via
    api_exposed=true (see PATCH /v1/vector_stores/{id}/exposure). This is the
    intended entry point for other applications and the MCP server.
    """
    try:
        vector_store_table = settings.table_names["vector_stores"]
        exposed_rows = await db.query_raw(
            f"SELECT id, name, metadata FROM {vector_store_table}"
        )
        exposed_ids_by_id = {
            row["id"]: row["name"]
            for row in exposed_rows
            if is_store_exposed(row["metadata"])
        }

        if request.vector_store_ids:
            requested = set(request.vector_store_ids)
            target_ids = [store_id for store_id in requested if store_id in exposed_ids_by_id]
        else:
            target_ids = list(exposed_ids_by_id.keys())

        if not target_ids:
            return SceneSearchResponse(search_query=request.query, data=[], has_more=False)

        limit = min(request.limit or 10, 100)
        fields = settings.db_fields
        table_name = settings.table_names["embeddings"]

        keyword_sql, keyword_params = _search_content_keywords_sql(request.query, limit, target_ids)
        keyword_results = await db.query_raw(keyword_sql, *keyword_params)
        if keyword_results:
            search_results = []
            for row in keyword_results:
                metadata = row["metadata"] or {}
                filename = metadata.get("filename", "document.txt")
                store_id = row["vector_store_id"]
                search_results.append(
                    SceneSearchResult(
                        vector_store_id=store_id,
                        vector_store_name=exposed_ids_by_id.get(store_id, "unknown"),
                        file_id=row["id"],
                        filename=filename,
                        score=0.0,
                        attributes=metadata if request.return_metadata else None,
                        content=[ContentChunk(type="text", text=row["content"])],
                        story_metadata=story_metadata_from_row(row) if row.get("sm_id") else None,
                    )
                )
            return SceneSearchResponse(search_query=request.query, data=search_results, has_more=False)

        query_embedding = await _safe_query_embedding(request.query)
        if query_embedding is not None:
            query_vector_str = "[" + ",".join(map(str, query_embedding)) + "]"

            query_params: list[Any] = [query_vector_str, target_ids]
            next_index = 3
            extra_where = ""
            if request.tag_filters:
                tag_clause, tag_params, next_index = build_tag_filter_sql(request.tag_filters, next_index)
                if tag_clause:
                    extra_where = f" AND {tag_clause}"
                query_params.extend(tag_params)

            limit_index = next_index
            query_params.append(limit)

            results = await db.query_raw(
                f"""
                SELECT
                    e.{fields.id_field} as id,
                    e.{fields.vector_store_id_field} as vector_store_id,
                    e.{fields.content_field} as content,
                    e.{fields.metadata_field} as metadata,
                    (e.{fields.embedding_field} <=> $1::halfvec) as distance,
                    sm.id as sm_id,
                    sm.genres, sm.romance_subgenres, sm.tropes, sm.relationship_types,
                    sm.character_archetypes, sm.occupations, sm.settings, sm.sports,
                    sm.military, sm.kinks, sm.emotional_tone, sm.major_conflicts,
                    sm.coming_out, sm.first_love, sm.forbidden_romance, sm.found_family,
                    sm.slow_burn, sm.enemies_to_lovers, sm.hurt_comfort, sm.age_gap,
                    sm.college, sm.military_romance, sm.heat_level, sm.content_warnings,
                    sm.search_keywords, sm.pov, sm.tone, sm.explicitness,
                    sm.relationship_structure, sm.ending
                FROM {table_name} e
                LEFT JOIN {STORY_METADATA_TABLE} sm
                    ON sm.vector_store_id = e.{fields.vector_store_id_field}
                   AND sm.filename = e.{fields.metadata_field}->>'filename'
                WHERE e.{fields.vector_store_id_field} = ANY($2::text[]){extra_where}
                ORDER BY distance ASC
                LIMIT ${limit_index}
                """,
                *query_params,
            )
        else:
            results = []

        search_results = []
        for row in results:
            similarity_score = max(0, 1 - (row["distance"] / 2))
            metadata = row["metadata"] or {}
            filename = metadata.get("filename", "document.txt")
            store_id = row["vector_store_id"]

            search_results.append(
                SceneSearchResult(
                    vector_store_id=store_id,
                    vector_store_name=exposed_ids_by_id.get(store_id, "unknown"),
                    file_id=row["id"],
                    filename=filename,
                    score=similarity_score,
                    attributes=metadata if request.return_metadata else None,
                    content=[ContentChunk(type="text", text=row["content"])],
                    story_metadata=story_metadata_from_row(row) if row.get("sm_id") else None,
                )
            )

        return SceneSearchResponse(search_query=request.query, data=search_results, has_more=False)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Scene search failed: {str(e)}")


@app.post("/v1/vector_stores/{vector_store_id}/search", response_model=VectorStoreSearchResponse)
@app.post("/vector_stores/{vector_store_id}/search", response_model=VectorStoreSearchResponse)
async def search_vector_store(
    vector_store_id: str,
    request: VectorStoreSearchRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Search a vector store for similar content.
    """
    try:
        # Check if vector store exists
        vector_store_table = settings.table_names["vector_stores"]
        vector_store_result = await db.query_raw(
            f"SELECT id FROM {vector_store_table} WHERE id = $1",
            vector_store_id
        )
        if not vector_store_result:
            raise HTTPException(status_code=404, detail="Vector store not found")
        
        limit = min(request.limit or 20, 100)  # Cap at 100 results

        # Fast keyword fallback first. This keeps search usable even when
        # the embedding backend is temporarily unavailable.
        keyword_sql, keyword_params = _search_content_keywords_sql(request.query, limit, [vector_store_id])
        keyword_results = await db.query_raw(keyword_sql, *keyword_params)
        if keyword_results:
            search_results = []
            for row in keyword_results:
                metadata = row["metadata"] or {}
                filename = metadata.get("filename", "document.txt")
                search_results.append(
                    SearchResult(
                        file_id=row["id"],
                        filename=filename,
                        score=0.0,
                        attributes=metadata if request.return_metadata else None,
                        content=[ContentChunk(type="text", text=row["content"])],
                    )
                )

            return VectorStoreSearchResponse(
                search_query=request.query,
                data=search_results,
                has_more=False,
                next_page=None,
            )

        # Fall back to vector similarity search only when we can generate an embedding.
        query_embedding = await _safe_query_embedding(request.query)
        if query_embedding is None:
            return VectorStoreSearchResponse(
                search_query=request.query,
                data=[],
                has_more=False,
                next_page=None,
            )

        query_vector_str = "[" + ",".join(map(str, query_embedding)) + "]"

        # Build the raw SQL query for vector similarity search
        
        # Base query with vector similarity using cosine distance
        # Use configurable field names
        fields = settings.db_fields
        table_name = settings.table_names["embeddings"]
        
        # Build query with proper parameter placeholders for Prisma
        param_count = 1
        query_params = [query_vector_str, vector_store_id]
        
        base_query = f"""
        SELECT 
            {fields.id_field},
            {fields.content_field},
            {fields.metadata_field},
            ({fields.embedding_field} <=> ${param_count}::halfvec) as distance
        FROM {table_name} 
        WHERE {fields.vector_store_id_field} = ${param_count + 1}
        """
        param_count += 2
        
        # Add metadata filters if provided
        filter_conditions = []
        
        if request.filters:
            for key, value in request.filters.items():
                filter_conditions.append(f"{fields.metadata_field}->>${param_count} = ${param_count + 1}")
                query_params.extend([key, str(value)])
                param_count += 2
        
        if filter_conditions:
            base_query += " AND " + " AND ".join(filter_conditions)
        
        # Add ordering and limit
        final_query = base_query + f" ORDER BY distance ASC LIMIT {limit}"
        
        # Execute the query
        results = await db.query_raw(final_query, *query_params)
        
        # Convert results to SearchResult objects
        search_results = []
        for row in results:
            # Convert distance to similarity score (1 - normalized_distance)
            # Cosine distance ranges from 0 (identical) to 2 (opposite)
            similarity_score = max(0, 1 - (row['distance'] / 2))
            
            # Extract filename from metadata or use a default
            metadata = row[fields.metadata_field] or {}
            filename = metadata.get('filename', 'document.txt')
            
            content_chunks = [ContentChunk(type="text", text=row[fields.content_field])]
            
            result = SearchResult(
                file_id=row[fields.id_field],
                filename=filename,
                score=similarity_score,
                attributes=metadata if request.return_metadata else None,
                content=content_chunks
            )
            search_results.append(result)
        
        return VectorStoreSearchResponse(
            search_query=request.query,
            data=search_results,
            has_more=False,  # TODO: Implement pagination
            next_page=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/v1/vector_stores/{vector_store_id}/embeddings", response_model=EmbeddingResponse)
async def create_embedding(
    vector_store_id: str,
    request: EmbeddingCreateRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Add a single embedding to a vector store.
    """
    try:
        # Check if vector store exists
        vector_store_table = settings.table_names["vector_stores"]
        vector_store_result = await db.query_raw(
            f"SELECT id FROM {vector_store_table} WHERE id = $1",
            vector_store_id
        )
        if not vector_store_result:
            raise HTTPException(status_code=404, detail="Vector store not found")
        
        # Convert embedding to vector string format
        embedding_vector_str = "[" + ",".join(map(str, request.embedding)) + "]"
        
        # Insert embedding using configurable field names
        fields = settings.db_fields
        table_name = settings.table_names["embeddings"]
        
        result = await db.query_raw(
            f"""
            INSERT INTO {table_name} ({fields.id_field}, {fields.vector_store_id_field}, {fields.content_field}, 
                                     {fields.embedding_field}, {fields.metadata_field}, {fields.created_at_field})
            VALUES (gen_random_uuid(), $1, $2, $3::halfvec, $4, NOW())
            RETURNING {fields.id_field}, {fields.vector_store_id_field}, {fields.content_field}, 
                     {fields.metadata_field}, EXTRACT(EPOCH FROM {fields.created_at_field})::bigint as created_at_timestamp
            """,
            vector_store_id,
            request.content,
            embedding_vector_str,
            request.metadata or {}
        )
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to create embedding")
            
        embedding = result[0]
        
        # Update vector store statistics
        await db.query_raw(
            f"""
            UPDATE {vector_store_table} 
            SET 
                file_counts = jsonb_set(
                    jsonb_set(
                        COALESCE(file_counts, '{{"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0}}'::jsonb),
                        '{{completed}}',
                        (COALESCE(file_counts->>'completed', '0')::int + 1)::text::jsonb
                    ),
                    '{{total}}',
                    (COALESCE(file_counts->>'total', '0')::int + 1)::text::jsonb
                ),
                usage_bytes = COALESCE(usage_bytes, 0) + LENGTH($2),
                last_active_at = NOW()
            WHERE id = $1
            """,
            vector_store_id,
            request.content
        )
        
        return EmbeddingResponse(
            id=embedding[fields.id_field],
            vector_store_id=embedding[fields.vector_store_id_field],
            content=embedding[fields.content_field],
            metadata=embedding[fields.metadata_field],
            created_at=int(embedding["created_at_timestamp"])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create embedding: {str(e)}")


@app.post("/v1/vector_stores/{vector_store_id}/embeddings/batch", response_model=EmbeddingBatchCreateResponse)
async def create_embeddings_batch(
    vector_store_id: str,
    request: EmbeddingBatchCreateRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Add multiple embeddings to a vector store in batch.
    """
    try:
        # Check if vector store exists
        vector_store_table = settings.table_names["vector_stores"]
        vector_store_result = await db.query_raw(
            f"SELECT id FROM {vector_store_table} WHERE id = $1",
            vector_store_id
        )
        if not vector_store_result:
            raise HTTPException(status_code=404, detail="Vector store not found")
        
        embeddings = await insert_embeddings_records(vector_store_id, request.embeddings)
        
        return EmbeddingBatchCreateResponse(
            data=embeddings,
            created=int(time.time())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create embeddings batch: {str(e)}")


@app.post("/v1/vector_stores/upload-text-files")
async def upload_text_files(
    files: list[UploadFile] = File(...),
    vector_store_id: Optional[str] = Form(None),
    vector_store_name: Optional[str] = Form(None),
    chunk_size: int = Form(1600),
    chunk_overlap: int = Form(200),
    api_key: str = Depends(get_api_key),
):
    if chunk_size < 200:
        raise HTTPException(status_code=400, detail="chunk_size must be at least 200")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=400,
            detail="chunk_overlap must be non-negative and smaller than chunk_size",
        )

    resolved_vector_store_id, resolved_vector_store_name, created_new_store = await get_vector_store_id_or_create(
        vector_store_id,
        vector_store_name,
    )
    record_activity(
        "upload_started",
        f"Upload started for store '{resolved_vector_store_name}'",
        vector_store_id=resolved_vector_store_id,
        store_name=resolved_vector_store_name,
        total_files=len(files),
    )

    embedding_requests: list[EmbeddingCreateRequest] = []
    file_summaries: list[dict[str, Any]] = []
    extracted_story_metadata: dict[str, StoryMetadata] = {}
    file_stage_details: dict[str, dict[str, Any]] = {}

    for upload in files:
        if not upload.filename:
            continue
        await record_ingest_file_stage(
            resolved_vector_store_id,
            resolved_vector_store_name,
            upload.filename,
            "queued",
            status="queued",
        )
        if not upload.filename.lower().endswith(".txt"):
            await record_ingest_file_stage(
                resolved_vector_store_id,
                resolved_vector_store_name,
                upload.filename,
                "failed",
                status="failed",
                error="Only .txt files are supported",
            )
            raise HTTPException(
                status_code=400,
                detail=f"Only .txt files are supported: {upload.filename}",
            )

        raw_bytes = await upload.read()
        try:
            backup_uploaded_text_file(resolved_vector_store_name, upload.filename, raw_bytes)
            await record_ingest_file_stage(
                resolved_vector_store_id,
                resolved_vector_store_name,
                upload.filename,
                "backed_up",
                status="completed",
            )
        except Exception as exc:
            await record_ingest_file_stage(
                resolved_vector_store_id,
                resolved_vector_store_name,
                upload.filename,
                "failed",
                status="failed",
                error=f"Backup failed: {str(exc)}",
            )
            raise HTTPException(status_code=500, detail=f"Failed to back up file {upload.filename}: {str(exc)}")
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("utf-8-sig", errors="replace")

        extracted_story_metadata[upload.filename] = await extract_story_metadata_for_upload(
            upload.filename,
            text,
            resolved_vector_store_name,
        )
        await record_ingest_file_stage(
            resolved_vector_store_id,
            resolved_vector_store_name,
            upload.filename,
            "metadata_extracted",
            status="completed",
        )

        chunks = chunk_text(text, max_chars=chunk_size, overlap=chunk_overlap)
        if not chunks:
            await record_ingest_file_stage(
                resolved_vector_store_id,
                resolved_vector_store_name,
                upload.filename,
                "skipped",
                status="skipped",
                detail={"reason": "No usable text content found"},
            )
            continue

        try:
            embeddings = await embedding_service.generate_embeddings(chunks)
            await record_ingest_file_stage(
                resolved_vector_store_id,
                resolved_vector_store_name,
                upload.filename,
                "embeddings_generated",
                status="completed",
                detail={"chunk_count": len(chunks)},
            )
        except Exception as exc:
            await record_ingest_file_stage(
                resolved_vector_store_id,
                resolved_vector_store_name,
                upload.filename,
                "failed",
                status="failed",
                error=f"Embedding generation failed: {str(exc)}",
            )
            raise HTTPException(
                status_code=502,
                detail=(
                    "Failed to generate embeddings via LiteLLM proxy. "
                    "Check EMBEDDING__BASE_URL and EMBEDDING__API_KEY settings. "
                    f"Details: {str(exc)}"
                ),
            )
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            embedding_requests.append(
                EmbeddingCreateRequest(
                    content=chunk,
                    embedding=embedding,
                    metadata={
                        "filename": upload.filename,
                        "chunk_index": index,
                        "chunk_count": len(chunks),
                        "source": "upload_ui",
                    },
                )
            )

        file_summaries.append(
            {
                "filename": upload.filename,
                "chunk_count": len(chunks),
                "character_count": len(text),
            }
        )
        file_stage_details[upload.filename] = {
            "chunk_count": len(chunks),
            "character_count": len(text),
        }
        record_activity(
            "file_completed",
            f"Processed {upload.filename}",
            vector_store_id=resolved_vector_store_id,
            store_name=resolved_vector_store_name,
            filename=upload.filename,
            chunk_count=len(chunks),
            character_count=len(text),
        )

    if not embedding_requests:
        raise HTTPException(status_code=400, detail="No usable text content found in uploaded files")

    created_embeddings = await insert_embeddings_records(
        resolved_vector_store_id,
        embedding_requests,
    )
    for filename, details in file_stage_details.items():
        await record_ingest_file_stage(
            resolved_vector_store_id,
            resolved_vector_store_name,
            filename,
            "embeddings_inserted",
            status="completed",
            detail=details,
        )

    metadata_rows_upserted = await upsert_story_metadata_rows(resolved_vector_store_id, extracted_story_metadata)
    for filename, details in file_stage_details.items():
        await record_ingest_file_stage(
            resolved_vector_store_id,
            resolved_vector_store_name,
            filename,
            "metadata_upserted",
            status="completed",
            detail=details,
        )
        await record_ingest_file_stage(
            resolved_vector_store_id,
            resolved_vector_store_name,
            filename,
            "completed",
            status="completed",
            detail=details,
        )
    record_activity(
        "upload_completed",
        f"Upload completed for store '{resolved_vector_store_name}'",
        vector_store_id=resolved_vector_store_id,
        store_name=resolved_vector_store_name,
        files_processed=len(file_summaries),
        embeddings_created=len(created_embeddings),
        story_metadata_rows_upserted=metadata_rows_upserted,
    )

    return {
        "vector_store_id": resolved_vector_store_id,
        "created_new_store": created_new_store,
        "files": file_summaries,
        "embeddings_created": len(created_embeddings),
        "story_metadata_rows_upserted": metadata_rows_upserted,
        "created": int(time.time()),
    }


@app.post("/v1/vector_stores/upload-text-files/stream")
async def upload_text_files_stream(
    files: list[UploadFile] = File(...),
    vector_store_id: Optional[str] = Form(None),
    vector_store_name: Optional[str] = Form(None),
    chunk_size: int = Form(1600),
    chunk_overlap: int = Form(200),
    api_key: str = Depends(get_api_key),
):
    async def event_stream():
        try:
            if chunk_size < 200:
                raise HTTPException(status_code=400, detail="chunk_size must be at least 200")
            if chunk_overlap < 0 or chunk_overlap >= chunk_size:
                raise HTTPException(
                    status_code=400,
                    detail="chunk_overlap must be non-negative and smaller than chunk_size",
                )

            resolved_vector_store_id, resolved_vector_store_name, created_new_store = await get_vector_store_id_or_create(
                vector_store_id,
                vector_store_name,
            )
            record_activity(
                "upload_started",
                f"Stream upload started for store '{resolved_vector_store_name}'",
                vector_store_id=resolved_vector_store_id,
                store_name=resolved_vector_store_name,
                total_files=len(files),
            )

            valid_filenames = [upload.filename for upload in files if upload.filename]
            total_files = len(valid_filenames)

            yield json.dumps(
                {
                    "type": "started",
                    "vector_store_id": resolved_vector_store_id,
                    "created_new_store": created_new_store,
                    "total_files": total_files,
                }
            ) + "\n"

            embedding_requests: list[EmbeddingCreateRequest] = []
            file_summaries: list[dict[str, Any]] = []
            extracted_story_metadata: dict[str, StoryMetadata] = {}
            file_stage_details: dict[str, dict[str, Any]] = {}
            embedding_batch_size = max(1, min(64, settings.embedding.concurrency * 8))

            processed_index = 0
            for upload in files:
                if not upload.filename:
                    continue

                processed_index += 1
                await record_ingest_file_stage(
                    resolved_vector_store_id,
                    resolved_vector_store_name,
                    upload.filename,
                    "queued",
                    status="queued",
                    detail={"index": processed_index, "total_files": total_files},
                )
                record_activity(
                    "file_started",
                    f"Starting {upload.filename}",
                    vector_store_id=resolved_vector_store_id,
                    store_name=resolved_vector_store_name,
                    filename=upload.filename,
                    index=processed_index,
                    total_files=total_files,
                )
                yield json.dumps(
                    {
                        "type": "file_started",
                        "index": processed_index,
                        "total_files": total_files,
                        "filename": upload.filename,
                    }
                ) + "\n"

                if not upload.filename.lower().endswith(".txt"):
                    await record_ingest_file_stage(
                        resolved_vector_store_id,
                        resolved_vector_store_name,
                        upload.filename,
                        "failed",
                        status="failed",
                        error="Only .txt files are supported",
                    )
                    raise HTTPException(
                        status_code=400,
                        detail=f"Only .txt files are supported: {upload.filename}",
                    )

                raw_bytes = await upload.read()
                try:
                    backup_uploaded_text_file(resolved_vector_store_name, upload.filename, raw_bytes)
                    await record_ingest_file_stage(
                        resolved_vector_store_id,
                        resolved_vector_store_name,
                        upload.filename,
                        "backed_up",
                        status="completed",
                    )
                except Exception as exc:
                    await record_ingest_file_stage(
                        resolved_vector_store_id,
                        resolved_vector_store_name,
                        upload.filename,
                        "failed",
                        status="failed",
                        error=f"Backup failed: {str(exc)}",
                    )
                    raise HTTPException(status_code=500, detail=f"Failed to back up file {upload.filename}: {str(exc)}")
                try:
                    text = raw_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    text = raw_bytes.decode("utf-8-sig", errors="replace")

                extracted_story_metadata[upload.filename] = await extract_story_metadata_for_upload(
                    upload.filename,
                    text,
                    resolved_vector_store_name,
                )
                await record_ingest_file_stage(
                    resolved_vector_store_id,
                    resolved_vector_store_name,
                    upload.filename,
                    "metadata_extracted",
                    status="completed",
                )

                chunks = chunk_text(text, max_chars=chunk_size, overlap=chunk_overlap)
                if not chunks:
                    await record_ingest_file_stage(
                        resolved_vector_store_id,
                        resolved_vector_store_name,
                        upload.filename,
                        "skipped",
                        status="skipped",
                        detail={"reason": "No usable text content found"},
                    )
                    yield json.dumps(
                        {
                            "type": "file_skipped",
                            "index": processed_index,
                            "total_files": total_files,
                            "filename": upload.filename,
                            "reason": "No usable text content found",
                        }
                    ) + "\n"
                    continue

                try:
                    embeddings: list[list[float]] = []
                    completed_chunks = 0
                    total_chunks = len(chunks)

                    for start in range(0, total_chunks, embedding_batch_size):
                        batch_chunks = chunks[start:start + embedding_batch_size]
                        batch_embeddings = await embedding_service.generate_embeddings(batch_chunks)
                        embeddings.extend(batch_embeddings)
                        completed_chunks += len(batch_chunks)

                        yield json.dumps(
                            {
                                "type": "file_progress",
                                "index": processed_index,
                                "total_files": total_files,
                                "filename": upload.filename,
                                "completed_chunks": completed_chunks,
                                "total_chunks": total_chunks,
                            }
                        ) + "\n"
                    await record_ingest_file_stage(
                        resolved_vector_store_id,
                        resolved_vector_store_name,
                        upload.filename,
                        "embeddings_generated",
                        status="completed",
                        detail={"chunk_count": len(chunks)},
                    )
                except Exception as exc:
                    await record_ingest_file_stage(
                        resolved_vector_store_id,
                        resolved_vector_store_name,
                        upload.filename,
                        "failed",
                        status="failed",
                        error=f"Embedding generation failed: {str(exc)}",
                    )
                    raise HTTPException(
                        status_code=502,
                        detail=(
                            "Failed to generate embeddings via LiteLLM proxy. "
                            "Check EMBEDDING__BASE_URL and EMBEDDING__API_KEY settings. "
                            f"Details: {str(exc)}"
                        ),
                    )

                if len(embeddings) != len(chunks):
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            f"Embedding count mismatch for {upload.filename}: "
                            f"expected {len(chunks)}, got {len(embeddings)}"
                        ),
                    )

                for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    embedding_requests.append(
                        EmbeddingCreateRequest(
                            content=chunk,
                            embedding=embedding,
                            metadata={
                                "filename": upload.filename,
                                "chunk_index": index,
                                "chunk_count": len(chunks),
                                "source": "upload_ui",
                            },
                        )
                    )

                file_summary = {
                    "filename": upload.filename,
                    "chunk_count": len(chunks),
                    "character_count": len(text),
                }
                file_summaries.append(file_summary)
                file_stage_details[upload.filename] = file_summary
                record_activity(
                    "file_completed",
                    f"Processed {upload.filename}",
                    vector_store_id=resolved_vector_store_id,
                    store_name=resolved_vector_store_name,
                    filename=upload.filename,
                    chunk_count=len(chunks),
                    character_count=len(text),
                    index=processed_index,
                    total_files=total_files,
                )

                yield json.dumps(
                    {
                        "type": "file_completed",
                        "index": processed_index,
                        "total_files": total_files,
                        **file_summary,
                    }
                ) + "\n"

            if not embedding_requests:
                raise HTTPException(status_code=400, detail="No usable text content found in uploaded files")

            created_embeddings = await insert_embeddings_records(
                resolved_vector_store_id,
                embedding_requests,
            )
            for filename, details in file_stage_details.items():
                await record_ingest_file_stage(
                    resolved_vector_store_id,
                    resolved_vector_store_name,
                    filename,
                    "embeddings_inserted",
                    status="completed",
                    detail=details,
                )

            metadata_rows_upserted = await upsert_story_metadata_rows(
                resolved_vector_store_id,
                extracted_story_metadata,
            )
            for filename, details in file_stage_details.items():
                await record_ingest_file_stage(
                    resolved_vector_store_id,
                    resolved_vector_store_name,
                    filename,
                    "metadata_upserted",
                    status="completed",
                    detail=details,
                )
                await record_ingest_file_stage(
                    resolved_vector_store_id,
                    resolved_vector_store_name,
                    filename,
                    "completed",
                    status="completed",
                    detail=details,
                )

            result_payload = {
                "vector_store_id": resolved_vector_store_id,
                "created_new_store": created_new_store,
                "files": file_summaries,
                "embeddings_created": len(created_embeddings),
                "story_metadata_rows_upserted": metadata_rows_upserted,
                "created": int(time.time()),
            }
            record_activity(
                "upload_completed",
                f"Stream upload completed for store '{resolved_vector_store_name}'",
                vector_store_id=resolved_vector_store_id,
                store_name=resolved_vector_store_name,
                files_processed=len(file_summaries),
                embeddings_created=len(created_embeddings),
                story_metadata_rows_upserted=metadata_rows_upserted,
            )
            yield json.dumps({"type": "complete", "result": result_payload}) + "\n"

        except HTTPException as exc:
            record_activity("upload_error", "Upload failed", status_code=exc.status_code, detail=str(exc.detail))
            yield json.dumps(
                {
                    "type": "error",
                    "status_code": exc.status_code,
                    "detail": exc.detail,
                }
            ) + "\n"
        except Exception as exc:
            record_activity("upload_error", "Upload failed", status_code=500, detail=str(exc))
            yield json.dumps(
                {
                    "type": "error",
                    "status_code": 500,
                    "detail": f"Unexpected upload failure: {str(exc)}",
                }
            ) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _active_ingest_jobs_snapshot() -> list[dict[str, Any]]:
        jobs = [
                dict(job)
                for job in vector_store_ingest_jobs.values()
                if job.get("status") in {"queued", "running"}
        ]

        queued = [j for j in jobs if str(j.get("status")) == "queued"]
        queued.sort(key=lambda j: int(j.get("queued_at") or 0))
        queue_pos = {
                str(j.get("job_id")): i + 1
                for i, j in enumerate(queued)
                if j.get("job_id")
        }
        queued_total = len(queued)
        running_total = sum(1 for j in jobs if str(j.get("status")) == "running")

        for job in jobs:
                pos = queue_pos.get(str(job.get("job_id")))
                job["queue_position"] = pos
                job["queued_ahead"] = (pos - 1) if pos else 0
                job["queued_total"] = queued_total
                job["running_total"] = running_total

        jobs.sort(key=lambda j: (int(j.get("queued_at") or 0), str(j.get("job_id") or "")))
        return jobs


@app.get("/health/activity")
async def public_health_activity(limit: int = 25):
        safe_limit = min(max(limit, 1), 500)
        items = list(activity_events)[-safe_limit:]
        return {"status": "healthy", "data": items, "count": len(items), "timestamp": int(time.time())}


@app.get("/health/summary")
async def public_health_summary():
        vector_store_table = settings.table_names["vector_stores"]
        stores = await db.query_raw(
                f"SELECT id, name, file_counts, usage_bytes, status, metadata FROM {vector_store_table}"
        )

        total_stores = len(stores)
        exposed_stores = sum(1 for s in stores if is_store_exposed(s.get("metadata")))
        total_usage_bytes = sum(int(s.get("usage_bytes") or 0) for s in stores)
        total_files = 0
        for store in stores:
                file_counts = store.get("file_counts") or {}
                total_files += int(file_counts.get("total") or 0)

        active_jobs = await _active_ingest_jobs_snapshot()
        recent_activity = list(activity_events)[-25:]

        return {
                "status": "healthy",
                "timestamp": int(time.time()),
                "vector_store_totals": {
                        "stores": total_stores,
                        "exposed_stores": exposed_stores,
                        "files": total_files,
                        "usage_bytes": total_usage_bytes,
                },
                "active_ingest_jobs": active_jobs,
                "recent_activity": recent_activity,
        }


@app.get("/v1/health")
async def health_check_json():
        """JSON health probe endpoint."""
        return {"status": "healthy", "timestamp": int(time.time())}


@app.get("/health", response_class=HTMLResponse)
async def health_dashboard():
        """Simple live health dashboard with ingest/job visibility."""
        return """<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Service Health</title>
    <style>
        :root { --bg:#0c1224; --card:#131b36; --line:#2a365f; --text:#e8eeff; --muted:#9eb0e8; --ok:#33d6a6; }
        body { margin:0; font-family: Georgia, \"Times New Roman\", serif; color:var(--text); background:linear-gradient(180deg,#0f1731,#0b1020); }
        main { width:min(1080px, calc(100vw - 24px)); margin:16px auto; display:grid; gap:14px; }
        .card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:14px; }
        h1 { margin:0; font-size:1.6rem; }
        .sub { color:var(--muted); margin-top:6px; }
        .stats { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }
        .pill { border:1px solid var(--line); border-radius:12px; padding:10px; background:rgba(255,255,255,0.02); }
        .k { color:var(--muted); font-size:.85rem; }
        .v { font-size:1.15rem; margin-top:4px; }
        .job, .evt { border:1px solid var(--line); border-radius:10px; padding:10px; margin-top:8px; background:rgba(255,255,255,0.02); }
        .row { display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap; }
        .ok { color:var(--ok); font-weight:700; }
        code { color:#b8c8ff; }
        @media (max-width: 800px) { .stats { grid-template-columns:repeat(2,minmax(0,1fr)); } }
    </style>
</head>
<body>
    <main>
        <section class=\"card\">
            <h1>Service Health</h1>
            <div class=\"sub\">Live status for vector stores, active ingest jobs, and recent activity.</div>
            <div class=\"sub\" id=\"stamp\">Loading...</div>
        </section>

        <section class=\"card\">
            <div class=\"stats\">
                <div class=\"pill\"><div class=\"k\">Stores</div><div class=\"v\" id=\"stores\">-</div></div>
                <div class=\"pill\"><div class=\"k\">Exposed</div><div class=\"v\" id=\"exposed\">-</div></div>
                <div class=\"pill\"><div class=\"k\">Files</div><div class=\"v\" id=\"files\">-</div></div>
                <div class=\"pill\"><div class=\"k\">Usage bytes</div><div class=\"v\" id=\"usage\">-</div></div>
            </div>
        </section>

        <section class=\"card\">
            <h2>Active ingest jobs</h2>
            <div id=\"jobs\"></div>
        </section>

        <section class=\"card\">
            <h2>Recent activity</h2>
            <div id=\"events\"></div>
        </section>
    </main>

    <script>
        const fmt = (ts) => new Date((ts || Math.floor(Date.now()/1000)) * 1000).toLocaleTimeString();
        const jobsEl = document.getElementById('jobs');
        const eventsEl = document.getElementById('events');

        function renderJobs(jobs) {
            jobsEl.innerHTML = '';
            if (!jobs || !jobs.length) {
                jobsEl.innerHTML = '<div class=\"sub\">No queued/running jobs.</div>';
                return;
            }
            for (const j of jobs) {
                const d = document.createElement('div');
                d.className = 'job';
                d.innerHTML = `<div class=\"row\"><strong>${j.store_name || j.topic || j.vector_store_id}</strong><span class=\"ok\">${j.status}</span></div>
                    <div class=\"sub\">job <code>${j.job_id}</code> · queued ${fmt(j.queued_at)} · queue ${j.queue_position || '-'} / ${j.queued_total || 0}</div>`;
                jobsEl.appendChild(d);
            }
        }

        function renderEvents(events) {
            eventsEl.innerHTML = '';
            if (!events || !events.length) {
                eventsEl.innerHTML = '<div class=\"sub\">No recent activity.</div>';
                return;
            }
            for (const e of events) {
                const d = document.createElement('div');
                d.className = 'evt';
                d.innerHTML = `<div class=\"row\"><strong>${e.type}</strong><span>${fmt(e.timestamp)}</span></div><div>${e.message}</div>`;
                eventsEl.appendChild(d);
            }
        }

        async function refresh() {
            const res = await fetch('/health/summary');
            const data = await res.json();
            document.getElementById('stamp').textContent = `Updated ${fmt(data.timestamp)} · status ${data.status}`;
            document.getElementById('stores').textContent = data.vector_store_totals?.stores ?? '-';
            document.getElementById('exposed').textContent = data.vector_store_totals?.exposed_stores ?? '-';
            document.getElementById('files').textContent = data.vector_store_totals?.files ?? '-';
            document.getElementById('usage').textContent = data.vector_store_totals?.usage_bytes ?? '-';
            renderJobs(data.active_ingest_jobs || []);
            renderEvents(data.recent_activity || []);
        }

        refresh();
        setInterval(refresh, 2000);
    </script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True) 