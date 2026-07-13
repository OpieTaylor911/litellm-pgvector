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
from fastapi.responses import FileResponse, StreamingResponse
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
    ContentChunk
)
from config import settings
from embedding_service import embedding_service

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


async def create_vector_store_record(
    name: str,
    expires_after: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    vector_store_table = settings.table_names["vector_stores"]
    result = await db.query_raw(
        f"""
        INSERT INTO {vector_store_table} (id, name, file_counts, status, usage_bytes, expires_after, metadata, created_at)
        VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6, NOW())
        RETURNING id, name, file_counts, status, usage_bytes, expires_after, expires_at, last_active_at, metadata,
                 EXTRACT(EPOCH FROM created_at)::bigint as created_at_timestamp
        """,
        name,
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
        return vector_store_id, result[0]["name"], False

    cleaned_name = (vector_store_name or "").strip()
    if not cleaned_name:
        raise HTTPException(
            status_code=400,
            detail="Provide either vector_store_id or vector_store_name",
        )

    vector_store = await create_vector_store_record(
        cleaned_name,
        metadata={"created_via": "upload_ui"},
    )
    return vector_store["id"], vector_store["name"], True


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
                f"(gen_random_uuid(), ${param_count}, ${param_count + 1}, ${param_count + 2}::vector, ${param_count + 3}, NOW())"
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
            metadata=vector_store["metadata"]
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
                metadata=row["metadata"]
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
        
        # Generate embedding for query
        query_embedding = await generate_query_embedding(request.query)
        query_vector_str = "[" + ",".join(map(str, query_embedding)) + "]"
        
        # Build the raw SQL query for vector similarity search
        limit = min(request.limit or 20, 100)  # Cap at 100 results
        
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
            ({fields.embedding_field} <=> ${param_count}::vector) as distance
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
            VALUES (gen_random_uuid(), $1, $2, $3::vector, $4, NOW())
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

    for upload in files:
        if not upload.filename:
            continue
        if not upload.filename.lower().endswith(".txt"):
            raise HTTPException(
                status_code=400,
                detail=f"Only .txt files are supported: {upload.filename}",
            )

        raw_bytes = await upload.read()
        try:
            backup_uploaded_text_file(resolved_vector_store_name, upload.filename, raw_bytes)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to back up file {upload.filename}: {str(exc)}")
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("utf-8-sig", errors="replace")

        chunks = chunk_text(text, max_chars=chunk_size, overlap=chunk_overlap)
        if not chunks:
            continue

        try:
            embeddings = await embedding_service.generate_embeddings(chunks)
        except Exception as exc:
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
    record_activity(
        "upload_completed",
        f"Upload completed for store '{resolved_vector_store_name}'",
        vector_store_id=resolved_vector_store_id,
        store_name=resolved_vector_store_name,
        files_processed=len(file_summaries),
        embeddings_created=len(created_embeddings),
    )

    return {
        "vector_store_id": resolved_vector_store_id,
        "created_new_store": created_new_store,
        "files": file_summaries,
        "embeddings_created": len(created_embeddings),
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
            embedding_batch_size = max(1, min(64, settings.embedding.concurrency * 8))

            processed_index = 0
            for upload in files:
                if not upload.filename:
                    continue

                processed_index += 1
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
                    raise HTTPException(
                        status_code=400,
                        detail=f"Only .txt files are supported: {upload.filename}",
                    )

                raw_bytes = await upload.read()
                try:
                    backup_uploaded_text_file(resolved_vector_store_name, upload.filename, raw_bytes)
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"Failed to back up file {upload.filename}: {str(exc)}")
                try:
                    text = raw_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    text = raw_bytes.decode("utf-8-sig", errors="replace")

                chunks = chunk_text(text, max_chars=chunk_size, overlap=chunk_overlap)
                if not chunks:
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
                except Exception as exc:
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

            result_payload = {
                "vector_store_id": resolved_vector_store_id,
                "created_new_store": created_new_store,
                "files": file_summaries,
                "embeddings_created": len(created_embeddings),
                "created": int(time.time()),
            }
            record_activity(
                "upload_completed",
                f"Stream upload completed for store '{resolved_vector_store_name}'",
                vector_store_id=resolved_vector_store_id,
                store_name=resolved_vector_store_name,
                files_processed=len(file_summaries),
                embeddings_created=len(created_embeddings),
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


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": int(time.time())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True) 