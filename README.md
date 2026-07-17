# OpenAI Vector Stores API with PGVector

A FastAPI application that provides OpenAI-compatible vector store endpoints using PGVector and LiteLLM proxy for embeddings.

## Features

- 🔌 OpenAI-compatible API endpoints
- 🗄️ PGVector for efficient vector storage and similarity search
- 🎛️ Configurable database field mappings
- 🔄 LiteLLM proxy integration for any embedding model
- 🐳 Docker support
- ⚡ FastAPI with async support

## API Endpoints

### 1. Create Vector Store
```bash
curl -X POST \
  http://localhost:8000/v1/vector_stores \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Support FAQ"
  }'
```

### 2. List Vector Stores
```bash
# List all vector stores
curl -X GET \
  http://localhost:8000/v1/vector_stores \
  -H "Authorization: Bearer your-api-key"

# List with pagination (limit and after parameters)
curl -X GET \
  "http://localhost:8000/v1/vector_stores?limit=10&after=vs_abc123" \
  -H "Authorization: Bearer your-api-key"
```

### 3. Add Single Embedding to Vector Store
```bash
curl -X POST \
  http://localhost:8000/v1/vector_stores/vs_abc123/embeddings \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Our return policy allows returns within 30 days of purchase.",
    "embedding": [0.1, 0.2, 0.3, ...],
    "metadata": {
      "category": "returns",
      "source": "faq",
      "id": "return_policy_1"
    }
  }'
```

### 4. Add Multiple Embeddings (Batch)
```bash
curl -X POST \
  http://localhost:8000/v1/vector_stores/vs_abc123/embeddings/batch \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "embeddings": [
      {
        "content": "Our return policy allows returns within 30 days of purchase.",
        "embedding": [0.1, 0.2, 0.3, ...],
        "metadata": {"category": "returns"}
      },
      {
        "content": "Shipping is free for orders over $50.",
        "embedding": [0.4, 0.5, 0.6, ...],
        "metadata": {"category": "shipping"}
      }
    ]
  }'
```

### 5. Search Vector Store
```bash
curl -X POST \
  http://localhost:8000/v1/vector_stores/vs_abc123/search \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the return policy?",
    "limit": 20,
    "filters": {"category": "support"}
  }'
```

## Configuration

### Environment Variables

Create a `.env` file with the following configuration:

```bash
# Database Configuration
DATABASE_URL="postgresql://username:password@localhost:5432/vectordb?schema=public"

# API Configuration
SERVER_API_KEY="your-api-key-here"

# Server Configuration
HOST="0.0.0.0"
PORT=8000

# LiteLLM Proxy Configuration
EMBEDDING__MODEL="text-embedding-ada-002"
EMBEDDING__BASE_URL="http://localhost:4000"
EMBEDDING__API_KEY="sk-1234"
EMBEDDING__DIMENSIONS=1536

# Database Field Configuration (optional)
DB_FIELDS__ID_FIELD="id"
DB_FIELDS__CONTENT_FIELD="content"
DB_FIELDS__METADATA_FIELD="metadata"
DB_FIELDS__EMBEDDING_FIELD="embedding"
DB_FIELDS__VECTOR_STORE_ID_FIELD="vector_store_id"
DB_FIELDS__CREATED_AT_FIELD="created_at"
```

### Database Field Mapping

You can customize the database field names by setting environment variables:

- `DB_FIELDS__ID_FIELD` - Primary key field (default: "id")
- `DB_FIELDS__CONTENT_FIELD` - Text content field (default: "content")
- `DB_FIELDS__METADATA_FIELD` - JSON metadata field (default: "metadata")
- `DB_FIELDS__EMBEDDING_FIELD` - Vector embedding field (default: "embedding")
- `DB_FIELDS__VECTOR_STORE_ID_FIELD` - Foreign key field (default: "vector_store_id")
- `DB_FIELDS__CREATED_AT_FIELD` - Timestamp field (default: "created_at")

### LiteLLM Proxy Configuration

The application uses LiteLLM proxy for embeddings. Configure it with:

- `EMBEDDING__MODEL` - Model name (e.g., "text-embedding-ada-002")
- `EMBEDDING__BASE_URL` - LiteLLM proxy URL (e.g., "http://localhost:4000")
- `EMBEDDING__API_KEY` - LiteLLM proxy API key
- `EMBEDDING__DIMENSIONS` - Embedding dimensions (default: 1536)

## Setup and Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Database Setup

```bash
# Generate Prisma client
prisma generate

# Run database migrations
prisma db push
```

### 3. Set up LiteLLM Proxy

Start LiteLLM proxy pointing to your preferred embedding model:

```bash
# Example: Start LiteLLM proxy for OpenAI
litellm --model text-embedding-ada-002 --port 4000
```

### 4. Run the Application

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Run As A System Service (Auto-Start On Boot)

On Linux systems with systemd, install this API as a boot-time service:

```bash
cd /opt/pgvector/litellm-pgvector
sudo ./scripts/install_systemd_service.sh
```

The installer will:

- Create/update `.venv` and install `requirements.txt`
- Generate Prisma client (if `prisma/schema.prisma` exists)
- Create `/etc/default/litellm-pgvector` if missing
- Install and enable `litellm-pgvector.service`

After installation, manage the service with:

```bash
sudo systemctl status litellm-pgvector
sudo systemctl restart litellm-pgvector
sudo journalctl -u litellm-pgvector -f
```

Service runtime config is read from:

```text
/etc/default/litellm-pgvector
```

Update `SERVER_API_KEY`, database connection, and embedding settings there, then restart the service.
The installer defaults the systemd service to port `18001` (change `PORT` in that file if needed).
The service also loads `litellm-pgvector/.env` and uses those values as overrides, so UI/API credentials in `.env` will take effect after restart.

## Docker Deployment

### Build and run with Docker:

```bash
# Build the image
docker build -t vector-store-api .

# Run the container
docker run -p 8000:8000 --env-file .env vector-store-api
```

## Database Schema

The application uses two main tables:

### vector_stores
- `id` (string, primary key)
- `name` (string)
- `file_counts` (json)
- `status` (string)
- `usage_bytes` (integer)
- `created_at` (timestamp)
- `expires_after` (json, optional)
- `expires_at` (timestamp, optional)
- `last_active_at` (timestamp, optional)
- `metadata` (json, optional)

### embeddings
- `id` (string, primary key)
- `vector_store_id` (string, foreign key)
- `content` (string)
- `embedding` (vector(1536))
- `metadata` (json, optional)
- `created_at` (timestamp)

### story_metadata
Structured per-file tags for the story-ingest workflow (one row per `vector_store_id` + `filename`, cascades on vector store delete):
- `id` (string, primary key)
- `vector_store_id` (string, foreign key, `onDelete: Cascade`)
- `filename` (string, unique together with `vector_store_id`)
- Tag arrays (json lists of strings): `genres`, `romance_subgenres`, `tropes`, `relationship_types`, `character_archetypes`, `occupations`, `settings`, `sports`, `military`, `kinks`, `emotional_tone`, `major_conflicts`, `content_warnings`, `search_keywords`, `tone`
- Trope booleans: `coming_out`, `first_love`, `forbidden_romance`, `found_family`, `slow_burn`, `enemies_to_lovers`, `hurt_comfort`, `age_gap`, `college`, `military_romance`
- Scalars: `heat_level` (int 1-5), `pov`, `explicitness`, `relationship_structure`, `ending` (all nullable strings)
- `created_at` / `updated_at` (timestamp)

Managed through `GET/PUT /v1/vector_stores/{id}/stories/{filename}/metadata` and `POST /v1/vector_stores/{id}/stories/metadata/bulk` — see [Manual Ingest & Metadata Workflow](#manual-ingest--metadata-workflow) below.

## Supported Models

Any embedding model supported by LiteLLM proxy can be used. Examples:

- OpenAI: `text-embedding-ada-002`, `text-embedding-3-small`, `text-embedding-3-large`
- Cohere: `embed-english-v3.0`, `embed-multilingual-v3.0`
- Voyage: `voyage-2`, `voyage-large-2`
- And many more...

## API Response Format

### Vector Store Response
```json
{
  "id": "vs_abc123",
  "object": "vector_store",
  "created_at": 1699024800,
  "name": "Support FAQ",
  "usage_bytes": 0,
  "file_counts": {
    "in_progress": 0,
    "completed": 0,
    "failed": 0,
    "cancelled": 0,
    "total": 0
  },
  "status": "completed",
  "metadata": {}
}
```

### Vector Store List Response
```json
{
  "object": "list",
  "data": [
    {
      "id": "vs_abc123",
      "object": "vector_store",
      "created_at": 1699024800,
      "name": "Support FAQ",
      "usage_bytes": 1024,
      "file_counts": {"completed": 5, "total": 5},
      "status": "completed",
      "metadata": {}
    }
  ],
  "first_id": "vs_abc123",
  "last_id": "vs_def456",
  "has_more": false
}
```

### Search Response
```json
{
  "object": "vector_store.search",
  "data": [
    {
      "id": "emb_123",
      "content": "Return policy text...",
      "score": 0.95,
      "metadata": {"category": "support"}
    }
  ],
  "usage": {
    "total_tokens": 1
  }
}
```

## Example Search Request

```bash
curl -X POST \
  http://localhost:8000/v1/vector_stores/vs_support_faq/search \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I return an item?",
    "limit": 5,
    "return_metadata": true
  }'
```

## Health Check

```bash
curl http://localhost:8000/health
```

## Migrating Existing Data

If you have an existing database with embeddings and content, you can easily migrate using the embedding APIs:

### 1. Create Vector Store
First, create a vector store for your data:

```bash
curl -X POST \
  http://localhost:8000/v1/vector_stores \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Migrated Data",
    "metadata": {"source": "legacy_system"}
  }'
```

### 2. Batch Insert Embeddings
Use the batch endpoint to efficiently insert multiple embeddings:

```bash
curl -X POST \
  http://localhost:8000/v1/vector_stores/vs_your_id/embeddings/batch \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "embeddings": [
      {
        "content": "Your text content here",
        "embedding": [0.1, 0.2, 0.3, ...1536 dimensions...],
        "metadata": {"source_id": "doc_123", "category": "support"}
      }
    ]
  }'
```

### 3. Migration Script Example

Here's a Python script example for migrating from an existing database:

```python
import psycopg2
import requests
import json

# Connect to your existing database
conn = psycopg2.connect("your_existing_db_url")
cur = conn.cursor()

# Fetch existing data
cur.execute("SELECT content, embedding, metadata FROM your_table")
rows = cur.fetchall()

# Prepare batch data
embeddings = []
for content, embedding, metadata in rows:
    embeddings.append({
        "content": content,
        "embedding": embedding.tolist(),  # Convert numpy array to list
        "metadata": metadata or {}
    })

# Send batch to API
response = requests.post(
    "http://localhost:8000/v1/vector_stores/your_vector_store_id/embeddings/batch",
    headers={
        "Authorization": "Bearer your-api-key",
        "Content-Type": "application/json"
    },
    json={"embeddings": embeddings}
)

print(f"Migrated {len(embeddings)} embeddings")
```

## Manual Ingest & Metadata Workflow

Besides the web UI, several standalone scripts in this directory can ingest stories and/or attach
structured metadata tags directly against the running API (default `http://127.0.0.1:18001`). All
of them read defaults from `litellm-pgvector/.env` (or `../.env`) — set `SERVER_API_KEY`,
`STORY_METADATA_API_BASE`/`STORY_METADATA_API_KEY`, and `STORY_METADATA_LLM_*` there so you don't
have to pass everything on the command line.

### 1. Bulk-sync a folder tree of stories (`scripts/sync_fiction_vectorstores.py`)

Syncs `<fiction-root>/<topic>/source/*.txt` into one vector store per topic (creating stores as
needed), re-uploading text via `upload-text-files`. Re-running against a topic clears that store's
existing `embeddings` **and** `story_metadata` rows first (via `--database-url`), so stale tags
from removed/renamed files don't linger:

```bash
cd litellm-pgvector
python3 scripts/sync_fiction_vectorstores.py \
  --fiction-root /mnt/commandjobs/fiction \
  --api-base http://127.0.0.1:18001 \
  --api-key "$SERVER_API_KEY" \
  --database-url "$DATABASE_URL" \
  --dry-run   # drop --dry-run to actually sync
```

Writes a `topic -> vector_store_id` mapping to `--mapping-file` (default
`/mnt/commandjobs/fiction/.vectorstore_ids.json`).

### 2. Extract + upload metadata for one story (`story_metadata_extractor.py`)

Runs an LLM extraction pass over a single `.txt` file against the `StoryMetadata` schema
(`models.py`) and, with `--upload`, `PUT`s the result to
`/v1/vector_stores/{vector_store_id}/stories/{filename}/metadata`:

```bash
python3 story_metadata_extractor.py path/to/story.txt \
  --vector-store-id vs_abc123 \
  --upload \
  --output extracted.json
```

### 3. Batch-extract metadata for a whole directory (`story_metadata_batch_runner.py`)

Same extraction as above, run over every `.txt` file in a directory, writing one JSON file per
story and optionally uploading each:

```bash
python3 story_metadata_batch_runner.py path/to/story-dir \
  --output-dir extracted-story-metadata \
  --vector-store-id vs_abc123 \
  --upload \
  --limit 50   # 0 = no limit
```

### 4. Full 3-part ingest prototype (`story_ingest_orchestrator.py`)

Runs scene splitting, story-graph extraction, and whole-story metadata extraction together for one
file or a directory, writing `metadata.json` / `scenes.json` / `graph.json` per story under
`--output-dir`, with the same `--upload`/`--vector-store-id` metadata upload option:

```bash
python3 story_ingest_orchestrator.py path/to/story-dir \
  --output-dir story-ingest-output \
  --model gpt-4.1-mini \
  --upload --vector-store-id vs_abc123
```

### 5. Spot-check extracted tags (`story_metadata_qa_sample.py`)

Prints a random sample of already-extracted JSON files (from either batch script above) for manual
QA before/instead of uploading:

```bash
python3 story_metadata_qa_sample.py extracted-story-metadata --count 10
```

### 6. Bulk-upsert pre-tagged metadata

If tags were generated offline (e.g. from an external tagging pass), push them directly without
re-running extraction, one request per vector store:

```bash
curl -X POST \
  http://127.0.0.1:18001/v1/vector_stores/vs_abc123/stories/metadata/bulk \
  -H "Authorization: Bearer $SERVER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"items": {"story-one.txt": {"genres": ["romance"], "heat_level": 3}}}'
```

## License

MIT License
