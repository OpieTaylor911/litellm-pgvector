#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

load_env() {
  if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$PROJECT_DIR/.env"
    set +a
  fi
}

normalize_database_url() {
  python3 - "$1" <<'PY'
import sys
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

database_url = sys.argv[1]
parts = urlsplit(database_url)
filtered_query = [
    (key, value)
    for key, value in parse_qsl(parts.query, keep_blank_values=True)
    if key.lower() != "schema"
]
print(urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(filtered_query), parts.fragment)))
PY
}

list_vector_stores() {
  python3 - "$API_BASE" "$API_KEY" <<'PY'
import json
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen

api_base = sys.argv[1].rstrip('/')
api_key = sys.argv[2]
stores = []
after = None

while True:
    params = {"limit": 100}
    if after:
        params["after"] = after
    url = f"{api_base}/v1/vector_stores?{urlencode(params)}"
    request = Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    batch = payload.get("data", [])
    stores.extend(batch)
    if not payload.get("has_more") or not batch:
        break
    after = payload.get("last_id")
    if not after:
        break

for store in stores:
    print(f"{store['name']}\t{store['id']}")
PY
}

create_vector_store() {
  python3 - "$API_BASE" "$API_KEY" "$1" <<'PY'
import json
import sys
from urllib.request import Request, urlopen

api_base = sys.argv[1].rstrip('/')
api_key = sys.argv[2]
topic = sys.argv[3]
url = f"{api_base}/v1/vector_stores"
payload = json.dumps({
    "name": topic,
    "metadata": {"topic": topic, "source": "fiction-sync-shell"},
}).encode("utf-8")
request = Request(
    url,
    data=payload,
    method="POST",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    },
)
with urlopen(request, timeout=60) as response:
    row = json.loads(response.read().decode("utf-8"))
print(row["id"])
PY
}

probe_embedding_model() {
  local model="$1"
  local base_url="$2"
  local api_key="$3"

  python3 - "$model" "$base_url" "$api_key" <<'PY'
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

model = sys.argv[1]
base_url = sys.argv[2].rstrip("/")
api_key = sys.argv[3]

url = f"{base_url}/embeddings"
payload = json.dumps({"model": model, "input": "healthcheck"}).encode("utf-8")
headers = {
  "Authorization": f"Bearer {api_key}",
  "Content-Type": "application/json",
  "Accept": "application/json",
}
request = Request(url, data=payload, method="POST", headers=headers)

try:
  with urlopen(request, timeout=20) as response:
    body = response.read().decode("utf-8")
  parsed = json.loads(body)
  if not parsed.get("data"):
    print(f"embedding preflight failed: no data returned for model={model}", file=sys.stderr)
    sys.exit(1)
  print(f"embedding preflight ok: model={model}")
except HTTPError as err:
  body = err.read().decode("utf-8", errors="replace")
  print(f"embedding preflight failed: HTTP {err.code} from {url}: {body}", file=sys.stderr)
  sys.exit(1)
except URLError as err:
  print(f"embedding preflight failed: endpoint unreachable at {url}: {err}", file=sys.stderr)
  sys.exit(1)
except Exception as err:
  print(f"embedding preflight failed: unexpected error: {err}", file=sys.stderr)
  sys.exit(1)
PY
}

fetch_existing_story_filenames() {
  local normalized_database_url="$1"
  local vector_store_id="$2"

  PGPASSWORD=${PGPASSWORD:-} psql -At -d "$normalized_database_url" -v ON_ERROR_STOP=1 -c \
    "SELECT filename FROM story_metadata WHERE vector_store_id = '${vector_store_id}' ORDER BY filename;"
}

upload_new_files() {
  local vector_store_id="$1"
  shift

  local files=("$@")
  local total=${#files[@]}
  local offset=0
  local uploaded=0

  while [[ $offset -lt $total ]]; do
    local batch=("${files[@]:$offset:$MAX_FILES_PER_REQUEST}")
    local attempt=1
    local success=0
    local response_file
    response_file=$(mktemp)
    trap 'rm -f "$response_file"' RETURN

    while [[ $attempt -le $UPLOAD_RETRIES ]]; do
      local curl_args=(
        -sS
        --max-time "$UPLOAD_MAX_TIME_SECONDS"
        -X POST
        "$API_BASE/v1/vector_stores/upload-text-files"
        -H "Authorization: Bearer $API_KEY"
        -F "vector_store_id=$vector_store_id"
        -F "chunk_size=$CHUNK_SIZE"
        -F "chunk_overlap=$CHUNK_OVERLAP"
      )

      local path
      for path in "${batch[@]}"; do
        curl_args+=(-F "files=@${path};type=text/plain")
      done

      local http_code="000"
      if http_code=$(curl "${curl_args[@]}" -o "$response_file" -w "%{http_code}"); then
        if [[ "$http_code" =~ ^2 ]]; then
          cat "$response_file"
          echo
          rm -f "$response_file"
          response_file=$(mktemp)
          trap 'rm -f "$response_file"' RETURN
          success=1
          break
        fi

        echo "upload batch failed (attempt ${attempt}/${UPLOAD_RETRIES}); status=${http_code}" >&2
        head -c 400 "$response_file" >&2 || true
        echo >&2
      else
        echo "upload batch failed (attempt ${attempt}/${UPLOAD_RETRIES}); curl transport error" >&2
      fi

      rm -f "$response_file"
      response_file=$(mktemp)
      trap 'rm -f "$response_file"' RETURN

      attempt=$((attempt + 1))
    done

    rm -f "$response_file"
    trap - RETURN

    if [[ $success -ne 1 ]]; then
      echo "ERROR: failed to upload batch starting at file index ${offset}" >&2
      return 1
    fi

    uploaded=$((uploaded + ${#batch[@]}))
    offset=$((offset + MAX_FILES_PER_REQUEST))
    echo "progress: uploaded ${uploaded}/${total} files for store ${vector_store_id}"
  done
}

load_env

FIC_ROOT=${FIC_ROOT:-/mnt/commandjobs/fiction}
API_BASE=${API_BASE:-http://127.0.0.1:18001}
API_KEY=${API_KEY:-${SERVER_API_KEY:-}}
DATABASE_URL=${DATABASE_URL:-}
CHUNK_SIZE=${CHUNK_SIZE:-1600}
CHUNK_OVERLAP=${CHUNK_OVERLAP:-200}
SKIP_EMPTY=${SKIP_EMPTY:-1}
TOPIC_FILTER=${TOPIC_FILTER:-}
MAX_FILES_PER_REQUEST=${MAX_FILES_PER_REQUEST:-20}
UPLOAD_RETRIES=${UPLOAD_RETRIES:-3}
UPLOAD_MAX_TIME_SECONDS=${UPLOAD_MAX_TIME_SECONDS:-1800}
PRECHECK_EMBEDDING=${PRECHECK_EMBEDDING:-1}

EMBED_MODEL=${EMBEDDING__MODEL:-}
EMBED_BASE_URL=${EMBEDDING__BASE_URL:-}
EMBED_API_KEY=${EMBEDDING__API_KEY:-}

if [[ -z "$API_KEY" ]]; then
  echo "ERROR: API_KEY or SERVER_API_KEY must be set" >&2
  exit 1
fi

if [[ -z "$DATABASE_URL" ]]; then
  echo "ERROR: DATABASE_URL must be set" >&2
  exit 1
fi

if [[ ! -d "$FIC_ROOT" ]]; then
  echo "ERROR: fiction root not found: $FIC_ROOT" >&2
  exit 1
fi

if [[ "$CHUNK_OVERLAP" -ge "$CHUNK_SIZE" ]]; then
  echo "ERROR: CHUNK_OVERLAP must be smaller than CHUNK_SIZE" >&2
  exit 1
fi

if [[ "$PRECHECK_EMBEDDING" == "1" ]]; then
  if [[ -z "$EMBED_MODEL" || -z "$EMBED_BASE_URL" || -z "$EMBED_API_KEY" ]]; then
    echo "WARN: skipping embedding preflight (EMBEDDING__MODEL/BASE_URL/API_KEY not fully set)" >&2
  else
    probe_embedding_model "$EMBED_MODEL" "$EMBED_BASE_URL" "$EMBED_API_KEY"
  fi
fi

NORMALIZED_DATABASE_URL=$(normalize_database_url "$DATABASE_URL")
declare -A STORE_IDS=()
while IFS=$'\t' read -r name id; do
  [[ -n "$name" && -n "$id" ]] || continue
  STORE_IDS["$name"]="$id"
done < <(list_vector_stores)

topic_count=0
uploaded_count=0
skipped_count=0

for topic_dir in "$FIC_ROOT"/*; do
  [[ -d "$topic_dir/source" ]] || continue

  if [[ -n "$TOPIC_FILTER" ]]; then
    topic_name=$(basename "$topic_dir")
    [[ "$topic_name" == "$TOPIC_FILTER" ]] || continue
  fi

  topic_count=$((topic_count + 1))
  topic=$(basename "$topic_dir")
  source_dir="$topic_dir/source"

  mapfile -t txt_files < <(find "$source_dir" -maxdepth 1 -type f -name '*.txt' | sort)
  if [[ ${#txt_files[@]} -eq 0 && "$SKIP_EMPTY" -eq 1 ]]; then
    echo "[$topic] no .txt files, skipping"
    continue
  fi

  if [[ -n "${STORE_IDS[$topic]+x}" ]]; then
    vector_store_id="${STORE_IDS[$topic]}"
  else
    vector_store_id=$(create_vector_store "$topic")
    STORE_IDS["$topic"]="$vector_store_id"
    echo "[$topic] created vector store: $vector_store_id"
  fi

  declare -A existing_files=()
  while IFS= read -r filename; do
    [[ -n "$filename" ]] || continue
    existing_files["$filename"]=1
  done < <(fetch_existing_story_filenames "$NORMALIZED_DATABASE_URL" "$vector_store_id")

  new_files=()
  for path in "${txt_files[@]}"; do
    filename=$(basename "$path")
    if [[ -z "${existing_files[$filename]+x}" ]]; then
      new_files+=("$path")
    fi
  done

  if [[ ${#new_files[@]} -eq 0 ]]; then
    echo "[$topic] no new files to upload"
    skipped_count=$((skipped_count + 1))
    continue
  fi

  echo "[$topic] uploading ${#new_files[@]} new file(s) to $vector_store_id"
  upload_new_files "$vector_store_id" "${new_files[@]}"
  uploaded_count=$((uploaded_count + ${#new_files[@]}))
done

echo "done: topics=$topic_count uploaded_files=$uploaded_count skipped_topics=$skipped_count"
