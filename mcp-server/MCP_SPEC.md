# Fiction Scene Search — MCP Server Spec

## Overview

| | |
|---|---|
| Server name | `fiction-scene-search` |
| Protocol | Model Context Protocol (MCP), 2025-06-18 revision |
| SDK | `mcp` Python SDK v1.28.1 (`mcp.server.fastmcp.FastMCP`) |
| Transport | Streamable HTTP (`streamable-http`) |
| Endpoint | `http://<host>:9913/mcp` |
| Auth | `Authorization: Bearer <MCP_BEARER_TOKEN>` (optional but strongly recommended — required whenever bound to a non-localhost interface) |
| Backing store | Wraps `POST /v1/scenes/search` on the litellm-pgvector API (`http://127.0.0.1:18001`) |

The server holds no content-access logic of its own. Every tool call is
proxied to the vector-store API, which enforces the single allowlist:
a vector store is only searchable once an admin sets
`api_exposed=true` on it (via the admin UI at `/ui` or
`PATCH /v1/vector_stores/{id}/exposure`). Currently only the `gay-military`
store is opted in.

## Transport details

- Implementation: FastMCP's `streamable_http_app()` (a Starlette ASGI app),
  served with `uvicorn`.
- Path: `/mcp` (FastMCP default `streamable_http_path`).
- Session handling: stateful streamable-HTTP session manager (default
  FastMCP behavior — a `Mcp-Session-Id` header is issued on `initialize`
  and must be echoed on subsequent requests, handled automatically by
  compliant MCP clients).
- Content types: JSON-RPC 2.0 messages over HTTP POST, with
  `Accept: application/json, text/event-stream` for streaming responses.

## Auth

If `MCP_BEARER_TOKEN` is set in the server's environment, every HTTP
request must include:

```
Authorization: Bearer <MCP_BEARER_TOKEN>
```

Requests without a matching header receive `401 Unauthorized`. If the
variable is unset, the server starts without an auth check and prints a
warning to stderr — only safe when bound to `127.0.0.1`.

## Capabilities

All tool names are prefixed with `VS_`. Every search tool shares the same
result shape and is proxied to `POST /v1/scenes/search`; they differ only in
how the input is framed into a query.

### Tools

#### `VS_List_Libraries`

Lists vector stores currently opted in for search.

- **Input:** none
- **Output:** JSON array of:
  ```json
  { "id": "string", "name": "string", "file_count": 0 }
  ```

#### `VS_Search_Scene`

Natural-language scene search across opted-in libraries.

- **Input:**
  | field | type | required | default | notes |
  |---|---|---|---|---|
  | `query` | string | yes | — | e.g. `"a scene with a military guy in it"` |
  | `limit` | integer | no | 10 | clamped to 1–100 |
- **Output:** JSON array of scene result objects (see below).

#### `VS_Search_Act`

Search for scenes depicting a specific sexual act.

- **Input:**
  | field | type | required | default | notes |
  |---|---|---|---|---|
  | `act` | string | yes | — | e.g. `"oral"`, `"anal"`, `"bondage"` |
  | `limit` | integer | no | 10 | clamped to 1–100 |
- **Output:** JSON array of scene result objects (see below).

#### `VS_Search_KeyWords`

Search by a general keyword or category (e.g. `"Marine"`, `"Army"`, `"College"`).

- **Input:**
  | field | type | required | default | notes |
  |---|---|---|---|---|
  | `keyword` | string | yes | — | e.g. `"Marine"` |
  | `limit` | integer | no | 10 | clamped to 1–100 |
- **Output:** JSON array of scene result objects (see below).

#### Scene result object

```json
{
  "library": "string",
  "filename": "string",
  "score": 0.0,
  "excerpt": "string"
}
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `SCENE_SEARCH_API_BASE` | `http://127.0.0.1:18001` | Base URL of the litellm-pgvector API |
| `SCENE_SEARCH_API_KEY` | *(required)* | Bearer token for that API |
| `MCP_TRANSPORT` | `streamable-http` | `streamable-http` or `stdio` |
| `MCP_HOST` | `0.0.0.0` | Bind host for streamable-http |
| `MCP_PORT` | `9913` | Bind port for streamable-http |
| `MCP_BEARER_TOKEN` | *(unset)* | Shared secret required from MCP clients |

## Running

```bash
cd mcp-server
MCP_BEARER_TOKEN="$(cat .mcp_bearer_token.txt)" \
SCENE_SEARCH_API_KEY="<server_api_key from ../.env>" \
.venv/bin/python scene_search_mcp.py
```

Or over stdio (e.g. for Claude Desktop / VS Code local process integration):

```bash
MCP_TRANSPORT=stdio SCENE_SEARCH_API_KEY="<key>" .venv/bin/python scene_search_mcp.py
```

## Client registration example (streamable-http)

```json
{
  "mcpServers": {
    "fiction-scene-search": {
      "url": "http://<host>:9913/mcp",
      "headers": { "Authorization": "Bearer <MCP_BEARER_TOKEN>" }
    }
  }
}
```
