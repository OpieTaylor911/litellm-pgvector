# Fiction Scene Search MCP Server

Exposes the vector-store API's cross-library scene search as MCP tools so
other applications (Claude Desktop, VS Code, etc.) can ask things like
*"find a scene with a military guy in it"*.

## How exposure works

This server calls `POST /v1/scenes/search` on the existing vector-store API.
That endpoint only searches vector stores that an admin has explicitly
opted in via the checkbox in the admin UI (`/ui`) or:

```
curl -X PATCH "$API_BASE/v1/vector_stores/<id>/exposure" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"api_exposed": true}'
```

Stores default to **not exposed**. There is exactly one allowlist (the
`api_exposed` flag on each vector store); this MCP server does not add or
need any additional filtering of its own.

## Running

Full protocol details are in [MCP_SPEC.md](MCP_SPEC.md).

By default the server listens over **streamable HTTP on port 9913**:

```bash
MCP_BEARER_TOKEN="<shared secret for MCP clients>" \
SCENE_SEARCH_API_BASE="http://127.0.0.1:18001" \
SCENE_SEARCH_API_KEY="<server_api_key from .env>" \
.venv/bin/python scene_search_mcp.py
```

It is currently running as a background process reachable at
`http://<host>:9913/mcp`. To stop/restart it, find the process with
`ss -ltnp | grep 9913` and send it a signal, then relaunch with the command
above.

To run over stdio instead (e.g. Claude Desktop / VS Code local process
integration):

```bash
MCP_TRANSPORT=stdio SCENE_SEARCH_API_KEY="<key>" .venv/bin/python scene_search_mcp.py
```

## Registering with an MCP client

Streamable HTTP (recommended):

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

stdio (local process):

```json
{
  "mcpServers": {
    "fiction-scene-search": {
      "command": "/opt/pgvector/litellm-pgvector/mcp-server/.venv/bin/python",
      "args": ["/opt/pgvector/litellm-pgvector/mcp-server/scene_search_mcp.py"],
      "env": {
        "MCP_TRANSPORT": "stdio",
        "SCENE_SEARCH_API_BASE": "http://127.0.0.1:18001",
        "SCENE_SEARCH_API_KEY": "<server_api_key from .env>"
      }
    }
  }
}
```

## Tools

All tools are prefixed with `VS_`:

- `VS_List_Libraries()` – lists the libraries currently opted in for search.
- `VS_Search_Scene(query, limit=10)` – natural-language scene search across opted-in libraries.
- `VS_Search_Act(act, limit=10)` – search for scenes depicting a specific sexual act (e.g. "oral", "anal").
- `VS_Search_KeyWords(keyword, limit=10)` – search by a general keyword/category (e.g. "Marine", "Army", "College").
