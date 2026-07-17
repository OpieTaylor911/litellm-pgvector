#!/usr/bin/env python3
"""MCP server exposing cross-store fiction "scene search" as a tool.

This server is a thin wrapper around the existing vector-store API's
POST /v1/scenes/search endpoint. That endpoint only ever searches vector
stores that have been explicitly opted in via
PATCH /v1/vector_stores/{id}/exposure (api_exposed=true) -- everything else
stays private. This server does not add any of its own filtering; it relies
entirely on the API-side allowlist so there is exactly one place that
controls what content is reachable.

Configuration is via environment variables:
  SCENE_SEARCH_API_BASE  Base URL of the vector-store API (default: http://127.0.0.1:18001)
  SCENE_SEARCH_API_KEY   Bearer token for the vector-store API (required)
  MCP_TRANSPORT          "stdio" or "streamable-http" (default: streamable-http)
  MCP_HOST               Bind host for streamable-http (default: 0.0.0.0)
  MCP_PORT               Bind port for streamable-http (default: 9913)
  MCP_BEARER_TOKEN       If set, clients must send `Authorization: Bearer <token>`
                         to reach this MCP server over streamable-http.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

API_BASE = os.environ.get("SCENE_SEARCH_API_BASE", "http://127.0.0.1:18001").rstrip("/")
API_KEY = os.environ.get("SCENE_SEARCH_API_KEY", "")

MCP_HOST = os.environ.get("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.environ.get("MCP_PORT", "9913"))
MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", "streamable-http")
MCP_BEARER_TOKEN = os.environ.get("MCP_BEARER_TOKEN", "")

mcp = FastMCP("fiction-scene-search", host=MCP_HOST, port=MCP_PORT)


def _client() -> httpx.Client:
    if not API_KEY:
        raise RuntimeError(
            "SCENE_SEARCH_API_KEY is not set. Configure it in the MCP server's "
            "environment before starting."
        )
    return httpx.Client(
        base_url=API_BASE,
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=30.0,
    )


def _search_backend(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Shared helper: run a natural-language query against
    POST /v1/scenes/search and normalize the response shape."""
    payload = {"query": query, "limit": max(1, min(limit, 100))}

    with _client() as client:
        response = client.post("/v1/scenes/search", json=payload)
        response.raise_for_status()
        results = response.json().get("data", [])

    scenes = []
    for item in results:
        text = "\n".join(chunk.get("text", "") for chunk in item.get("content", []))
        scenes.append(
            {
                "library": item.get("vector_store_name"),
                "filename": item.get("filename"),
                "score": item.get("score"),
                "excerpt": text,
            }
        )
    return scenes


@mcp.tool()
def VS_List_Libraries() -> list[dict[str, Any]]:
    """List the story libraries (vector stores) that are currently opted in
    for search. Only libraries returned here can be searched -- everything
    else has been intentionally kept private by the administrator."""
    with _client() as client:
        response = client.get("/v1/vector_stores", params={"limit": 100})
        response.raise_for_status()
        stores = response.json().get("data", [])

    return [
        {
            "id": store["id"],
            "name": store["name"],
            "file_count": (store.get("file_counts") or {}).get("total", 0),
        }
        for store in stores
        if store.get("api_exposed")
    ]


@mcp.tool()
def VS_Search_Scene(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search story libraries for scenes matching a natural-language
    description, e.g. "a scene with a military guy in it". Only searches
    libraries that have been explicitly made searchable by the administrator.

    Args:
        query: Natural-language description of the scene you're looking for.
        limit: Maximum number of matching scene excerpts to return (default 10, max 100).
    """
    return _search_backend(query, limit)


@mcp.tool()
def VS_Search_Act(act: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search story libraries for scenes depicting a specific sexual act,
    e.g. "oral" or "anal". Only searches libraries that have been explicitly
    made searchable by the administrator.

    Args:
        act: The sexual act to search for (e.g. "oral", "anal", "bondage").
        limit: Maximum number of matching scene excerpts to return (default 10, max 100).
    """
    return _search_backend(f"a scene depicting {act}", limit)


@mcp.tool()
def VS_Search_KeyWords(keyword: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search story libraries by a general keyword or category, e.g.
    "Marine", "Army", "College". Only searches libraries that have been
    explicitly made searchable by the administrator.

    Args:
        keyword: The keyword or category to search for (e.g. "Marine", "Army", "College").
        limit: Maximum number of matching scene excerpts to return (default 10, max 100).
    """
    return _search_backend(keyword, limit)


def _build_http_app():
    """Wrap the FastMCP streamable-http ASGI app with an optional bearer-token
    auth check, so this can be safely bound to 0.0.0.0 for other applications
    to reach over the network."""
    app = mcp.streamable_http_app()

    if not MCP_BEARER_TOKEN:
        return app

    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    class BearerAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            header = request.headers.get("authorization", "")
            expected = f"Bearer {MCP_BEARER_TOKEN}"
            if header != expected:
                return JSONResponse({"error": "Unauthorized"}, status_code=401)
            return await call_next(request)

    app.add_middleware(BearerAuthMiddleware)
    return app


if __name__ == "__main__":
    if MCP_TRANSPORT == "streamable-http":
        import uvicorn

        if not MCP_BEARER_TOKEN:
            print(
                "WARNING: MCP_BEARER_TOKEN is not set; this MCP server is reachable "
                f"by anyone who can connect to {MCP_HOST}:{MCP_PORT}. Set "
                "MCP_BEARER_TOKEN to require an Authorization header.",
            )
        uvicorn.run(_build_http_app(), host=MCP_HOST, port=MCP_PORT)
    else:
        mcp.run(transport=MCP_TRANSPORT)
