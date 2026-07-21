#!/usr/bin/env python3
"""MCP server exposing cross-store fiction scene search and operations tools."""

from __future__ import annotations

import json
import os
import re
import signal
import time
from pathlib import Path
from typing import Any, Optional

import httpx
from mcp.server.fastmcp import FastMCP

API_BASE = os.environ.get("SCENE_SEARCH_API_BASE", "http://127.0.0.1:18001").rstrip("/")
API_KEY = os.environ.get("SCENE_SEARCH_API_KEY", "")

MCP_HOST = os.environ.get("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.environ.get("MCP_PORT", "9913"))
MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", "streamable-http")
MCP_BEARER_TOKEN = os.environ.get("MCP_BEARER_TOKEN", "")

mcp = FastMCP("fiction-scene-search", host=MCP_HOST, port=MCP_PORT)

SAVED_SEARCHES_PATH = Path(__file__).resolve().parent / ".saved_searches.json"
SAVED_SEARCH_HISTORY_PATH = Path(__file__).resolve().parent / ".saved_search_history.json"
STARTED_INGEST_JOBS: dict[str, dict[str, Any]] = {}

INGEST_COOLDOWN_SECONDS = int(os.environ.get("MCP_INGEST_COOLDOWN_SECONDS", "120"))
INGEST_PER_TOPIC_WINDOW_SECONDS = int(os.environ.get("MCP_INGEST_RATE_WINDOW_SECONDS", "600"))
INGEST_PER_TOPIC_MAX_IN_WINDOW = int(os.environ.get("MCP_INGEST_RATE_MAX", "2"))
SEARCH_ZERO_RESULT_ALERT_THRESHOLD = int(os.environ.get("MCP_SEARCH_ZERO_RESULT_ALERT_THRESHOLD", "3"))


def _client() -> httpx.Client:
    if not API_KEY:
        raise RuntimeError("SCENE_SEARCH_API_KEY is not set")
    return httpx.Client(
        base_url=API_BASE,
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=30.0,
    )


def _api_get(path: str, *, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    with _client() as client:
        response = client.get(path, params=params)
        response.raise_for_status()
        return response.json()


def _api_post(path: str, *, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    with _client() as client:
        response = client.post(path, json=(payload or {}))
        response.raise_for_status()
        return response.json()


def _api_patch(path: str, *, payload: dict[str, Any]) -> dict[str, Any]:
    with _client() as client:
        response = client.patch(path, json=payload)
        response.raise_for_status()
        return response.json()


def _search_backend(query: str, limit: int = 10) -> list[dict[str, Any]]:
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
                "vector_store_id": item.get("vector_store_id"),
                "filename": item.get("filename"),
                "score": item.get("score"),
                "excerpt": text,
                "story_metadata": item.get("story_metadata"),
            }
        )
    return scenes


def _list_all_libraries() -> list[dict[str, Any]]:
    return _api_get("/v1/vector_stores", params={"limit": 100}).get("data", [])


def _list_exposed_libraries() -> list[dict[str, Any]]:
    return [store for store in _list_all_libraries() if store.get("api_exposed")]


def _resolve_single_library(library: str, *, require_exposed: bool = False) -> dict[str, Any]:
    needle = library.strip().lower()
    if not needle:
        raise ValueError("library is required")

    stores = _list_exposed_libraries() if require_exposed else _list_all_libraries()
    exact_name = [s for s in stores if str(s.get("name", "")).lower() == needle]
    exact_id = [s for s in stores if str(s.get("id", "")).lower() == needle]
    partial = [
        s
        for s in stores
        if needle in str(s.get("name", "")).lower() or needle in str(s.get("id", "")).lower()
    ]
    matches = exact_id or exact_name or partial
    if not matches:
        raise ValueError(f"No matching library found: {library}")
    if len(matches) > 1:
        names = ", ".join(f"{m.get('name')} ({str(m.get('id'))[:8]})" for m in matches[:6])
        raise ValueError(f"Ambiguous library '{library}'. Matches: {names}")
    return matches[0]


def _resolve_multiple_libraries(libraries: list[str], *, require_exposed: bool = False) -> list[dict[str, Any]]:
    seen: set[str] = set()
    resolved: list[dict[str, Any]] = []
    for item in libraries:
        store = _resolve_single_library(item, require_exposed=require_exposed)
        store_id = str(store.get("id") or "")
        if store_id and store_id not in seen:
            seen.add(store_id)
            resolved.append(store)
    return resolved


def _resolve_vector_store_ids(topic_or_library: Optional[str]) -> Optional[list[str]]:
    if not topic_or_library:
        return None

    needle = topic_or_library.strip().lower()
    if not needle:
        return None

    matches: list[str] = []
    for store in _list_exposed_libraries():
        name = str(store.get("name", "")).lower()
        store_id = str(store.get("id", ""))
        if not store_id:
            continue
        if needle == store_id.lower() or needle == name or needle in name:
            matches.append(store_id)

    if not matches:
        return []
    return sorted(set(matches))


def _list_story_files(vector_store_id: str) -> list[dict[str, Any]]:
    return _api_get(f"/v1/vector_stores/{vector_store_id}/stories")


def _is_default_story_metadata(metadata: Optional[dict[str, Any]]) -> bool:
    if not metadata:
        return True
    for value in metadata.values():
        if isinstance(value, bool) and value:
            return False
        if isinstance(value, int) and value != 1:
            return False
        if isinstance(value, str) and value.strip():
            return False
        if isinstance(value, list) and value:
            return False
    return True


def _compute_facets(files: list[dict[str, Any]]) -> dict[str, Any]:
    counters: dict[str, dict[str, int]] = {
        "genres": {},
        "settings": {},
        "military": {},
        "kinks": {},
        "tone": {},
    }
    bool_counts: dict[str, int] = {
        "coming_out": 0,
        "first_love": 0,
        "forbidden_romance": 0,
        "found_family": 0,
        "slow_burn": 0,
        "enemies_to_lovers": 0,
        "hurt_comfort": 0,
        "age_gap": 0,
        "college": 0,
        "military_romance": 0,
    }
    heat: dict[str, int] = {}

    def bump(counter: dict[str, int], key: str) -> None:
        if not key:
            return
        counter[key] = counter.get(key, 0) + 1

    for row in files:
        md = row.get("metadata") or {}
        for key in counters.keys():
            for tag in md.get(key) or []:
                bump(counters[key], str(tag).strip().lower())
        for key in bool_counts.keys():
            if md.get(key) is True:
                bool_counts[key] += 1
        heat_level = md.get("heat_level")
        if isinstance(heat_level, int):
            bump(heat, str(heat_level))

    return {
        "top_genres": sorted(counters["genres"].items(), key=lambda x: x[1], reverse=True)[:20],
        "top_settings": sorted(counters["settings"].items(), key=lambda x: x[1], reverse=True)[:20],
        "top_military": sorted(counters["military"].items(), key=lambda x: x[1], reverse=True)[:20],
        "top_kinks": sorted(counters["kinks"].items(), key=lambda x: x[1], reverse=True)[:20],
        "top_tone": sorted(counters["tone"].items(), key=lambda x: x[1], reverse=True)[:20],
        "heat_level_distribution": heat,
        "boolean_counts": bool_counts,
    }


def _rerank_results(
    results: list[dict[str, Any]],
    *,
    rerank: str = "none",
    preferred_library: Optional[str] = None,
) -> list[dict[str, Any]]:
    mode = (rerank or "none").strip().lower()
    if mode == "none":
        return results

    def score(item: dict[str, Any]) -> tuple[float, float]:
        base = float(item.get("score") or 0.0)
        md = item.get("story_metadata") or {}
        boost = 0.0
        if mode == "metadata_match":
            boost += 0.05 if md else 0.0
            boost += 0.03 if (md.get("genres") or []) else 0.0
            boost += 0.02 if (md.get("search_keywords") or []) else 0.0
        elif mode == "topic_priority":
            lib = str(item.get("library") or "").lower()
            pref = str(preferred_library or "").lower()
            if pref and pref in lib:
                boost += 0.15
        elif mode == "recency":
            boost += 0.0
        return (base + boost, base)

    return sorted(results, key=score, reverse=True)


def _search_backend_advanced(
    query: str,
    *,
    limit: int = 10,
    topic_or_library: Optional[str] = None,
    tag_filters: Optional[dict[str, Any]] = None,
    rerank: str = "none",
    preferred_library: Optional[str] = None,
) -> list[dict[str, Any]]:
    payload: dict[str, Any] = {
        "query": query,
        "limit": max(1, min(limit, 100)),
        "return_metadata": True,
    }

    scoped_ids = _resolve_vector_store_ids(topic_or_library)
    if scoped_ids == []:
        return []
    if scoped_ids:
        payload["vector_store_ids"] = scoped_ids

    if tag_filters:
        payload["tag_filters"] = tag_filters

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
                "vector_store_id": item.get("vector_store_id"),
                "filename": item.get("filename"),
                "score": item.get("score"),
                "excerpt": text,
                "story_metadata": item.get("story_metadata"),
            }
        )
    return _rerank_results(scenes, rerank=rerank, preferred_library=preferred_library)


def _load_saved_searches() -> dict[str, Any]:
    if not SAVED_SEARCHES_PATH.is_file():
        return {}
    try:
        return json.loads(SAVED_SEARCHES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_saved_searches(items: dict[str, Any]) -> None:
    SAVED_SEARCHES_PATH.write_text(json.dumps(items, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_saved_search_history() -> list[dict[str, Any]]:
    if not SAVED_SEARCH_HISTORY_PATH.is_file():
        return []
    try:
        data = json.loads(SAVED_SEARCH_HISTORY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_saved_search_history(entries: list[dict[str, Any]]) -> None:
    SAVED_SEARCH_HISTORY_PATH.write_text(
        json.dumps(entries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _record_saved_search_run(
    *,
    name: str,
    config: dict[str, Any],
    result_count: int,
    top_result: Optional[dict[str, Any]],
    latency_ms: Optional[float] = None,
    error: bool = False,
    error_message: Optional[str] = None,
    rerank_changed_top_result: Optional[bool] = None,
    rerank_order_delta_top10: Optional[int] = None,
) -> dict[str, Any]:
    entries = _load_saved_search_history()
    row = {
        "timestamp": int(time.time()),
        "name": name,
        "query": config.get("query"),
        "library": config.get("library"),
        "limit": config.get("limit"),
        "rerank": config.get("rerank"),
        "result_count": int(result_count),
        "latency_ms": round(float(latency_ms), 2) if latency_ms is not None else None,
        "error": bool(error),
        "error_message": error_message,
        "rerank_changed_top_result": rerank_changed_top_result,
        "rerank_order_delta_top10": rerank_order_delta_top10,
        "top_result": {
            "library": (top_result or {}).get("library"),
            "filename": (top_result or {}).get("filename"),
            "score": (top_result or {}).get("score"),
        }
        if top_result
        else None,
    }
    entries.append(row)
    max_entries = 1000
    if len(entries) > max_entries:
        entries = entries[-max_entries:]
    _save_saved_search_history(entries)
    return row


def _normalize_saved_search_name(name: str) -> str:
    clean = (name or "").strip()
    if not clean:
        raise ValueError("name is required")
    return clean


def _normalized_filename_stem(filename: Optional[str]) -> str:
    name = (filename or "").strip().lower()
    if not name:
        return ""
    stem = Path(name).stem
    stem = re.sub(r"[^a-z0-9]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem


def _compute_order_delta_topk(base: list[dict[str, Any]], reranked: list[dict[str, Any]], k: int = 10) -> int:
    base_top = base[:k]
    reranked_top = reranked[:k]
    base_pos = {
        f"{row.get('library')}::{row.get('filename')}": idx
        for idx, row in enumerate(base_top)
    }
    rerank_pos = {
        f"{row.get('library')}::{row.get('filename')}": idx
        for idx, row in enumerate(reranked_top)
    }
    overlap = set(base_pos.keys()) & set(rerank_pos.keys())
    return sum(abs(base_pos[key] - rerank_pos[key]) for key in overlap)


def _list_active_ingest_jobs() -> list[dict[str, Any]]:
    try:
        summary = _api_get("/health/summary")
    except Exception:
        return []
    return summary.get("active_ingest_jobs", [])


def _augment_jobs_with_queue_positions(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queued = [j for j in jobs if str(j.get("status") or "").lower() == "queued"]
    queued.sort(key=lambda j: int(j.get("queued_at") or 0))
    queue_pos_by_id = {
        str(j.get("job_id")): idx + 1
        for idx, j in enumerate(queued)
        if j.get("job_id")
    }
    queued_total = len(queued)
    running_total = sum(1 for j in jobs if str(j.get("status") or "").lower() == "running")

    out: list[dict[str, Any]] = []
    for job in jobs:
        row = dict(job)
        job_id = str(row.get("job_id") or "")
        queue_position = queue_pos_by_id.get(job_id)
        row["queue_position"] = queue_position
        row["queued_ahead"] = (queue_position - 1) if queue_position else 0
        row["queued_total"] = queued_total
        row["running_total"] = running_total
        out.append(row)
    return out


def _evaluate_ingest_guardrails(store: dict[str, Any]) -> dict[str, Any]:
    topic_name = str(store.get("name") or "")
    store_id = str(store.get("id") or "")
    now_ts = int(time.time())

    active = _augment_jobs_with_queue_positions(_list_active_ingest_jobs())
    topic_active = [
        job
        for job in active
        if str(job.get("vector_store_id") or "") == store_id
        or str(job.get("store_name") or "").lower() == topic_name.lower()
    ]
    if topic_active:
        topic_active.sort(
            key=lambda j: (
                int(j.get("queue_position") or 999999),
                int(j.get("queued_at") or 0),
            )
        )
        active_job = topic_active[0]
        job_id = str(active_job.get("job_id") or "")
        status = str(active_job.get("status") or "")
        if status == "queued":
            recommended_next = f'VS_Get_Ingest_Status(job_id="{job_id}", topic="{topic_name}")'
        else:
            recommended_next = f'VS_Get_Ingest_Log(job_id="{job_id}", lines=100)'
        return {
            "allowed": False,
            "reason": "active_job_exists",
            "message": "A job for this topic is already active.",
            "topic": topic_name,
            "job": active_job,
            "active_jobs": topic_active,
            "recommended_next_command": recommended_next,
        }

    recent_starts: list[dict[str, Any]] = []
    for data in STARTED_INGEST_JOBS.values():
        if str(data.get("vector_store_id") or "") != store_id:
            continue
        started_at = int(data.get("started_at") or 0)
        if started_at > 0:
            recent_starts.append(data)
    recent_starts.sort(key=lambda d: int(d.get("started_at") or 0), reverse=True)

    if recent_starts:
        latest = int(recent_starts[0].get("started_at") or 0)
        elapsed = now_ts - latest
        if elapsed < max(0, INGEST_COOLDOWN_SECONDS):
            return {
                "allowed": False,
                "reason": "cooldown",
                "message": "Topic ingest is in cooldown window.",
                "retry_after_seconds": max(0, INGEST_COOLDOWN_SECONDS - elapsed),
                "recommended_next_command": f'VS_List_Active_Ingests()  # then VS_Get_Ingest_Status(..., topic="{topic_name}")',
            }

    window_seconds = max(1, INGEST_PER_TOPIC_WINDOW_SECONDS)
    window_start = now_ts - window_seconds
    starts_in_window = [d for d in recent_starts if int(d.get("started_at") or 0) >= window_start]
    if len(starts_in_window) >= max(1, INGEST_PER_TOPIC_MAX_IN_WINDOW):
        oldest_kept = min(int(d.get("started_at") or now_ts) for d in starts_in_window)
        retry_after = max(0, window_seconds - (now_ts - oldest_kept))
        return {
            "allowed": False,
            "reason": "rate_limited",
            "message": "Topic ingest rate limit exceeded in current window.",
            "retry_after_seconds": retry_after,
            "window_seconds": window_seconds,
            "max_in_window": max(1, INGEST_PER_TOPIC_MAX_IN_WINDOW),
            "starts_in_window": len(starts_in_window),
            "recommended_next_command": f'VS_Backfill_Metadata_For_Topic(topic="{topic_name}", dry_run=true)',
        }

    return {"allowed": True, "reason": "ok"}


def _detect_zero_result_alert(name: str, threshold: int) -> Optional[dict[str, Any]]:
    if threshold <= 0:
        return None
    entries = _load_saved_search_history()
    same = [row for row in entries if str(row.get("name", "")).lower() == name.lower()]
    same.sort(key=lambda r: int(r.get("timestamp") or 0), reverse=True)
    if len(same) < threshold:
        return None
    latest = same[:threshold]
    if all(int(row.get("result_count") or 0) == 0 for row in latest):
        return {
            "triggered": True,
            "threshold": threshold,
            "consecutive_zero_results": threshold,
            "latest_timestamps": [int(r.get("timestamp") or 0) for r in latest],
            "message": f"Saved search '{name}' returned zero results for the last {threshold} runs.",
        }
    return None


def _recommend_actions_from_readiness(readiness: dict[str, Any]) -> list[str]:
    checks = readiness.get("checks") or {}
    recs: list[str] = []
    if not checks.get("is_exposed"):
        recs.append("Set api_exposed=true to make this topic searchable via MCP.")
    if not checks.get("has_files"):
        recs.append("Run VS_Start_Ingest for this topic to load story files.")
    if checks.get("has_files") and not checks.get("has_chunks"):
        recs.append("Re-run ingest and inspect VS_Get_Ingest_Log for chunking/embedding failures.")
    if checks.get("has_files") and not checks.get("has_any_metadata"):
        recs.append("Run VS_Backfill_Metadata_For_Topic(dry_run=false) to remediate metadata gaps.")
    if not recs:
        recs.append("No action needed; topic appears ready.")
    return recs


@mcp.tool()
def VS_List_Libraries() -> list[dict[str, Any]]:
    return [
        {
            "id": store["id"],
            "name": store["name"],
            "file_count": (store.get("file_counts") or {}).get("total", 0),
        }
        for store in _list_exposed_libraries()
    ]


@mcp.tool()
def VS_List_Topics(exposed_only: bool = True) -> list[dict[str, Any]]:
    stores = _list_exposed_libraries() if exposed_only else _list_all_libraries()
    return [
        {
            "id": s.get("id"),
            "topic": s.get("name"),
            "file_count": (s.get("file_counts") or {}).get("total", 0),
            "api_exposed": bool(s.get("api_exposed")),
        }
        for s in stores
    ]


@mcp.tool()
def VS_Search_Scene(query: str, limit: int = 10) -> list[dict[str, Any]]:
    return _search_backend(query, limit)


@mcp.tool()
def VS_Search_Act(act: str, limit: int = 10) -> list[dict[str, Any]]:
    return _search_backend(f"a scene depicting {act}", limit)


@mcp.tool()
def VS_Search_KeyWords(keyword: str, limit: int = 10) -> list[dict[str, Any]]:
    return _search_backend(keyword, limit)


@mcp.tool()
def VS_Search_By_Library(query: str, library: str, limit: int = 10) -> list[dict[str, Any]]:
    return _search_backend_advanced(query, limit=limit, topic_or_library=library)


@mcp.tool()
def VS_Search_By_Metadata(
    query: str,
    limit: int = 10,
    library: Optional[str] = None,
    genres: Optional[list[str]] = None,
    settings: Optional[list[str]] = None,
    military: Optional[list[str]] = None,
    kinks: Optional[list[str]] = None,
    any_keyword: Optional[list[str]] = None,
    heat_level_min: Optional[int] = None,
    heat_level_max: Optional[int] = None,
    coming_out: Optional[bool] = None,
    first_love: Optional[bool] = None,
    forbidden_romance: Optional[bool] = None,
    found_family: Optional[bool] = None,
    slow_burn: Optional[bool] = None,
    enemies_to_lovers: Optional[bool] = None,
    hurt_comfort: Optional[bool] = None,
    age_gap: Optional[bool] = None,
    college: Optional[bool] = None,
    military_romance: Optional[bool] = None,
    rerank: str = "none",
) -> list[dict[str, Any]]:
    tag_filters: dict[str, Any] = {}

    if genres:
        tag_filters["genres"] = genres
    if settings:
        tag_filters["settings"] = settings
    if military:
        tag_filters["military"] = military
    if kinks:
        tag_filters["kinks"] = kinks
    if any_keyword:
        tag_filters["any_keyword"] = any_keyword
    if heat_level_min is not None:
        tag_filters["heat_level_min"] = int(heat_level_min)
    if heat_level_max is not None:
        tag_filters["heat_level_max"] = int(heat_level_max)

    bool_filters = {
        "coming_out": coming_out,
        "first_love": first_love,
        "forbidden_romance": forbidden_romance,
        "found_family": found_family,
        "slow_burn": slow_burn,
        "enemies_to_lovers": enemies_to_lovers,
        "hurt_comfort": hurt_comfort,
        "age_gap": age_gap,
        "college": college,
        "military_romance": military_romance,
    }
    for key, value in bool_filters.items():
        if value is not None:
            tag_filters[key] = bool(value)

    return _search_backend_advanced(
        query,
        limit=limit,
        topic_or_library=library,
        tag_filters=tag_filters or None,
        rerank=rerank,
        preferred_library=library,
    )


@mcp.tool()
def VS_Search_By_Multiple_Topics(
    query: str,
    topics: list[str],
    limit: int = 10,
    weights: Optional[dict[str, float]] = None,
    rerank: str = "none",
    dedupe_by_stem: bool = False,
) -> dict[str, Any]:
    clean_topics = [t.strip() for t in topics if str(t).strip()]
    if not clean_topics:
        raise ValueError("topics must include at least one topic")
    if len(clean_topics) > 20:
        raise ValueError("topics supports at most 20 entries")

    stores = _resolve_multiple_libraries(clean_topics, require_exposed=True)
    if not stores:
        return {"query": query, "topics": [], "results": []}

    vector_store_ids = [str(s.get("id")) for s in stores]
    store_name_by_id = {str(s.get("id")): str(s.get("name")) for s in stores}
    weight_map = {(k or "").strip().lower(): float(v) for k, v in (weights or {}).items() if k}

    payload: dict[str, Any] = {
        "query": query,
        "limit": max(1, min(max(limit * 3, limit), 100)),
        "return_metadata": True,
        "vector_store_ids": vector_store_ids,
    }
    with _client() as client:
        response = client.post("/v1/scenes/search", json=payload)
        response.raise_for_status()
        rows = response.json().get("data", [])

    results: list[dict[str, Any]] = []
    for item in rows:
        store_id = str(item.get("vector_store_id") or "")
        library = item.get("vector_store_name")
        base_score = float(item.get("score") or 0.0)
        lookup_id = store_id.lower()
        lookup_name = str(library or "").lower()
        topic_weight = weight_map.get(lookup_id, weight_map.get(lookup_name, 1.0))

        text = "\n".join(chunk.get("text", "") for chunk in item.get("content", []))
        results.append(
            {
                "library": library,
                "topic": library,
                "vector_store_id": store_id,
                "filename": item.get("filename"),
                "score": base_score,
                "weighted_score": base_score * topic_weight,
                "weight": topic_weight,
                "excerpt": text,
                "story_metadata": item.get("story_metadata"),
                "resolved_topic": store_name_by_id.get(store_id, library),
            }
        )

    reranked = _rerank_results(results, rerank=rerank)
    reranked.sort(
        key=lambda x: (float(x.get("weighted_score") or 0.0), float(x.get("score") or 0.0)),
        reverse=True,
    )

    if dedupe_by_stem:
        deduped: list[dict[str, Any]] = []
        seen_stems: set[str] = set()
        for row in reranked:
            stem = _normalized_filename_stem(row.get("filename"))
            if stem and stem in seen_stems:
                continue
            if stem:
                seen_stems.add(stem)
            deduped.append(row)
        reranked = deduped

    return {
        "query": query,
        "topics": [{"id": s.get("id"), "name": s.get("name")} for s in stores],
        "weights": weight_map,
        "dedupe_by_stem": bool(dedupe_by_stem),
        "results": reranked[: max(1, min(limit, 100))],
    }


@mcp.tool()
def VS_List_Files_In_Topic(topic: str, limit: int = 200) -> list[dict[str, Any]]:
    store = _resolve_single_library(topic)
    files = _list_story_files(str(store["id"]))
    return files[: max(1, min(limit, 1000))]


@mcp.tool()
def VS_Get_Topic_Stats(topic: str) -> dict[str, Any]:
    store = _resolve_single_library(topic)
    files = _list_story_files(str(store["id"]))
    with_metadata = sum(1 for row in files if row.get("has_metadata"))
    missing_metadata = len(files) - with_metadata
    total_chunks = sum(int(row.get("chunk_count") or 0) for row in files)
    return {
        "id": store.get("id"),
        "topic": store.get("name"),
        "api_exposed": bool(store.get("api_exposed")),
        "usage_bytes": int(store.get("usage_bytes") or 0),
        "file_count": len(files),
        "total_chunks": total_chunks,
        "with_metadata": with_metadata,
        "missing_metadata": missing_metadata,
        "status": store.get("status"),
    }


@mcp.tool()
def VS_Metadata_Facets(topic: str, top_n: int = 20) -> dict[str, Any]:
    store = _resolve_single_library(topic)
    files = _list_story_files(str(store["id"]))
    facets = _compute_facets(files)

    def trim(items: list[tuple[str, int]]) -> list[dict[str, Any]]:
        return [{"value": k, "count": v} for k, v in items[: max(1, min(top_n, 100))]]

    return {
        "topic": store.get("name"),
        "store_id": store.get("id"),
        "top_genres": trim(facets["top_genres"]),
        "top_settings": trim(facets["top_settings"]),
        "top_military": trim(facets["top_military"]),
        "top_kinks": trim(facets["top_kinks"]),
        "top_tone": trim(facets["top_tone"]),
        "heat_level_distribution": facets["heat_level_distribution"],
        "boolean_counts": facets["boolean_counts"],
    }


@mcp.tool()
def VS_Find_Similar_Stories(filename: str, library: str, limit: int = 10) -> list[dict[str, Any]]:
    store = _resolve_single_library(library, require_exposed=True)
    files = _list_story_files(str(store["id"]))
    target = next((row for row in files if str(row.get("filename", "")).lower() == filename.lower()), None)
    if not target:
        raise ValueError(f"Filename not found in {store.get('name')}: {filename}")

    md = target.get("metadata") or {}
    keywords = md.get("search_keywords") or []
    genres = md.get("genres") or []
    settings = md.get("settings") or []
    query_terms = keywords[:3] or genres[:2] or settings[:2] or [filename]
    query = "similar story: " + ", ".join(query_terms)

    results = _search_backend_advanced(
        query,
        limit=max(2, min(limit + 3, 100)),
        topic_or_library=str(store.get("id")),
        tag_filters={"genres": genres[:3]} if genres else None,
        rerank="metadata_match",
        preferred_library=str(store.get("name")),
    )
    return [r for r in results if str(r.get("filename", "")).lower() != filename.lower()][:limit]


@mcp.tool()
def VS_Search_Interactions(
    interaction: str,
    library: Optional[str] = None,
    limit: int = 10,
    rerank: str = "metadata_match",
) -> list[dict[str, Any]]:
    lower = interaction.strip().lower()
    tag_filters: dict[str, Any] = {}
    synonym_map = {
        "rivals": "enemies_to_lovers",
        "enemies to lovers": "enemies_to_lovers",
        "hurt comfort": "hurt_comfort",
        "first love": "first_love",
        "found family": "found_family",
        "coming out": "coming_out",
        "military romance": "military_romance",
        "college": "college",
    }
    if lower in synonym_map:
        tag_filters[synonym_map[lower]] = True
    else:
        tag_filters["any_keyword"] = [interaction]

    return _search_backend_advanced(
        f"interaction: {interaction}",
        limit=limit,
        topic_or_library=library,
        tag_filters=tag_filters,
        rerank=rerank,
        preferred_library=library,
    )


@mcp.tool()
def VS_Build_Filtered_Query(intent: str, library: Optional[str] = None, limit: int = 10) -> dict[str, Any]:
    text = intent.strip().lower()
    tag_filters: dict[str, Any] = {}

    bool_map = {
        "slow burn": "slow_burn",
        "enemies to lovers": "enemies_to_lovers",
        "hurt comfort": "hurt_comfort",
        "age gap": "age_gap",
        "coming out": "coming_out",
        "first love": "first_love",
        "found family": "found_family",
        "military romance": "military_romance",
        "college": "college",
    }
    for phrase, key in bool_map.items():
        if phrase in text:
            tag_filters[key] = True

    heat_min_match = re.search(r"heat\s*(?:level)?\s*(?:>=|over|at least)\s*(\d)", text)
    if heat_min_match:
        tag_filters["heat_level_min"] = int(heat_min_match.group(1))

    keyword_candidates = [
        k for k in ["military", "college", "marine", "army", "football", "wrestling"] if k in text
    ]
    if keyword_candidates:
        tag_filters["any_keyword"] = keyword_candidates

    results = _search_backend_advanced(
        intent,
        limit=limit,
        topic_or_library=library,
        tag_filters=tag_filters or None,
        rerank="metadata_match",
        preferred_library=library,
    )
    return {
        "intent": intent,
        "library": library,
        "tag_filters": tag_filters,
        "results": results,
    }


@mcp.tool()
def VS_Compare_Topics(topic_a: str, topic_b: str, top_n: int = 10) -> dict[str, Any]:
    stats_a = VS_Get_Topic_Stats(topic_a)
    stats_b = VS_Get_Topic_Stats(topic_b)
    facets_a = VS_Metadata_Facets(topic_a, top_n=top_n)
    facets_b = VS_Metadata_Facets(topic_b, top_n=top_n)
    return {
        "topic_a": stats_a,
        "topic_b": stats_b,
        "delta": {
            "file_count": stats_a["file_count"] - stats_b["file_count"],
            "total_chunks": stats_a["total_chunks"] - stats_b["total_chunks"],
            "with_metadata": stats_a["with_metadata"] - stats_b["with_metadata"],
            "usage_bytes": stats_a["usage_bytes"] - stats_b["usage_bytes"],
        },
        "facets": {
            "topic_a": facets_a,
            "topic_b": facets_b,
        },
    }


@mcp.tool()
def VS_Start_Ingest(topic: str) -> dict[str, Any]:
    store = _resolve_single_library(topic)
    guard = _evaluate_ingest_guardrails(store)
    if not guard.get("allowed"):
        return {
            "started": False,
            "topic": store.get("name"),
            "store_id": store.get("id"),
            "guardrail": guard,
        }

    job = _api_post(f"/v1/vector_stores/{store['id']}/ingest-background")
    job_payload = dict(job.get("job") or {})
    job_id = str(job_payload.get("job_id") or job.get("job_id") or "")
    STARTED_INGEST_JOBS[job_id] = {
        "vector_store_id": str(store.get("id")),
        "store_name": store.get("name"),
        "topic": store.get("name"),
        "started_at": int(time.time()),
    }
    return job


@mcp.tool()
def VS_Backfill_Metadata_For_Topic(
    topic: str,
    dry_run: bool = True,
    include_default_metadata: bool = True,
    max_candidates: int = 1000,
) -> dict[str, Any]:
    store = _resolve_single_library(topic, require_exposed=False)
    files = _list_story_files(str(store.get("id")))

    missing = [row for row in files if not row.get("has_metadata")]
    defaulted = [
        row
        for row in files
        if row.get("has_metadata") and include_default_metadata and _is_default_story_metadata(row.get("metadata"))
    ]

    candidate_rows = (missing + defaulted)[: max(1, min(max_candidates, 10000))]
    summary = {
        "topic": store.get("name"),
        "store_id": store.get("id"),
        "file_count": len(files),
        "missing_metadata_count": len(missing),
        "default_metadata_count": len(defaulted),
        "candidate_count": len(candidate_rows),
        "dry_run": bool(dry_run),
        "backend_metadata_only_supported": False,
        "remediation_mode": "ingest_background_fallback",
        "sample_candidates": [
            {
                "filename": row.get("filename"),
                "has_metadata": bool(row.get("has_metadata")),
                "chunk_count": row.get("chunk_count"),
            }
            for row in candidate_rows[:20]
        ],
    }

    if dry_run:
        return summary

    if not candidate_rows:
        summary["started"] = False
        summary["message"] = "No metadata remediation candidates found."
        return summary

    guard = _evaluate_ingest_guardrails(store)
    if not guard.get("allowed"):
        summary["started"] = False
        summary["guardrail"] = guard
        return summary

    job = _api_post(f"/v1/vector_stores/{store['id']}/ingest-background")
    job_payload = dict(job.get("job") or {})
    job_id = str(job_payload.get("job_id") or job.get("job_id") or "")
    STARTED_INGEST_JOBS[job_id] = {
        "vector_store_id": str(store.get("id")),
        "store_name": store.get("name"),
        "topic": store.get("name"),
        "reason": "metadata_backfill",
        "candidate_count": len(candidate_rows),
        "started_at": int(time.time()),
    }
    summary["started"] = True
    summary["job"] = job
    return summary


@mcp.tool()
def VS_Get_Ingest_Status(job_id: str, topic: Optional[str] = None) -> dict[str, Any]:
    active = _augment_jobs_with_queue_positions(_list_active_ingest_jobs())
    active_by_id = {str(j.get("job_id")): j for j in active if j.get("job_id")}

    if topic:
        store = _resolve_single_library(topic)
        payload = _api_get(f"/v1/vector_stores/{store['id']}/ingest-background/{job_id}")
        return {**payload, **active_by_id.get(str(job_id), {})}

    known = STARTED_INGEST_JOBS.get(job_id)
    if known:
        payload = _api_get(f"/v1/vector_stores/{known['vector_store_id']}/ingest-background/{job_id}")
        return {**payload, **active_by_id.get(str(job_id), {})}

    for job in active:
        if str(job.get("job_id")) == str(job_id):
            return job
    raise ValueError("Job not found. Provide topic/library if this is an older completed job.")


@mcp.tool()
def VS_List_Active_Ingests() -> list[dict[str, Any]]:
    return _augment_jobs_with_queue_positions(_list_active_ingest_jobs())


@mcp.tool()
def VS_Cancel_Ingest(job_id: str, topic: Optional[str] = None) -> dict[str, Any]:
    job = VS_Get_Ingest_Status(job_id=job_id, topic=topic)
    pid = job.get("process_id")
    if not pid:
        return {"job_id": job_id, "status": job.get("status"), "cancelled": False, "reason": "No process_id"}

    try:
        os.kill(int(pid), signal.SIGTERM)
        return {"job_id": job_id, "process_id": pid, "cancelled": True}
    except ProcessLookupError:
        return {"job_id": job_id, "process_id": pid, "cancelled": False, "reason": "Process not found"}
    except PermissionError:
        return {"job_id": job_id, "process_id": pid, "cancelled": False, "reason": "Permission denied"}


@mcp.tool()
def VS_List_Metadata_Errors(limit: int = 100) -> list[dict[str, Any]]:
    payload = _api_get("/v1/activity", params={"after_id": 0, "limit": max(1, min(limit, 500))})
    items = payload.get("data", [])
    return [row for row in items if str(row.get("type")) == "story_metadata_extraction_failed"]


@mcp.tool()
def VS_Find_Default_Metadata(topic: str, limit: int = 100) -> list[dict[str, Any]]:
    store = _resolve_single_library(topic)
    files = _list_story_files(str(store["id"]))
    defaults = [
        {
            "filename": row.get("filename"),
            "chunk_count": row.get("chunk_count"),
            "metadata": row.get("metadata"),
        }
        for row in files
        if _is_default_story_metadata(row.get("metadata"))
    ]
    return defaults[: max(1, min(limit, 1000))]


@mcp.tool()
def VS_Find_Missing_Metadata(topic: str, limit: int = 100) -> list[dict[str, Any]]:
    store = _resolve_single_library(topic)
    files = _list_story_files(str(store["id"]))
    missing = [
        {
            "filename": row.get("filename"),
            "chunk_count": row.get("chunk_count"),
            "has_metadata": row.get("has_metadata"),
        }
        for row in files
        if not row.get("has_metadata")
    ]
    return missing[: max(1, min(limit, 1000))]


@mcp.tool()
def VS_Set_Library_Exposure(topic: str, api_exposed: bool, confirm: bool = False) -> dict[str, Any]:
    store = _resolve_single_library(topic, require_exposed=False)
    if not confirm:
        return {
            "ok": False,
            "message": "Confirmation required. Re-run with confirm=true.",
            "topic": store.get("name"),
            "current_api_exposed": bool(store.get("api_exposed")),
            "requested_api_exposed": bool(api_exposed),
        }
    return _api_patch(
        f"/v1/vector_stores/{store['id']}/exposure",
        payload={"api_exposed": bool(api_exposed)},
    )


@mcp.tool()
def VS_Save_Search(
    name: str,
    query: str,
    library: Optional[str] = None,
    limit: int = 10,
    tag_filters: Optional[dict[str, Any]] = None,
    rerank: str = "none",
) -> dict[str, Any]:
    clean = _normalize_saved_search_name(name)
    items = _load_saved_searches()
    items[clean] = {
        "query": query,
        "library": library,
        "limit": max(1, min(limit, 100)),
        "tag_filters": tag_filters or {},
        "rerank": rerank,
    }
    _save_saved_searches(items)
    return {"saved": True, "name": clean, "count": len(items)}


@mcp.tool()
def VS_Run_Saved_Search(name: str) -> dict[str, Any]:
    name = _normalize_saved_search_name(name)
    items = _load_saved_searches()
    if name not in items:
        raise ValueError(f"Saved search not found: {name}")
    conf = items[name]

    started = time.perf_counter()
    try:
        base_results = _search_backend_advanced(
            conf.get("query", ""),
            limit=int(conf.get("limit", 10)),
            topic_or_library=conf.get("library"),
            tag_filters=conf.get("tag_filters") or None,
            rerank="none",
            preferred_library=conf.get("library"),
        )

        rerank_mode = str(conf.get("rerank", "none") or "none")
        if rerank_mode == "none":
            results = base_results
            changed_top = None
            order_delta = None
        else:
            results = _rerank_results(base_results, rerank=rerank_mode, preferred_library=conf.get("library"))
            base_top = base_results[0] if base_results else None
            rerank_top = results[0] if results else None
            changed_top = bool(
                base_top
                and rerank_top
                and (
                    base_top.get("filename") != rerank_top.get("filename")
                    or base_top.get("library") != rerank_top.get("library")
                )
            )
            order_delta = _compute_order_delta_topk(base_results, results, k=10)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        history_entry = _record_saved_search_run(
            name=name,
            config=conf,
            result_count=len(results),
            top_result=results[0] if results else None,
            latency_ms=elapsed_ms,
            error=False,
            error_message=None,
            rerank_changed_top_result=changed_top,
            rerank_order_delta_top10=order_delta,
        )
        alert = _detect_zero_result_alert(name, SEARCH_ZERO_RESULT_ALERT_THRESHOLD)
        return {
            "name": name,
            "config": conf,
            "results": results,
            "history_entry": history_entry,
            "alert": alert,
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        _record_saved_search_run(
            name=name,
            config=conf,
            result_count=0,
            top_result=None,
            latency_ms=elapsed_ms,
            error=True,
            error_message=str(exc),
            rerank_changed_top_result=None,
            rerank_order_delta_top10=None,
        )
        raise


@mcp.tool()
def VS_List_Saved_Searches() -> dict[str, Any]:
    items = _load_saved_searches()
    return {"count": len(items), "items": items}


@mcp.tool()
def VS_Saved_Search_History(name: Optional[str] = None, limit: int = 50) -> dict[str, Any]:
    max_limit = max(1, min(limit, 1000))
    entries = _load_saved_search_history()
    if name:
        target = name.strip().lower()
        entries = [row for row in entries if str(row.get("name", "")).lower() == target]
    sliced = entries[-max_limit:]
    sliced.reverse()
    alert = _detect_zero_result_alert(name, SEARCH_ZERO_RESULT_ALERT_THRESHOLD) if name else None
    return {
        "count": len(sliced),
        "total_available": len(entries),
        "alert": alert,
        "items": sliced,
    }


@mcp.tool()
def VS_Delete_Saved_Search(name: str) -> dict[str, Any]:
    name = _normalize_saved_search_name(name)
    items = _load_saved_searches()
    if name not in items:
        return {"deleted": False, "name": name, "message": "Saved search not found"}
    del items[name]
    _save_saved_searches(items)
    return {"deleted": True, "name": name, "count": len(items)}


@mcp.tool()
def VS_Rename_Saved_Search(old_name: str, new_name: str, overwrite: bool = False) -> dict[str, Any]:
    source = _normalize_saved_search_name(old_name)
    target = _normalize_saved_search_name(new_name)
    if source == target:
        return {"renamed": False, "old_name": source, "new_name": target, "message": "Names are identical"}

    items = _load_saved_searches()
    if source not in items:
        return {"renamed": False, "old_name": source, "new_name": target, "message": "Source not found"}
    if target in items and not overwrite:
        return {
            "renamed": False,
            "old_name": source,
            "new_name": target,
            "message": "Target already exists. Re-run with overwrite=true.",
        }

    items[target] = items[source]
    del items[source]
    _save_saved_searches(items)
    return {"renamed": True, "old_name": source, "new_name": target, "count": len(items)}


@mcp.tool()
def VS_Get_Topic_Readiness(topic: str) -> dict[str, Any]:
    stats = VS_Get_Topic_Stats(topic)
    ready_checks = {
        "is_exposed": bool(stats.get("api_exposed")),
        "has_files": int(stats.get("file_count") or 0) > 0,
        "has_chunks": int(stats.get("total_chunks") or 0) > 0,
        "has_any_metadata": int(stats.get("with_metadata") or 0) > 0,
    }

    issues: list[str] = []
    if not ready_checks["is_exposed"]:
        issues.append("Topic is not api_exposed; searches are intentionally blocked.")
    if not ready_checks["has_files"]:
        issues.append("Topic has zero files; ingest is required.")
    if not ready_checks["has_chunks"]:
        issues.append("Topic has zero chunks; embedding/chunking ingest likely incomplete.")
    if ready_checks["has_files"] and not ready_checks["has_any_metadata"]:
        issues.append("Topic has files but no structured metadata yet.")

    if all(ready_checks.values()):
        status = "ready"
    elif ready_checks["is_exposed"] and ready_checks["has_files"] and ready_checks["has_chunks"]:
        status = "searchable_no_metadata"
    else:
        status = "not_ready"

    return {
        "topic": stats.get("topic"),
        "store_id": stats.get("id"),
        "status": status,
        "checks": ready_checks,
        "issues": issues,
        "recommended_actions": _recommend_actions_from_readiness({"checks": ready_checks}),
        "stats": stats,
    }


@mcp.tool()
def VS_List_NotReady_Topics(exposed_only: bool = True) -> dict[str, Any]:
    topics = VS_List_Topics(exposed_only=exposed_only)
    not_ready: list[dict[str, Any]] = []

    for row in topics:
        topic_name = str(row.get("topic") or "")
        if not topic_name:
            continue
        snapshot = VS_Get_Topic_Readiness(topic_name)
        if snapshot.get("status") != "ready":
            not_ready.append(snapshot)

    return {
        "count": len(not_ready),
        "items": not_ready,
    }


@mcp.tool()
def VS_Topic_Coverage_Report(exposed_only: bool = True, limit: int = 50) -> dict[str, Any]:
    topics = VS_List_Topics(exposed_only=exposed_only)
    rows: list[dict[str, Any]] = []

    for topic_row in topics:
        topic_name = str(topic_row.get("topic") or "")
        if not topic_name:
            continue
        readiness = VS_Get_Topic_Readiness(topic_name)
        checks = readiness.get("checks") or {}
        stats = readiness.get("stats") or {}
        missing_count = int(stats.get("missing_metadata") or 0)

        gap_score = 0.0
        gap_score += 4.0 if not checks.get("is_exposed") else 0.0
        gap_score += 3.0 if not checks.get("has_files") else 0.0
        gap_score += 2.0 if not checks.get("has_chunks") else 0.0
        gap_score += 1.0 if not checks.get("has_any_metadata") else 0.0
        gap_score += min(missing_count, 10000) / 10000.0

        rows.append(
            {
                "topic": readiness.get("topic"),
                "store_id": readiness.get("store_id"),
                "status": readiness.get("status"),
                "gap_score": round(gap_score, 4),
                "file_count": int(stats.get("file_count") or 0),
                "total_chunks": int(stats.get("total_chunks") or 0),
                "with_metadata": int(stats.get("with_metadata") or 0),
                "missing_metadata": missing_count,
                "issues": readiness.get("issues") or [],
                "recommended_actions": _recommend_actions_from_readiness(readiness),
            }
        )

    rows.sort(key=lambda r: (float(r.get("gap_score") or 0.0), int(r.get("missing_metadata") or 0)), reverse=True)
    safe_limit = max(1, min(limit, 500))
    sliced = rows[:safe_limit]
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "count": len(sliced),
        "total_topics": len(rows),
        "status_counts": status_counts,
        "items": sliced,
    }


@mcp.tool()
def VS_Get_Health_Summary() -> dict[str, Any]:
    return _api_get("/health/summary")


@mcp.tool()
def VS_Get_Recent_Activity(limit: int = 100) -> dict[str, Any]:
    return _api_get("/v1/activity", params={"after_id": 0, "limit": max(1, min(limit, 500))})


@mcp.tool()
def VS_Get_Ingest_Log(job_id: str, lines: int = 100) -> dict[str, Any]:
    log_path = None

    try:
        summary = _api_get("/health/summary")
        for j in summary.get("active_ingest_jobs", []):
            if str(j.get("job_id")) == str(job_id):
                log_path = j.get("log_path")
                break
    except Exception:
        pass

    if not log_path:
        guess = Path(__file__).resolve().parent.parent / "logs" / "ingest-jobs" / f"{job_id}.log"
        if guess.is_file():
            log_path = str(guess)

    if not log_path:
        return {"job_id": job_id, "found": False, "message": "Log file not found"}

    path = Path(log_path)
    if not path.is_file():
        return {"job_id": job_id, "found": False, "log_path": log_path, "message": "Log file not found"}

    max_lines = max(1, min(lines, 2000))
    content_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {
        "job_id": job_id,
        "found": True,
        "log_path": str(path),
        "line_count": len(content_lines),
        "tail": "\n".join(content_lines[-max_lines:]),
    }


def _build_http_app():
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
