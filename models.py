from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime


class VectorStoreCreateRequest(BaseModel):
    name: str
    file_ids: Optional[List[str]] = None
    expires_after: Optional[Dict[str, Any]] = None
    chunking_strategy: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class VectorStoreResponse(BaseModel):
    id: str
    object: str = "vector_store"
    created_at: int
    name: str
    usage_bytes: int
    file_counts: Dict[str, int]
    status: str
    expires_after: Optional[Dict[str, Any]] = None
    expires_at: Optional[int] = None
    last_active_at: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    # Whether this store may be searched via the public /v1/scenes/search API
    # and the MCP server. Opt-in: defaults to False unless explicitly enabled.
    api_exposed: bool = False


class VectorStoreExposureUpdateRequest(BaseModel):
    api_exposed: bool


class VectorStoreSearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 20
    filters: Optional[Dict[str, Any]] = None
    return_metadata: Optional[bool] = True


class ContentChunk(BaseModel):
    type: str = "text"
    text: str


class SearchResult(BaseModel):
    file_id: str
    filename: str
    score: float
    attributes: Optional[Dict[str, Any]] = None
    content: List[ContentChunk]


class VectorStoreSearchResponse(BaseModel):
    object: str = "vector_store.search_results.page"
    search_query: str
    data: List[SearchResult]
    has_more: bool = False
    next_page: Optional[str] = None


class EmbeddingCreateRequest(BaseModel):
    content: str
    embedding: List[float]
    metadata: Optional[Dict[str, Any]] = None


class EmbeddingResponse(BaseModel):
    id: str
    object: str = "embedding"
    vector_store_id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: int


class EmbeddingBatchCreateRequest(BaseModel):
    embeddings: List[EmbeddingCreateRequest]


class EmbeddingBatchCreateResponse(BaseModel):
    object: str = "embedding.batch"
    data: List[EmbeddingResponse]
    created: int


class VectorStoreListResponse(BaseModel):
    object: str = "list"
    data: List[VectorStoreResponse]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool = False


class StoryMetadata(BaseModel):
    """Structured tags describing a whole story (one row per
    vector_store_id + filename, shared by every chunk of that file)."""
    genres: List[str] = []
    romance_subgenres: List[str] = []
    tropes: List[str] = []
    relationship_types: List[str] = []
    character_archetypes: List[str] = []
    occupations: List[str] = []
    settings: List[str] = []
    sports: List[str] = []
    military: List[str] = []
    kinks: List[str] = []
    emotional_tone: List[str] = []
    major_conflicts: List[str] = []
    coming_out: bool = False
    first_love: bool = False
    forbidden_romance: bool = False
    found_family: bool = False
    slow_burn: bool = False
    enemies_to_lovers: bool = False
    hurt_comfort: bool = False
    age_gap: bool = False
    college: bool = False
    military_romance: bool = False
    heat_level: int = Field(1, ge=1, le=5)
    content_warnings: List[str] = []
    search_keywords: List[str] = []
    # POV: "dual" | "first person" | "third person"
    pov: Optional[str] = None
    # Tone (may hold more than one): wholesome | emotional | erotic | suspenseful | humorous
    tone: List[str] = []
    # Explicitness: "kissing" | "fade to black" | "explicit" | "very explicit"
    explicitness: Optional[str] = None
    # Relationship structure: "monogamous" | "open" | "poly"
    relationship_structure: Optional[str] = None
    # Ending: "HEA" | "HFN"
    ending: Optional[str] = None


class StoryMetadataResponse(StoryMetadata):
    vector_store_id: str
    filename: str
    created_at: int
    updated_at: int


class StoryMetadataBulkUpsertRequest(BaseModel):
    # filename -> tags
    items: Dict[str, StoryMetadata]


class StoryMetadataExtractionRequest(BaseModel):
    filename: str
    text: str


class StoryFileInfo(BaseModel):
    filename: str
    chunk_count: int
    has_metadata: bool
    metadata: Optional[StoryMetadata] = None


class SceneSearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10
    vector_store_ids: Optional[List[str]] = None
    return_metadata: Optional[bool] = True
    # Optional structured-tag filters, e.g. {"military": ["marine"]},
    # {"coming_out": true}, {"heat_level_min": 3}, {"any_keyword": ["Marine"]}.
    # See STORY_METADATA_ARRAY_FIELDS / BOOL_FIELDS / STRING_FIELDS in main.py
    # for the full set of supported keys.
    tag_filters: Optional[Dict[str, Any]] = None


class SceneSearchResult(BaseModel):
    vector_store_id: str
    vector_store_name: str
    file_id: str
    filename: str
    score: float
    attributes: Optional[Dict[str, Any]] = None
    content: List[ContentChunk]
    story_metadata: Optional[StoryMetadata] = None


class SceneSearchResponse(BaseModel):
    object: str = "scene.search_results.page"
    search_query: str
    data: List[SceneSearchResult]
    has_more: bool = False