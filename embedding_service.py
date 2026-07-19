from typing import List, Optional
import asyncio
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from config import settings, EmbeddingConfig
import logging

class EmbeddingService:
    """Service for generating embeddings using OpenAI SDK pointed at LiteLLM proxy"""
    _MODEL_LOADING_ERROR_MARKERS = (
        "model is unloaded",
        "no models loaded",
        "failed to load model",
        "error loading model",
        "still loading",
        "loading model",
    )
    _CONNECTIVITY_ERROR_MARKERS = (
        "connection refused",
        "failed to establish a new connection",
        "max retries exceeded",
        "read timed out",
        "connect timeout",
        "temporary failure in name resolution",
        "name or service not known",
        "nodename nor servname provided",
        "connection reset by peer",
        "network is unreachable",
        "timed out",
    )
    _INVALID_MODEL_ERROR_MARKERS = (
        "invalid model identifier",
        "model not found",
        "unknown model",
        "model does not exist",
    )
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or settings.embedding

    def _litellm_model(self) -> str:
        return self.config.model

    def _is_model_loading_error(self, error_message: str) -> bool:
        normalized = error_message.lower()
        return any(marker in normalized for marker in self._MODEL_LOADING_ERROR_MARKERS)

    def _is_connectivity_error(self, error_message: str) -> bool:
        normalized = error_message.lower()
        return any(marker in normalized for marker in self._CONNECTIVITY_ERROR_MARKERS)

    def _is_invalid_model_error(self, error_message: str) -> bool:
        normalized = error_message.lower()
        return any(marker in normalized for marker in self._INVALID_MODEL_ERROR_MARKERS)

    def _normalize_api_base(self, base_url: str) -> str:
        normalized = base_url.strip().rstrip("/")
        if normalized.endswith("/v1"):
            return normalized
        return f"{normalized}/v1"

    def _embedding_targets(self) -> list[tuple[str, str, str]]:
        targets: list[tuple[str, str, str]] = []

        if settings.local_lm_studio_url:
            targets.append(
                (
                    self._normalize_api_base(settings.local_lm_studio_url),
                    settings.local_lm_studio_api_key or self.config.api_key,
                    "local_lm_studio",
                )
            )

        if settings.local_lm_studio_tailscale_url:
            targets.append(
                (
                    self._normalize_api_base(settings.local_lm_studio_tailscale_url),
                    settings.local_lm_studio_api_key or self.config.api_key,
                    "tailscale_lm_studio",
                )
            )

        targets.append((self.config.base_url, self.config.api_key, "embedding_base_url"))

        deduped_targets: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str]] = set()
        for api_base, api_key, source in targets:
            dedupe_key = (api_base, api_key)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            deduped_targets.append((api_base, api_key, source))

        return deduped_targets

    def _sync_openai_embeddings_request(self, api_base: str, api_key: str, texts: List[str]) -> dict:
        endpoint = f"{api_base.rstrip('/')}/embeddings"
        payload = json.dumps({"model": self.config.model, "input": texts}).encode("utf-8")
        request = Request(
            endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        try:
            with urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"HTTP {exc.code}: {body}")
        except URLError as exc:
            raise RuntimeError(str(exc))

    async def _generate_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        max_attempts = max(1, int(self.config.load_retry_attempts))
        delay_seconds = max(0.0, float(self.config.load_retry_delay_seconds))
        backoff = max(1.0, float(self.config.load_retry_backoff))
        max_delay = max(delay_seconds, float(self.config.load_retry_max_delay_seconds))
        targets = self._embedding_targets()

        for attempt in range(1, max_attempts + 1):
            saw_loading_error = False
            saw_connectivity_error = False
            last_error_message: Optional[str] = None
            try:
                for api_base, api_key, source in targets:
                    try:
                        response = await asyncio.to_thread(
                            self._sync_openai_embeddings_request,
                            api_base,
                            api_key,
                            texts,
                        )
                        logging.debug(f"Embedding response: {response}")

                        data = sorted(response.get("data", []), key=lambda item: item.get("index", 0))
                        embeddings = [item["embedding"] for item in data]
                        returned_dimensions = sorted({len(embedding) for embedding in embeddings})
                        logging.debug(
                            "Embedding batch completed: model=%s batch_size=%s dimensions=%s source=%s api_base=%s",
                            self.config.model,
                            len(texts),
                            returned_dimensions,
                            source,
                            api_base,
                        )

                        if len(embeddings) != len(texts):
                            raise ValueError(f"Expected {len(texts)} embeddings, got {len(embeddings)}")

                        for embedding in embeddings:
                            if len(embedding) != self.config.dimensions:
                                raise ValueError(
                                    f"Expected embedding dimension {self.config.dimensions}, "
                                    f"got {len(embedding)}"
                                )

                        return embeddings
                    except Exception as target_error:
                        message = str(target_error)
                        last_error_message = message

                        if self._is_connectivity_error(message):
                            saw_connectivity_error = True
                            logging.warning(
                                "Embedding endpoint unreachable (attempt %s/%s, source=%s, api_base=%s): %s",
                                attempt,
                                max_attempts,
                                source,
                                api_base,
                                message,
                            )
                            continue

                        if self._is_invalid_model_error(message):
                            logging.warning(
                                "Embedding model unavailable on endpoint (attempt %s/%s, model=%s, source=%s, api_base=%s): %s",
                                attempt,
                                max_attempts,
                                self.config.model,
                                source,
                                api_base,
                                message,
                            )
                            continue

                        if self.config.retry_on_loading_errors and self._is_model_loading_error(message):
                            saw_loading_error = True
                            logging.warning(
                                "Embedding model not ready (attempt %s/%s, model=%s, source=%s, api_base=%s): %s",
                                attempt,
                                max_attempts,
                                self.config.model,
                                source,
                                api_base,
                                message,
                            )
                            continue

                        raise RuntimeError(f"Failed to generate embeddings: {message}")
            except RuntimeError:
                raise

            should_retry = attempt < max_attempts and (saw_loading_error or saw_connectivity_error)
            if should_retry:
                logging.warning(
                    "Embedding request retry scheduled (attempt %s/%s, model=%s, loading_error=%s, connectivity_error=%s). Retrying in %.1fs",
                    attempt,
                    max_attempts,
                    self.config.model,
                    saw_loading_error,
                    saw_connectivity_error,
                    delay_seconds,
                )
                if delay_seconds > 0:
                    await asyncio.sleep(delay_seconds)
                    delay_seconds = min(max_delay, delay_seconds * backoff)
                continue

            if last_error_message:
                raise RuntimeError(f"Failed to generate embeddings: {last_error_message}")
            raise RuntimeError("Failed to generate embeddings: no configured embedding targets succeeded")

        raise RuntimeError("Failed to generate embeddings: model did not become ready in time")

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using LiteLLM proxy
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            return (await self._generate_embedding_batch([text]))[0]
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            if not texts:
                return []

            batch_size = max(1, int(self.config.batch_size))
            embeddings: List[List[float]] = []

            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                embeddings.extend(await self._generate_embedding_batch(batch))

            return embeddings
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}")
    
    def update_config(self, new_config: EmbeddingConfig):
        """Update the embedding configuration"""
        self.config = new_config


# Global embedding service instance
embedding_service = EmbeddingService() 