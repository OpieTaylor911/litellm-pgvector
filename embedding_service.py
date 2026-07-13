import asyncio
from typing import List, Optional
from config import settings, EmbeddingConfig
from litellm.types.utils import EmbeddingResponse
import litellm
import logging

class EmbeddingService:
    """Service for generating embeddings using OpenAI SDK pointed at LiteLLM proxy"""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or settings.embedding


    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using LiteLLM proxy
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            response: EmbeddingResponse = await litellm.aembedding(
                model=self.config.model,
                input=[text],
                api_base=self.config.base_url,
                api_key=self.config.api_key
            )
            logging.debug(f"Embedding response: {response}")
            
            # Extract embedding from response
            embedding = response.data[0]["embedding"]
            
            # Validate embedding dimensions
            if len(embedding) != self.config.dimensions:
                raise ValueError(
                    f"Expected embedding dimension {self.config.dimensions}, "
                    f"got {len(embedding)}"
                )
            
            return embedding
            
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

            concurrency = max(1, int(self.config.concurrency))
            semaphore = asyncio.Semaphore(concurrency)

            async def embed_with_limit(index: int, text: str) -> tuple[int, List[float]]:
                async with semaphore:
                    embedding = await self.generate_embedding(text)
                    return index, embedding

            tasks = [embed_with_limit(i, text) for i, text in enumerate(texts)]
            results = await asyncio.gather(*tasks)
            results.sort(key=lambda item: item[0])
            return [embedding for _, embedding in results]
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}")
    
    def update_config(self, new_config: EmbeddingConfig):
        """Update the embedding configuration"""
        self.config = new_config


# Global embedding service instance
embedding_service = EmbeddingService() 