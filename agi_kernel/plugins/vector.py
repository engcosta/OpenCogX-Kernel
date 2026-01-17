"""
📊 Vector Plugin
================

Adapter for Vector Database (Qdrant).

This is a PLUGIN, not part of the kernel.
Provides semantic search capabilities.
"""

from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING
import structlog

if TYPE_CHECKING:
    from agi_kernel.core.memory import MemoryItem

logger = structlog.get_logger()


class VectorPlugin:
    """
    Vector Database Plugin using Qdrant.
    
    Provides:
    - Vector storage for semantic search
    - Similarity search for memory recall
    - Hybrid search combining metadata filters
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: str = "agi_kernel_memory",
        vector_size: int = 1024,
        llm_plugin=None,
    ):
        """
        Initialize the Vector Plugin.
        
        Args:
            host: Qdrant host
            port: Qdrant port
            collection_name: Name of the collection
            vector_size: Size of embedding vectors
            llm_plugin: LLM plugin for generating embeddings
        """
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = port or int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.llm_plugin = llm_plugin
        
        self._client = None
        self._initialized = False
        
        logger.info(
            "vector_plugin_created",
            host=self.host,
            port=self.port,
            collection=collection_name,
        )
    
    async def initialize(self, force_recreate: bool = False) -> bool:
        """
        Initialize connection to Qdrant.
        
        Args:
            force_recreate: If True, delete and recreate the collection
        
        Returns:
            True if successful
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            
            self._client = QdrantClient(host=self.host, port=self.port)
            
            # Check if collection exists
            collections = self._client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if exists and force_recreate:
                self._client.delete_collection(self.collection_name)
                exists = False
            
            if not exists:
                # Create collection with initial vector size
                # Will be recreated if embedding dimension differs
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("vector_collection_created", name=self.collection_name, size=self.vector_size)
            else:
                # Check existing collection dimension
                info = self._client.get_collection(self.collection_name)
                existing_size = info.config.params.vectors.size
                if existing_size != self.vector_size:
                    logger.warning(
                        "vector_dimension_mismatch",
                        expected=self.vector_size,
                        actual=existing_size,
                    )
                    # Update our size to match existing
                    self.vector_size = existing_size
            
            self._initialized = True
            logger.info("vector_plugin_initialized")
            return True
            
        except Exception as e:
            logger.error("vector_plugin_init_failed", error=str(e))
            return False
    
    async def ensure_dimension(self, embedding: list[float]) -> bool:
        """
        Ensure collection dimension matches embedding size.
        Recreates collection if needed.
        """
        if len(embedding) != self.vector_size:
            logger.info(
                "recreating_collection_for_dimension",
                old_size=self.vector_size,
                new_size=len(embedding),
            )
            self.vector_size = len(embedding)
            
            if self._client:
                try:
                    self._client.delete_collection(self.collection_name)
                except:
                    pass
            
            return await self.initialize()
        return True
    
    async def store_memory(
        self,
        item: MemoryItem,
        embedding: Optional[list[float]] = None,
    ) -> bool:
        """
        Store a memory item in the vector database.
        
        Args:
            item: The memory item to store
            embedding: Pre-computed embedding (optional)
            
        Returns:
            True if successful
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            from qdrant_client.models import PointStruct
            
            # Generate embedding if not provided
            if embedding is None and self.llm_plugin:
                text = str(item.content)
                embedding = await self.llm_plugin.embed(text)
            
            if embedding is None:
                logger.warning("no_embedding_available", item_id=item.id)
                return False
            
            # Ensure collection dimension matches
            await self.ensure_dimension(embedding)
            
            # Create point
            point = PointStruct(
                id=hash(item.id) % (2**63 - 1),  # Qdrant needs int64 IDs
                vector=embedding,
                payload={
                    "memory_id": item.id,
                    "type": item.type.value,
                    "content": item.content,
                    "timestamp": item.timestamp.isoformat(),
                    "confidence": item.confidence,
                    "source": item.source,
                },
            )
            
            # Upsert point
            self._client.upsert(
                collection_name=self.collection_name,
                points=[point],
            )
            
            logger.debug("memory_stored_in_vector", item_id=item.id)
            return True
            
        except Exception as e:
            logger.error("vector_store_failed", error=str(e))
            return False
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.5,
        filter_type: Optional[str] = None,
    ) -> list[MemoryItem]:
        """
        Search for similar memories.
        
        Args:
            query: Search query
            limit: Maximum results
            score_threshold: Minimum similarity score
            filter_type: Filter by memory type
            
        Returns:
            List of matching memory items
        """
        if not self._initialized:
            await self.initialize()
        
        if not self.llm_plugin:
            logger.warning("no_llm_plugin_for_search")
            return []
        
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            from agi_kernel.core.memory import MemoryItem, MemoryType
            
            # Generate query embedding
            query_embedding = await self.llm_plugin.embed(query)
            
            # Build filter if type specified
            query_filter = None
            if filter_type:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="type",
                            match=MatchValue(value=filter_type),
                        )
                    ]
                )
            
            # Search
            results = self._client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
            )
            
            # Convert to MemoryItems
            memories = []
            for hit in results:
                payload = hit.payload
                if payload:
                    from datetime import datetime
                    
                    item = MemoryItem(
                        id=payload.get("memory_id", ""),
                        type=MemoryType(payload.get("type", "semantic")),
                        content=payload.get("content", {}),
                        timestamp=datetime.fromisoformat(payload.get("timestamp", datetime.utcnow().isoformat())),
                        confidence=payload.get("confidence", 1.0),
                        source=payload.get("source", "vector_search"),
                    )
                    memories.append(item)
            
            logger.debug(
                "vector_search_complete",
                query=query[:30],
                results=len(memories),
            )
            
            return memories
            
        except Exception as e:
            logger.error("vector_search_failed", error=str(e))
            return []
    
    async def delete_memory(self, item_id: str) -> bool:
        """Delete a memory from the vector database."""
        if not self._initialized:
            return False
        
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="memory_id",
                            match=MatchValue(value=item_id),
                        )
                    ]
                ),
            )
            
            logger.debug("memory_deleted_from_vector", item_id=item_id)
            return True
            
        except Exception as e:
            logger.error("vector_delete_failed", error=str(e))
            return False
    
    def get_stats(self) -> dict:
        """Get statistics about the vector database."""
        if not self._initialized or not self._client:
            return {"initialized": False}
        
        try:
            info = self._client.get_collection(self.collection_name)
            return {
                "initialized": True,
                "collection": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
            }
        except Exception as e:
            return {"initialized": True, "error": str(e)}
    
    async def close(self) -> None:
        """Close the client connection."""
        if self._client:
            self._client.close()
