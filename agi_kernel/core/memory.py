"""
🧠 Memory System
================

Memory is treated as a living structure, not a database.

Memory Types:
- Semantic Memory: Facts and concepts (knowledge)
- Episodic Memory: Experiences (question → answer → outcome)
- Temporal Memory: Validity over time, decay, contradictions

Key Principles:
- Forgetting is a feature, not a bug
- Contradictions are recorded, not deleted
- Memory has temporal validity
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import structlog

logger = structlog.get_logger()


class MemoryType(Enum):
    """Types of memory in the system."""
    SEMANTIC = "semantic"   # Facts and concepts
    EPISODIC = "episodic"   # Experiences and outcomes
    TEMPORAL = "temporal"   # Time-bound knowledge


@dataclass
class MemoryItem:
    """
    A single item stored in memory.
    
    All memory items have temporal validity and can decay.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MemoryType = MemoryType.SEMANTIC
    content: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Validity period
    valid_from: datetime = field(default_factory=datetime.utcnow)
    valid_until: Optional[datetime] = None  # None = no expiry
    
    # Strength and reliability
    strength: float = 1.0  # Decays over time
    access_count: int = 0  # How often recalled
    last_accessed: Optional[datetime] = None
    
    # Source tracking
    source: str = "observation"
    confidence: float = 1.0
    
    # Contradiction tracking
    contradicted_by: list[str] = field(default_factory=list)
    contradicts: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Serialize memory item for storage."""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "valid_from": self.valid_from.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "strength": self.strength,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "source": self.source,
            "confidence": self.confidence,
            "contradicted_by": self.contradicted_by,
            "contradicts": self.contradicts,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> MemoryItem:
        """Deserialize memory item from storage."""
        return cls(
            id=data["id"],
            type=MemoryType(data["type"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            valid_from=datetime.fromisoformat(data["valid_from"]),
            valid_until=datetime.fromisoformat(data["valid_until"]) if data.get("valid_until") else None,
            strength=data.get("strength", 1.0),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
            source=data.get("source", "observation"),
            confidence=data.get("confidence", 1.0),
            contradicted_by=data.get("contradicted_by", []),
            contradicts=data.get("contradicts", []),
        )


@dataclass
class EpisodicMemory:
    """
    An episodic memory storing an experience.
    
    Structure: Question → Answer → Outcome
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    question: str = ""
    answer: str = ""
    outcome: str = ""  # PASS, FAIL, PARTIAL
    confidence: float = 1.0
    reasoning_strategy: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: dict[str, Any] = field(default_factory=dict)
    
    # Learning metadata
    failure_reason: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> dict:
        """Serialize episodic memory for storage."""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "outcome": self.outcome,
            "confidence": self.confidence,
            "reasoning_strategy": self.reasoning_strategy,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "failure_reason": self.failure_reason,
            "retry_count": self.retry_count,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> EpisodicMemory:
        """Deserialize episodic memory from storage."""
        return cls(
            id=data["id"],
            question=data["question"],
            answer=data["answer"],
            outcome=data["outcome"],
            confidence=data.get("confidence", 1.0),
            reasoning_strategy=data.get("reasoning_strategy", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            context=data.get("context", {}),
            failure_reason=data.get("failure_reason"),
            retry_count=data.get("retry_count", 0),
        )


class Memory:
    """
    The Memory System maintains all knowledge and experiences.
    
    Core API (Mandatory):
    - store(item, type): Store a memory item
    - recall(query, context): Retrieve relevant memories
    - decay(): Apply time-based memory decay
    
    Laws:
    - Forgetting is mandatory (decay)
    - Contradictions are recorded, not deleted
    - Everything has temporal validity
    """
    
    def __init__(
        self,
        vector_plugin=None,
        decay_rate: float = 0.1,
    ):
        """
        Initialize the Memory System.
        
        Args:
            vector_plugin: Optional Qdrant plugin for semantic search
            decay_rate: Rate of memory decay per decay cycle
        """
        self.semantic: dict[str, MemoryItem] = {}
        self.episodic: dict[str, EpisodicMemory] = {}
        self.temporal: dict[str, MemoryItem] = {}
        
        self.vector_plugin = vector_plugin
        self.decay_rate = decay_rate
        
        # Contradiction tracking
        self.contradictions: list[tuple[str, str, str]] = []  # (id1, id2, reason)
        
        logger.info("memory_system_initialized", decay_rate=decay_rate)
    
    def store(
        self,
        content: dict[str, Any],
        memory_type: MemoryType = MemoryType.SEMANTIC,
        source: str = "observation",
        confidence: float = 1.0,
        valid_until: Optional[datetime] = None,
    ) -> MemoryItem:
        """
        Store a new memory item.
        
        Args:
            content: The content to store
            memory_type: Type of memory (semantic, episodic, temporal)
            source: Where this knowledge came from
            confidence: How confident we are in this knowledge
            valid_until: When this knowledge expires
            
        Returns:
            The stored memory item
        """
        item = MemoryItem(
            type=memory_type,
            content=content,
            source=source,
            confidence=confidence,
            valid_until=valid_until,
        )
        
        # Store in appropriate storage
        if memory_type == MemoryType.SEMANTIC:
            self.semantic[item.id] = item
        elif memory_type == MemoryType.TEMPORAL:
            self.temporal[item.id] = item
        
        # Store in vector DB for semantic search
        if self.vector_plugin:
            self.vector_plugin.store_memory(item)
        
        logger.info(
            "memory_stored",
            memory_id=item.id,
            type=memory_type.value,
            source=source,
        )
        
        return item
    
    def store_episode(
        self,
        question: str,
        answer: str,
        outcome: str,
        reasoning_strategy: str = "",
        confidence: float = 1.0,
        context: Optional[dict] = None,
        failure_reason: Optional[str] = None,
    ) -> EpisodicMemory:
        """
        Store an episodic memory (experience).
        
        Args:
            question: The question asked
            answer: The answer generated
            outcome: PASS, FAIL, or PARTIAL
            reasoning_strategy: Strategy used
            confidence: Confidence in the answer
            context: Additional context
            failure_reason: Why it failed (if applicable)
            
        Returns:
            The stored episodic memory
        """
        episode = EpisodicMemory(
            question=question,
            answer=answer,
            outcome=outcome,
            reasoning_strategy=reasoning_strategy,
            confidence=confidence,
            context=context or {},
            failure_reason=failure_reason,
        )
        
        self.episodic[episode.id] = episode
        
        logger.info(
            "episode_stored",
            episode_id=episode.id,
            outcome=outcome,
            strategy=reasoning_strategy,
        )
        
        return episode
    
    def recall(
        self,
        query: str,
        context: Optional[dict] = None,
        memory_types: Optional[list[MemoryType]] = None,
        limit: int = 10,
        min_strength: float = 0.1,
    ) -> list[MemoryItem | EpisodicMemory]:
        """
        Recall memories relevant to a query.
        
        Args:
            query: The query to search for
            context: Additional context for filtering
            memory_types: Types of memory to search
            limit: Maximum number of results
            min_strength: Minimum memory strength to include
            
        Returns:
            List of relevant memories
        """
        results: list[MemoryItem | EpisodicMemory] = []
        memory_types = memory_types or [MemoryType.SEMANTIC, MemoryType.EPISODIC]
        now = datetime.utcnow()
        
        # Search semantic memory
        if MemoryType.SEMANTIC in memory_types:
            for item in self.semantic.values():
                if item.strength >= min_strength:
                    # Check temporal validity
                    if item.valid_until and item.valid_until < now:
                        continue
                    
                    # Update access metadata
                    item.access_count += 1
                    item.last_accessed = now
                    
                    results.append(item)
        
        # Search episodic memory
        if MemoryType.EPISODIC in memory_types:
            for episode in self.episodic.values():
                results.append(episode)
        
        # Use vector search if available
        if self.vector_plugin:
            vector_results = self.vector_plugin.search(query, limit=limit)
            # Merge with local results
            seen_ids = {r.id for r in results}
            for item in vector_results:
                if item.id not in seen_ids:
                    results.append(item)
        
        # Sort by relevance (using recency as proxy for now)
        results.sort(
            key=lambda x: x.timestamp if isinstance(x, (MemoryItem, EpisodicMemory)) else datetime.min,
            reverse=True
        )
        
        logger.debug(
            "memory_recalled",
            query=query[:50],
            result_count=len(results[:limit]),
        )
        
        return results[:limit]
    
    def decay(self) -> dict[str, int]:
        """
        Apply time-based decay to memories.
        
        Forgetting is a feature, not a bug.
        
        Returns:
            Statistics about the decay process
        """
        now = datetime.utcnow()
        decayed_count = 0
        forgotten_count = 0
        
        # Decay semantic memories
        for item in list(self.semantic.values()):
            # Calculate time since last access
            last_access = item.last_accessed or item.timestamp
            hours_since_access = (now - last_access).total_seconds() / 3600
            
            # Apply decay based on time and access frequency
            access_factor = max(0.5, 1.0 - (1.0 / (item.access_count + 1)))
            decay_amount = self.decay_rate * hours_since_access * 0.01 * (1 - access_factor)
            
            item.strength = max(0, item.strength - decay_amount)
            
            if item.strength < 0.1:
                # Memory is too weak, forget it
                del self.semantic[item.id]
                forgotten_count += 1
            else:
                decayed_count += 1
        
        # Decay temporal memories
        expired_count = 0
        for item in list(self.temporal.values()):
            if item.valid_until and item.valid_until < now:
                del self.temporal[item.id]
                expired_count += 1
        
        logger.info(
            "memory_decay_applied",
            decayed=decayed_count,
            forgotten=forgotten_count,
            expired=expired_count,
        )
        
        return {
            "decayed": decayed_count,
            "forgotten": forgotten_count,
            "expired": expired_count,
        }
    
    def record_contradiction(
        self,
        item1_id: str,
        item2_id: str,
        reason: str,
    ) -> None:
        """
        Record a contradiction between two memories.
        
        Contradictions are recorded, not deleted.
        
        Args:
            item1_id: First memory item
            item2_id: Second memory item
            reason: Why they contradict
        """
        self.contradictions.append((item1_id, item2_id, reason))
        
        # Update the items themselves
        if item1_id in self.semantic:
            self.semantic[item1_id].contradicted_by.append(item2_id)
            self.semantic[item1_id].contradicts.append(item2_id)
        if item2_id in self.semantic:
            self.semantic[item2_id].contradicted_by.append(item1_id)
            self.semantic[item2_id].contradicts.append(item1_id)
        
        logger.info(
            "contradiction_recorded",
            item1=item1_id,
            item2=item2_id,
            reason=reason,
        )
    
    def get_contradictions(self) -> list[dict]:
        """Get all recorded contradictions."""
        return [
            {
                "item1_id": c[0],
                "item2_id": c[1],
                "reason": c[2],
            }
            for c in self.contradictions
        ]
    
    def get_failed_episodes(
        self,
        limit: int = 100,
    ) -> list[EpisodicMemory]:
        """
        Get episodes that failed.
        
        Failure is signal, not error.
        
        Args:
            limit: Maximum number to return
            
        Returns:
            List of failed episodes
        """
        failed = [
            e for e in self.episodic.values()
            if e.outcome == "FAIL"
        ]
        failed.sort(key=lambda x: x.timestamp, reverse=True)
        return failed[:limit]
    
    def get_stats(self) -> dict:
        """Get statistics about the memory system."""
        return {
            "semantic_count": len(self.semantic),
            "episodic_count": len(self.episodic),
            "temporal_count": len(self.temporal),
            "contradiction_count": len(self.contradictions),
            "failed_episodes": sum(
                1 for e in self.episodic.values() if e.outcome == "FAIL"
            ),
            "passed_episodes": sum(
                1 for e in self.episodic.values() if e.outcome == "PASS"
            ),
            "average_strength": (
                sum(m.strength for m in self.semantic.values()) / len(self.semantic)
                if self.semantic else 0
            ),
        }
