"""
📥 Ingestion Module
===================

Phase 1 of the POC: Load and process documents.

Responsibilities:
- Hierarchical chunking
- Vector storage
- Graph extraction (entities + relations)

Note: We don't care about high accuracy here.
We care that the system builds "something" to think about.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import structlog

from agi_kernel.core.memory import Memory, MemoryType
from agi_kernel.core.world import WorldModel, Event
from agi_kernel.plugins.llm import LLMPlugin
from agi_kernel.plugins.vector import VectorPlugin
from agi_kernel.plugins.graph import GraphPlugin

logger = structlog.get_logger()


@dataclass
class Chunk:
    """A chunk of text from a document."""
    id: str
    content: str
    source: str
    level: int  # 0 = document, 1 = section, 2 = paragraph
    parent_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Entity:
    """An extracted entity."""
    id: str
    name: str
    type: str
    source: str
    properties: dict = field(default_factory=dict)


@dataclass
class Relation:
    """A relation between entities."""
    from_entity: str
    to_entity: str
    relation_type: str
    source: str
    confidence: float = 1.0


class IngestionPipeline:
    """
    Document ingestion pipeline.
    
    Processes documents through:
    1. Hierarchical Chunking
    2. Entity Extraction (using GLiNER or LLM)
    3. Relation Extraction
    4. Vector Storage
    5. Graph Storage
    """
    
    def __init__(
        self,
        memory: Memory,
        world: WorldModel,
        llm_plugin: Optional[LLMPlugin] = None,
        vector_plugin: Optional[VectorPlugin] = None,
        graph_plugin: Optional[GraphPlugin] = None,
        use_gliner: bool = True,
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            memory: Memory system for storage
            world: World model for state tracking
            llm_plugin: LLM for extraction
            vector_plugin: Vector DB for embeddings
            graph_plugin: Graph DB for entities/relations
            use_gliner: Use GLiNER for entity extraction
        """
        self.memory = memory
        self.world = world
        self.llm = llm_plugin
        self.vector = vector_plugin
        self.graph = graph_plugin
        self.use_gliner = use_gliner
        
        # GLiNER model (lazy loaded)
        self._gliner = None
        
        # Stats
        self.chunks_processed = 0
        self.entities_extracted = 0
        self.relations_extracted = 0
        
        logger.info("ingestion_pipeline_initialized")
    
    def _load_gliner(self):
        """Lazy load GLiNER model."""
        if self._gliner is None and self.use_gliner:
            try:
                from gliner import GLiNER
                self._gliner = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
                logger.info("gliner_model_loaded")
            except Exception as e:
                logger.warning("gliner_load_failed", error=str(e))
                self._gliner = False  # Mark as failed
    
    async def ingest_file(self, file_path: str) -> dict:
        """
        Ingest a single file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Statistics about the ingestion
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.error("file_not_found", path=file_path)
            return {"error": "File not found"}
        
        logger.info("ingesting_file", path=file_path)
        
        # Read content
        content = path.read_text(encoding="utf-8")
        
        # Chunk the document
        chunks = self._hierarchical_chunk(content, source=path.name)
        
        # Process chunks
        entities: list[Entity] = []
        relations: list[Relation] = []
        
        for chunk in chunks:
            # Store in memory
            self.memory.store(
                content={"text": chunk.content, "source": chunk.source, "level": chunk.level},
                memory_type=MemoryType.SEMANTIC,
                source="ingestion",
            )
            
            # Store in vector DB
            if self.vector:
                await self._store_chunk_vector(chunk)
            
            # Extract entities
            chunk_entities = await self._extract_entities(chunk)
            entities.extend(chunk_entities)
            
            # Extract relations
            chunk_relations = await self._extract_relations(chunk, chunk_entities)
            relations.extend(chunk_relations)
            
            self.chunks_processed += 1
        
        # Store in graph
        if self.graph:
            await self._store_graph(entities, relations)
        
        # Record ingestion event in world model
        event = Event(
            actor="ingestion_pipeline",
            action="ingest_file",
            context={
                "file": path.name,
                "chunks": len(chunks),
                "entities": len(entities),
                "relations": len(relations),
            },
        )
        self.world.observe(event)
        
        logger.info(
            "file_ingested",
            path=file_path,
            chunks=len(chunks),
            entities=len(entities),
            relations=len(relations),
        )
        
        return {
            "file": path.name,
            "chunks": len(chunks),
            "entities": len(entities),
            "relations": len(relations),
        }
    
    async def ingest_directory(self, directory: str, extensions: list[str] = None) -> dict:
        """
        Ingest all files in a directory.
        
        Args:
            directory: Directory path
            extensions: File extensions to include (default: [".txt", ".md"])
            
        Returns:
            Summary statistics
        """
        extensions = extensions or [".txt", ".md", ".json"]
        directory_path = Path(directory)
        
        if not directory_path.exists():
            logger.error("directory_not_found", path=directory)
            return {"error": "Directory not found"}
        
        results = []
        
        for ext in extensions:
            for file_path in directory_path.glob(f"*{ext}"):
                result = await self.ingest_file(str(file_path))
                results.append(result)
        
        return {
            "files_processed": len(results),
            "total_chunks": sum(r.get("chunks", 0) for r in results),
            "total_entities": sum(r.get("entities", 0) for r in results),
            "total_relations": sum(r.get("relations", 0) for r in results),
            "details": results,
        }
    
    def _hierarchical_chunk(
        self,
        content: str,
        source: str,
        max_chunk_size: int = 500,
    ) -> list[Chunk]:
        """
        Split content into hierarchical chunks.
        
        Levels:
        0 - Full document (summary)
        1 - Sections (by headers)
        2 - Paragraphs
        """
        chunks = []
        chunk_id = 0
        
        # Level 0: Document summary
        doc_id = f"{source}_doc_{chunk_id}"
        chunks.append(Chunk(
            id=doc_id,
            content=content[:1000],  # First 1000 chars as summary
            source=source,
            level=0,
            metadata={"type": "document_summary"},
        ))
        chunk_id += 1
        
        # Split by sections (markdown headers or empty lines)
        section_pattern = r'\n#{1,3} |\n\n+'
        sections = re.split(section_pattern, content)
        
        for i, section in enumerate(sections):
            section = section.strip()
            if not section or len(section) < 50:
                continue
            
            # Level 1: Section
            section_id = f"{source}_sec_{chunk_id}"
            chunks.append(Chunk(
                id=section_id,
                content=section[:max_chunk_size * 2],
                source=source,
                level=1,
                parent_id=doc_id,
                metadata={"section_index": i},
            ))
            chunk_id += 1
            
            # Level 2: Paragraphs within section
            paragraphs = section.split('\n\n')
            for j, para in enumerate(paragraphs):
                para = para.strip()
                if len(para) < 30:
                    continue
                
                # Split large paragraphs
                if len(para) > max_chunk_size:
                    para_chunks = [
                        para[k:k + max_chunk_size]
                        for k in range(0, len(para), max_chunk_size)
                    ]
                else:
                    para_chunks = [para]
                
                for pc in para_chunks:
                    chunks.append(Chunk(
                        id=f"{source}_para_{chunk_id}",
                        content=pc,
                        source=source,
                        level=2,
                        parent_id=section_id,
                        metadata={"paragraph_index": j},
                    ))
                    chunk_id += 1
        
        logger.debug("chunking_complete", source=source, chunks=len(chunks))
        return chunks
    
    async def _extract_entities(self, chunk: Chunk) -> list[Entity]:
        """
        Extract entities from a chunk.
        
        Uses GLiNER if available, otherwise falls back to LLM.
        """
        entities = []
        
        # Try GLiNER first
        self._load_gliner()
        
        if self._gliner and self._gliner is not False:
            try:
                # Define entity types to look for
                labels = ["person", "organization", "location", "concept", "technology", "event"]
                
                predictions = self._gliner.predict_entities(
                    chunk.content,
                    labels,
                    threshold=0.5,
                )
                
                for pred in predictions:
                    entity = Entity(
                        id=f"entity_{hash(pred['text'])}",
                        name=pred["text"],
                        type=pred["label"],
                        source=chunk.source,
                        properties={"span": pred.get("span", [])},
                    )
                    entities.append(entity)
                    self.entities_extracted += 1
                    
            except Exception as e:
                logger.warning("gliner_extraction_failed", error=str(e))
        
        # Fallback to LLM if no entities found
        if not entities and self.llm:
            entities = await self._extract_entities_llm(chunk)
        
        logger.debug("entities_extracted", chunk=chunk.id, count=len(entities))
        return entities
    
    async def _extract_entities_llm(self, chunk: Chunk) -> list[Entity]:
        """Extract entities using LLM."""
        if not self.llm:
            return []
        
        prompt = f"""Extract key entities from this text. For each entity, provide:
- Name
- Type (person, organization, concept, technology, event, location)

Text:
{chunk.content[:500]}

Format each entity on a new line as: NAME | TYPE
"""
        
        response = await self.llm.generate(
            prompt=prompt,
            model_type="solver",
            temperature=0.3,
        )
        
        entities = []
        for line in response.strip().split('\n'):
            if '|' in line:
                parts = line.split('|')
                if len(parts) >= 2:
                    name = parts[0].strip()
                    entity_type = parts[1].strip().lower()
                    
                    entity = Entity(
                        id=f"entity_{hash(name)}",
                        name=name,
                        type=entity_type,
                        source=chunk.source,
                    )
                    entities.append(entity)
                    self.entities_extracted += 1
        
        return entities
    
    async def _extract_relations(
        self,
        chunk: Chunk,
        entities: list[Entity],
    ) -> list[Relation]:
        """
        Extract relations between entities.
        """
        if len(entities) < 2:
            return []
        
        relations = []
        
        if self.llm:
            # Use LLM to find relations
            entity_names = [e.name for e in entities[:10]]  # Limit to 10 entities
            
            prompt = f"""Given these entities: {', '.join(entity_names)}

And this text:
{chunk.content[:400]}

Find relations between entities. Format each relation as:
ENTITY1 | RELATION | ENTITY2

Common relations: causes, enables, precedes, relates_to, part_of, uses
"""
            
            response = await self.llm.generate(
                prompt=prompt,
                model_type="solver",
                temperature=0.3,
            )
            
            for line in response.strip().split('\n'):
                parts = line.split('|')
                if len(parts) >= 3:
                    from_name = parts[0].strip()
                    relation_type = parts[1].strip().lower().replace(' ', '_')
                    to_name = parts[2].strip()
                    
                    # Find matching entities
                    from_entity = next((e for e in entities if e.name.lower() == from_name.lower()), None)
                    to_entity = next((e for e in entities if e.name.lower() == to_name.lower()), None)
                    
                    if from_entity and to_entity:
                        relation = Relation(
                            from_entity=from_entity.id,
                            to_entity=to_entity.id,
                            relation_type=relation_type,
                            source=chunk.source,
                        )
                        relations.append(relation)
                        self.relations_extracted += 1
        
        logger.debug("relations_extracted", chunk=chunk.id, count=len(relations))
        return relations
    
    async def _store_chunk_vector(self, chunk: Chunk) -> None:
        """Store chunk in vector database."""
        if not self.vector or not self.llm:
            return
        
        try:
            # Generate embedding
            embedding = await self.llm.embed(chunk.content)
            
            # Create memory item for storage
            from agi_kernel.core.memory import MemoryItem, MemoryType
            
            item = MemoryItem(
                id=chunk.id,
                type=MemoryType.SEMANTIC,
                content={"text": chunk.content, "source": chunk.source},
                source="ingestion",
            )
            
            await self.vector.store_memory(item, embedding=embedding)
            
        except Exception as e:
            logger.warning("vector_store_failed", chunk=chunk.id, error=str(e))
    
    async def _store_graph(
        self,
        entities: list[Entity],
        relations: list[Relation],
    ) -> None:
        """Store entities and relations in graph database."""
        if not self.graph:
            return
        
        # Store entities
        for entity in entities:
            await self.graph.store_entity(
                entity_id=entity.id,
                entity_type=entity.type,
                properties={"name": entity.name, **entity.properties},
            )
        
        # Store relations
        for relation in relations:
            await self.graph.store_relation(
                from_entity=relation.from_entity,
                to_entity=relation.to_entity,
                relation_type=relation.relation_type,
                properties={"source": relation.source},
            )
    
    def get_stats(self) -> dict:
        """Get ingestion statistics."""
        return {
            "chunks_processed": self.chunks_processed,
            "entities_extracted": self.entities_extracted,
            "relations_extracted": self.relations_extracted,
        }
