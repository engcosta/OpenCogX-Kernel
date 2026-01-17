"""
📥 Ingestion Module
===================

Phase 1 of the POC: Load and process documents.

Responsibilities:
- Hierarchical chunking
- Vector storage
- Graph extraction (entities + relations)
- Deduplication (skip already-ingested content)

Note: We don't care about high accuracy here.
We care that the system builds "something" to think about.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Set
import structlog

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import print as rprint

from agi_kernel.core.memory import Memory, MemoryType
from agi_kernel.core.world import WorldModel, Event
from agi_kernel.plugins.llm import LLMPlugin
from agi_kernel.plugins.vector import VectorPlugin
from agi_kernel.plugins.graph import GraphPlugin

logger = structlog.get_logger()
console = Console()


def content_hash(content: str) -> str:
    """Generate a deterministic hash from content."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class Chunk:
    """A chunk of text from a document."""
    id: str
    content: str
    source: str
    level: int  # 0 = document, 1 = section, 2 = paragraph
    parent_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    content_hash: str = ""
    
    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = content_hash(self.content)


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
    4. Vector Storage (with deduplication)
    5. Graph Storage (with deduplication)
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
        
        # Track ingested content to avoid duplicates
        self._ingested_files: Set[str] = set()  # file content hashes
        self._ingested_chunks: Set[str] = set()  # chunk content hashes
        
        # Dynamic entity type discovery (per document)
        self._document_entity_types: dict[str, list[str]] = {}  # source -> types
        self._default_entity_types = ["person", "organization", "location", "concept", "technology"]
        
        # Stats
        self.chunks_processed = 0
        self.chunks_skipped = 0
        self.entities_extracted = 0
        self.entities_skipped = 0
        self.relations_extracted = 0
        self.relations_skipped = 0
        
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
    
    async def _discover_entity_types(self, content: str, source: str) -> list[str]:
        """
        Use LLM to discover domain-specific entity types from document content.
        
        This makes entity extraction more intelligent by adapting to the document's domain.
        
        Args:
            content: Document content (first ~1000 chars)
            source: Source file name
            
        Returns:
            List of discovered entity types for this document
        """
        # Check cache first
        if source in self._document_entity_types:
            return self._document_entity_types[source]
        
        # If no LLM available, use defaults
        if not self.llm:
            return self._default_entity_types
        
        try:
            prompt = f"""Analyze this document and identify the types of entities that should be extracted.

Document excerpt:
{content[:1500]}

Based on this content, list 5-10 specific entity types that would be valuable to extract.
Focus on domain-specific types relevant to this content.

Examples of entity types:
- For tech docs: protocol, algorithm, technology, pattern, component, service
- For business: organization, product, metric, process, stakeholder
- For science: concept, theory, method, phenomenon, measurement

Format: Return ONLY a comma-separated list of entity types (lowercase, singular form).
Example: protocol, algorithm, consistency_model, replication_strategy, failure_mode
"""
            
            response = await self.llm.generate(
                prompt=prompt,
                model_type="quick",
                temperature=0.3,
            )
            
            # Parse response
            types = []
            for part in response.strip().split(','):
                entity_type = part.strip().lower().replace(' ', '_')
                if entity_type and len(entity_type) > 1:
                    types.append(entity_type)
            
            # Always include some base types
            base_types = ["concept", "technology"]
            for bt in base_types:
                if bt not in types:
                    types.append(bt)
            
            # Cache the discovered types
            self._document_entity_types[source] = types
            
            logger.info("entity_types_discovered", source=source, types=types)
            console.print(f"   [dim]🔍 Discovered entity types: {', '.join(types)}[/dim]")
            
            return types
            
        except Exception as e:
            logger.warning("entity_type_discovery_failed", error=str(e))
            return self._default_entity_types
    
    async def ingest_file(self, file_path: str, force: bool = False, verbose: bool = True) -> dict:
        """
        Ingest a single file.
        
        Args:
            file_path: Path to the file
            force: If True, re-ingest even if already processed
            verbose: If True, print detailed console output
            
        Returns:
            Statistics about the ingestion
        """
        path = Path(file_path)
        
        if not path.exists():
            if verbose:
                console.print(f"[red]❌ File not found:[/red] {file_path}")
            logger.error("file_not_found", path=file_path)
            return {"error": "File not found"}
        
        # Read content
        file_content = path.read_text(encoding="utf-8")
        file_hash = content_hash(file_content)
        file_size = len(file_content)
        
        if verbose:
            console.print(f"\n[cyan]📄 Processing:[/cyan] {path.name}")
            console.print(f"   [dim]Hash: {file_hash} | Size: {file_size:,} bytes[/dim]")
        
        # Check if file already ingested
        if not force and file_hash in self._ingested_files:
            if verbose:
                console.print(f"[yellow]⏭️  Skipped:[/yellow] Already ingested (same content hash)")
            logger.info("file_already_ingested", path=file_path, hash=file_hash)
            return {
                "file": path.name,
                "skipped": True,
                "reason": "Already ingested (same content hash)",
            }
        
        logger.info("ingesting_file", path=file_path, hash=file_hash)
        
        # Discover domain-specific entity types using LLM
        if self.llm:
            await self._discover_entity_types(file_content, path.name)
        
        # Chunk the document
        chunks = self._hierarchical_chunk(file_content, source=path.name)
        
        if verbose:
            console.print(f"   [dim]Chunks created: {len(chunks)}[/dim]")
        
        # Process chunks
        entities: list[Entity] = []
        relations: list[Relation] = []
        chunks_new = 0
        chunks_skipped = 0
        vectors_stored = 0
        
        # Use progress bar for chunk processing
        if verbose:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("[cyan]Processing chunks...", total=len(chunks))
                
                for i, chunk in enumerate(chunks):
                    # Check if chunk already exists (by content hash)
                    if not force and chunk.content_hash in self._ingested_chunks:
                        chunks_skipped += 1
                        self.chunks_skipped += 1
                        progress.update(task, advance=1)
                        continue
                    
                    # Mark as ingested
                    self._ingested_chunks.add(chunk.content_hash)
                    
                    # Store in memory
                    self.memory.store(
                        content={"text": chunk.content, "source": chunk.source, "level": chunk.level},
                        memory_type=MemoryType.SEMANTIC,
                        source="ingestion",
                    )
                    
                    # Store in vector DB (has its own dedup)
                    if self.vector:
                        stored = await self._store_chunk_vector(chunk)
                        if stored:
                            vectors_stored += 1
                    
                    # Extract entities
                    chunk_entities = await self._extract_entities(chunk)
                    entities.extend(chunk_entities)
                    
                    # Extract relations
                    chunk_relations = await self._extract_relations(chunk, chunk_entities)
                    relations.extend(chunk_relations)
                    
                    self.chunks_processed += 1
                    chunks_new += 1
                    progress.update(task, advance=1)
        else:
            # Non-verbose mode
            for chunk in chunks:
                if not force and chunk.content_hash in self._ingested_chunks:
                    chunks_skipped += 1
                    self.chunks_skipped += 1
                    continue
                
                self._ingested_chunks.add(chunk.content_hash)
                
                self.memory.store(
                    content={"text": chunk.content, "source": chunk.source, "level": chunk.level},
                    memory_type=MemoryType.SEMANTIC,
                    source="ingestion",
                )
                
                if self.vector:
                    stored = await self._store_chunk_vector(chunk)
                    if stored:
                        vectors_stored += 1
                
                chunk_entities = await self._extract_entities(chunk)
                entities.extend(chunk_entities)
                
                chunk_relations = await self._extract_relations(chunk, chunk_entities)
                relations.extend(chunk_relations)
                
                self.chunks_processed += 1
                chunks_new += 1
        
        # Store in graph (uses MERGE for dedup)
        if self.graph:
            await self._store_graph(entities, relations)
        
        # Mark file as ingested
        self._ingested_files.add(file_hash)
        
        # Print results table
        if verbose:
            table = Table(title="📊 Ingestion Results", show_header=True, header_style="bold cyan")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_column("Status", style="yellow")
            
            table.add_row("Chunks New", str(chunks_new), "✅ Inserted")
            table.add_row("Chunks Skipped", str(chunks_skipped), "⏭️  Duplicate" if chunks_skipped > 0 else "—")
            table.add_row("Vectors Stored", str(vectors_stored), "✅ Indexed" if vectors_stored > 0 else "⚠️ No LLM")
            table.add_row("Entities Found", str(len(entities)), f"🏷️  {', '.join(set(e.type for e in entities[:5]))}" if entities else "—")
            table.add_row("Relations Found", str(len(relations)), f"🔗 {', '.join(set(r.relation_type for r in relations[:5]))}" if relations else "—")
            
            console.print(table)
            
            # Show sample entities if any
            if entities:
                console.print(f"\n[dim]Sample entities:[/dim]")
                for e in entities[:5]:
                    console.print(f"   [green]• {e.name}[/green] [dim]({e.type})[/dim]")
                if len(entities) > 5:
                    console.print(f"   [dim]... and {len(entities) - 5} more[/dim]")
        
        # Record ingestion event in world model
        event = Event(
            actor="ingestion_pipeline",
            action="ingest_file",
            context={
                "file": path.name,
                "file_hash": file_hash,
                "chunks_new": chunks_new,
                "chunks_skipped": chunks_skipped,
                "entities": len(entities),
                "relations": len(relations),
            },
        )
        self.world.observe(event)
        
        logger.info(
            "file_ingested",
            path=file_path,
            chunks_new=chunks_new,
            chunks_skipped=chunks_skipped,
            entities=len(entities),
            relations=len(relations),
        )
        
        return {
            "file": path.name,
            "file_hash": file_hash,
            "chunks": chunks_new,
            "chunks_skipped": chunks_skipped,
            "vectors_stored": vectors_stored,
            "entities": len(entities),
            "relations": len(relations),
        }
    
    async def ingest_directory(self, directory: str, extensions: list[str] = None, verbose: bool = True) -> dict:
        """
        Ingest all files in a directory.
        
        Args:
            directory: Directory path
            extensions: File extensions to include (default: [".txt", ".md"])
            verbose: If True, print detailed console output
            
        Returns:
            Summary statistics
        """
        extensions = extensions or [".txt", ".md", ".json"]
        directory_path = Path(directory)
        
        if not directory_path.exists():
            if verbose:
                console.print(f"[red]❌ Directory not found:[/red] {directory}")
            logger.error("directory_not_found", path=directory)
            return {"error": "Directory not found"}
        
        # Find all matching files
        files = []
        for ext in extensions:
            files.extend(directory_path.glob(f"*{ext}"))
        
        if verbose:
            console.print(Panel(
                f"[bold cyan]📁 Ingesting Directory: {directory_path.name}[/bold cyan]\n"
                f"[dim]Path: {directory_path.absolute()}[/dim]\n"
                f"[dim]Extensions: {', '.join(extensions)}[/dim]\n"
                f"[dim]Files found: {len(files)}[/dim]",
                title="🔄 Batch Ingestion",
                border_style="cyan"
            ))
        
        results = []
        files_new = 0
        files_skipped = 0
        
        for file_path in files:
            result = await self.ingest_file(str(file_path), verbose=verbose)
            results.append(result)
            
            if result.get("skipped"):
                files_skipped += 1
            else:
                files_new += 1
        
        # Summary
        total_chunks = sum(r.get("chunks", 0) for r in results)
        total_chunks_skipped = sum(r.get("chunks_skipped", 0) for r in results)
        total_vectors = sum(r.get("vectors_stored", 0) for r in results)
        total_entities = sum(r.get("entities", 0) for r in results)
        total_relations = sum(r.get("relations", 0) for r in results)
        
        if verbose:
            summary_table = Table(title="📊 Directory Ingestion Summary", show_header=True, header_style="bold green")
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Count", style="green")
            
            summary_table.add_row("Files Processed", str(files_new))
            summary_table.add_row("Files Skipped (Duplicate)", str(files_skipped))
            summary_table.add_row("Total Chunks Inserted", str(total_chunks))
            summary_table.add_row("Total Chunks Skipped", str(total_chunks_skipped))
            summary_table.add_row("Total Vectors Stored", str(total_vectors))
            summary_table.add_row("Total Entities Extracted", str(total_entities))
            summary_table.add_row("Total Relations Extracted", str(total_relations))
            
            console.print("\n")
            console.print(summary_table)
            console.print(f"\n[green]✅ Ingestion complete![/green]\n")
        
        return {
            "files_processed": files_new,
            "files_skipped": files_skipped,
            "total_chunks": total_chunks,
            "total_chunks_skipped": total_chunks_skipped,
            "total_vectors": total_vectors,
            "total_entities": total_entities,
            "total_relations": total_relations,
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
                # Use dynamic labels if discovered, otherwise defaults
                labels = self._document_entity_types.get(chunk.source, self._default_entity_types)
                
                predictions = self._gliner.predict_entities(
                    chunk.content,
                    labels,
                    threshold=0.5,
                )
                
                for pred in predictions:
                    # Create readable entity ID from name
                    name = pred["text"].strip()
                    entity_id = self._make_entity_id(name)
                    entity_type = pred["label"].upper()
                    
                    entity = Entity(
                        id=entity_id,
                        name=name,
                        type=entity_type,
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
    
    def _make_entity_id(self, name: str) -> str:
        """Create a readable, unique entity ID from name."""
        import re
        # Slugify: lowercase, replace spaces/special chars with underscores
        slug = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
        # Truncate if too long
        if len(slug) > 50:
            slug = slug[:50]
        return slug or f"entity_{hash(name) % 100000}"
    
    async def _extract_entities_llm(self, chunk: Chunk) -> list[Entity]:
        """Extract entities using LLM."""
        if not self.llm:
            return []
        
        # Get types for this document
        types = self._document_entity_types.get(chunk.source, self._default_entity_types)
        types_str = ", ".join(types)
        
        prompt = f"""Extract key entities from this text.
        
Target Entity Types: {types_str}

Text:
{chunk.content[:600]}

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
                    entity_type = parts[1].strip().upper()
                    
                    # Validate type (fuzzy match)
                    if not any(t.upper() in entity_type for t in types):
                        # If LLM hallucinated a type, map to nearest or generic
                        entity_type = "CONCEPT"
                    
                    entity = Entity(
                        id=self._make_entity_id(name),
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
    
    async def _store_chunk_vector(self, chunk: Chunk) -> bool:
        """Store chunk in vector database. Returns True if stored."""
        if not self.vector or not self.llm:
            return False
        
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
            
            result = await self.vector.store_memory(item, embedding=embedding)
            return result
            
        except Exception as e:
            logger.warning("vector_store_failed", chunk=chunk.id, error=str(e))
            return False
    
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
                properties={"name": entity.name, "source": entity.source, **entity.properties},
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
            "chunks_skipped": self.chunks_skipped,
            "entities_extracted": self.entities_extracted,
            "entities_skipped": self.entities_skipped,
            "relations_extracted": self.relations_extracted,
            "relations_skipped": self.relations_skipped,
            "files_ingested": len(self._ingested_files),
            "unique_chunks": len(self._ingested_chunks),
        }
