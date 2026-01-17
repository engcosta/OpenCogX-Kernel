"""
🔗 Graph Plugin
===============

Adapter for Graph Database (Neo4j).

This is a PLUGIN, not part of the kernel.
Provides knowledge graph capabilities.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional, TYPE_CHECKING, Any
import structlog

if TYPE_CHECKING:
    from agi_kernel.core.world import State, Event, StateTransition
    from agi_kernel.core.memory import MemoryItem

logger = structlog.get_logger()


def content_hash(content: any) -> str:
    """Generate a deterministic hash from content."""
    if isinstance(content, dict):
        content_str = json.dumps(content, sort_keys=True)
    else:
        content_str = str(content)
    return hashlib.sha256(content_str.encode()).hexdigest()[:16]


class GraphPlugin:
    """
    Graph Database Plugin using Neo4j.
    
    Provides:
    - Knowledge graph storage
    - Entity and relation management
    - Multi-hop reasoning paths
    - Knowledge gap detection
    - Deduplication via content hashing
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Initialize the Graph Plugin.
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        
        self._driver = None
        self._initialized = False
        
        logger.info(
            "graph_plugin_created",
            uri=self.uri,
            user=self.user,
        )
    
    async def initialize(self) -> bool:
        """
        Initialize connection to Neo4j.
        
        Returns:
            True if successful
        """
        try:
            from neo4j import AsyncGraphDatabase
            
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            
            # Verify connection
            async with self._driver.session() as session:
                result = await session.run("RETURN 1 AS test")
                await result.single()
            
            # Create constraints/indexes
            await self._setup_schema()
            
            self._initialized = True
            logger.info("graph_plugin_initialized")
            return True
            
        except Exception as e:
            logger.error("graph_plugin_init_failed", error=str(e))
            return False
    
    async def _setup_schema(self) -> None:
        """Set up graph schema constraints and indexes."""
        async with self._driver.session() as session:
            # Create constraints for unique IDs
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (s:State) REQUIRE s.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (ev:Event) REQUIRE ev.id IS UNIQUE",
                # Add index for content_hash to speed up dedup lookups
                "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.content_hash)",
            ]
            
            for constraint in constraints:
                try:
                    await session.run(constraint)
                except Exception as e:
                    logger.debug("constraint_exists_or_failed", error=str(e))
    
    async def entity_exists(self, entity_id: str) -> bool:
        """
        Check if an entity already exists.
        
        Args:
            entity_id: Entity ID to check
            
        Returns:
            True if entity exists
        """
        if not self._initialized:
            return False
        
        try:
            async with self._driver.session() as session:
                result = await session.run(
                    "MATCH (e:Entity {id: $id}) RETURN count(e) > 0 AS exists",
                    id=entity_id,
                )
                record = await result.single()
                return record["exists"] if record else False
        except Exception:
            return False
    
    async def relation_exists(self, from_id: str, to_id: str, relation_type: str) -> bool:
        """
        Check if a relation already exists.
        
        Args:
            from_id: Source entity ID
            to_id: Target entity ID
            relation_type: Relation type
            
        Returns:
            True if relation exists
        """
        if not self._initialized:
            return False
        
        try:
            safe_relation = relation_type.upper().replace(" ", "_")
            async with self._driver.session() as session:
                query = f"""
                MATCH (a:Entity {{id: $from_id}})-[r:{safe_relation}]->(b:Entity {{id: $to_id}})
                RETURN count(r) > 0 AS exists
                """
                result = await session.run(query, from_id=from_id, to_id=to_id)
                record = await result.single()
                return record["exists"] if record else False
        except Exception:
            return False
    
    async def store_entity(
        self,
        entity_id: str,
        entity_type: str,
        properties: dict[str, Any],
    ) -> bool:
        """
        Store an entity in the graph with proper labels and properties.
        
        Args:
            entity_id: Unique entity identifier
            entity_type: Type of entity (e.g., "Person", "Concept", "Technology")
            properties: Entity properties (must include 'name')
            
        Returns:
            True if successful
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get entity name for display (required)
            entity_name = properties.get("name", entity_id)
            
            # Sanitize entity type for use as Neo4j label
            safe_type = entity_type.replace(" ", "_").replace("-", "_").title()
            if not safe_type:
                safe_type = "Entity"
            
            # Prepare properties to store (dynamic)
            node_props = properties.copy()
            node_props.update({
                "name": entity_name,
                "type": entity_type,
                "source": properties.get("source", "unknown"),
            })
            
            # Exclude id from props if it matches to avoid duplication/confusion
            if "id" in node_props:
                del node_props["id"]

            async with self._driver.session() as session:
                # Create node with both :Entity label and type-specific label
                # Use += to dynamically set all properties from the dict
                query = f"""
                MERGE (e:Entity:{safe_type} {{id: $id}})
                SET e += $props,
                    e.updated_at = datetime()
                RETURN e
                """
                await session.run(
                    query,
                    id=entity_id,
                    props=node_props,
                )
            
            logger.debug("entity_stored", entity_id=entity_id, name=entity_name, type=safe_type)
            return True
            
        except Exception as e:
            logger.error("entity_store_failed", error=str(e))
            return False
    
    async def store_relation(
        self,
        from_entity: str,
        to_entity: str,
        relation_type: str,
        properties: Optional[dict] = None,
    ) -> bool:
        """
        Store a relation between entities.
        
        Args:
            from_entity: Source entity ID
            to_entity: Target entity ID
            relation_type: Type of relation (e.g., "RELATED_TO", "CAUSES")
            properties: Relation properties
            
        Returns:
            True if successful
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self._driver.session() as session:
                # Ensure relation type is valid for Cypher
                safe_relation = relation_type.upper().replace(" ", "_")
                
                query = f"""
                MATCH (a:Entity {{id: $from_id}})
                MATCH (b:Entity {{id: $to_id}})
                MERGE (a)-[r:{safe_relation}]->(b)
                SET r.properties = $properties,
                    r.created_at = datetime()
                RETURN r
                """
                await session.run(
                    query,
                    from_id=from_entity,
                    to_id=to_entity,
                    properties=str(properties or {}),
                )
                
            logger.debug(
                "relation_stored",
                from_entity=from_entity,
                to_entity=to_entity,
                relation=relation_type,
            )
            return True
            
        except Exception as e:
            logger.error("relation_store_failed", error=str(e))
            return False
    
    async def store_state(self, state: State) -> bool:
        """Store a world state in the graph."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Create meaningful name for display
            try:
                features_dict = eval(str(state.features)) if isinstance(state.features, str) else state.features
                action_name = features_dict.get("action", "Unknown")
                state_name = f"State: {action_name}"
            except:
                state_name = "State"

            async with self._driver.session() as session:
                query = """
                MERGE (s:State {id: $id})
                SET s.features = $features,
                    s.timestamp = $timestamp,
                    s.confidence = $confidence,
                    s.source = $source,
                    s.name = $name
                RETURN s
                """
                await session.run(
                    query,
                    id=state.id,
                    features=str(state.features),
                    timestamp=state.timestamp.isoformat(),
                    confidence=state.confidence,
                    source=state.source,
                    name=state_name,
                )
                
            logger.debug("state_stored", state_id=state.id)
            return True
            
        except Exception as e:
            logger.error("state_store_failed", error=str(e))
            return False
    
    async def store_event(self, event: Event) -> bool:
        """Store an event in the graph."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Improve naming for specific events
            if event.action == "ingest_file" and isinstance(event.context, dict) and "file" in event.context:
                 event_name = f"Ingestion: {event.context['file']}"
            else:
                 event_name = f"Event: {event.action}"
            
            # Prepare properties
            event_props = {
                "actor": event.actor,
                "action": event.action,
                "name": event_name,
                "timestamp": event.timestamp.isoformat(),
            }
            
            # Flatten context properties into top-level props
            if isinstance(event.context, dict):
                for k, v in event.context.items():
                    # Validate keys to avoid collisions or reserved words if needed
                    event_props[k] = v
            else:
                event_props["context"] = str(event.context)

            async with self._driver.session() as session:
                query = """
                MERGE (ev:Event {id: $id})
                SET ev += $props
                RETURN ev
                """
                await session.run(
                    query,
                    id=event.id,
                    props=event_props,
                )
                
            logger.debug("event_stored", event_id=event.id)
            return True
            
        except Exception as e:
            logger.error("event_store_failed", error=str(e))
            return False
    
    async def store_transition(self, transition: StateTransition) -> bool:
        """Store a state transition in the graph."""
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self._driver.session() as session:
                query = """
                MATCH (s1:State {id: $from_id})
                MATCH (s2:State {id: $to_id})
                MERGE (s1)-[t:LEADS_TO]->(s2)
                SET t.probability = $probability,
                    t.relation = $relation,
                    t.evidence_count = $evidence
                RETURN t
                """
                await session.run(
                    query,
                    from_id=transition.from_state_id,
                    to_id=transition.to_state_id,
                    probability=transition.probability,
                    relation=transition.relation.value,
                    evidence=transition.evidence_count,
                )
                
            logger.debug(
                "transition_stored",
                from_state=transition.from_state_id,
                to_state=transition.to_state_id,
            )
            return True
            
        except Exception as e:
            logger.error("transition_store_failed", error=str(e))
            return False
    
    async def find_path(
        self,
        start_entity: str,
        end_entity: str,
        max_hops: int = 5,
    ) -> list[dict]:
        """
        Find path between two entities.
        
        Args:
            start_entity: Starting entity ID
            end_entity: Target entity ID
            max_hops: Maximum path length
            
        Returns:
            List of path steps
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self._driver.session() as session:
                query = f"""
                MATCH path = shortestPath(
                    (a:Entity {{id: $start}})-[*1..{max_hops}]-(b:Entity {{id: $end}})
                )
                RETURN [n IN nodes(path) | n.id] AS node_ids,
                       [r IN relationships(path) | type(r)] AS relation_types
                """
                result = await session.run(
                    query,
                    start=start_entity,
                    end=end_entity,
                )
                record = await result.single()
                
                if record:
                    return {
                        "node_ids": record["node_ids"],
                        "relations": record["relation_types"],
                    }
                return {}
                
        except Exception as e:
            logger.error("path_find_failed", error=str(e))
            return {}
    
    async def find_knowledge_gaps(
        self,
        min_relations: int = 2,
    ) -> list[dict]:
        """
        Find entities with few relations (knowledge gaps).
        
        Args:
            min_relations: Threshold for "few" relations
            
        Returns:
            List of under-connected entities
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self._driver.session() as session:
                query = """
                MATCH (e:Entity)
                WITH e, size([(e)-[]-() | 1]) AS rel_count
                WHERE rel_count < $min_relations
                RETURN e.id AS entity, e.type AS type, rel_count
                ORDER BY rel_count ASC
                LIMIT 20
                """
                result = await session.run(query, min_relations=min_relations)
                records = await result.values()
                
                return [
                    {
                        "entity": r[0],
                        "type": r[1],
                        "relation_count": r[2],
                    }
                    for r in records
                ]
                
        except Exception as e:
            logger.error("gap_detection_failed", error=str(e))
            return []
    
    async def get_entity_neighbors(
        self,
        entity_id: str,
        depth: int = 1,
    ) -> list[dict]:
        """
        Get neighbors of an entity up to a certain depth.
        
        Args:
            entity_id: Entity to explore
            depth: How many hops to traverse
            
        Returns:
            List of neighbor entities and relations
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self._driver.session() as session:
                query = f"""
                MATCH (e:Entity {{id: $id}})-[r*1..{depth}]-(n:Entity)
                RETURN DISTINCT n.id AS neighbor_id, 
                       n.type AS neighbor_type,
                       [rel IN r | type(rel)] AS relations
                LIMIT 50
                """
                result = await session.run(query, id=entity_id)
                records = await result.values()
                
                return [
                    {
                        "id": r[0],
                        "type": r[1],
                        "relations": r[2],
                    }
                    for r in records
                ]
                
        except Exception as e:
            logger.error("neighbor_fetch_failed", error=str(e))
            return []
    
    async def run_cypher(
        self,
        query: str,
        parameters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Run a custom Cypher query.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            Query results
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self._driver.session() as session:
                result = await session.run(query, parameters or {})
                records = await result.values()
                return [dict(zip(result.keys(), r)) for r in records]
                
        except Exception as e:
            logger.error("cypher_query_failed", error=str(e))
            return []
    
    def get_stats(self) -> dict:
        """Get statistics about the graph database."""
        return {"initialized": self._initialized, "uri": self.uri}
    
    async def close(self) -> None:
        """Close the database connection."""
        if self._driver:
            await self._driver.close()
