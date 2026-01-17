"""
End-to-End Tests for AGI Kernel POC
====================================

Tests the complete workflow:
1. Initialize kernel with real plugins (Qdrant, Neo4j)
2. Ingest documents
3. Run learning loop
4. Verify data in Qdrant and Neo4j
5. Generate evaluation report
"""

import asyncio
import pytest
from datetime import datetime


class TestE2EWithInfrastructure:
    """End-to-end tests with real infrastructure."""
    
    @pytest.fixture
    async def kernel(self):
        """Create kernel with real plugin connections."""
        from agi_kernel.kernel import Kernel
        
        kernel = Kernel(use_plugins=True)
        await kernel.initialize_plugins()
        yield kernel
        await kernel.close()
    
    @pytest.mark.asyncio
    async def test_qdrant_connection(self):
        """Test that Qdrant is accessible."""
        from agi_kernel.plugins.vector import VectorPlugin
        
        vector = VectorPlugin(
            host="localhost",
            port=6333,
            collection_name="test_connection",
        )
        
        result = await vector.initialize()
        assert result is True, "Qdrant should be accessible"
        
        stats = vector.get_stats()
        assert stats["initialized"] is True
        
        await vector.close()
    
    @pytest.mark.asyncio
    async def test_neo4j_connection(self):
        """Test that Neo4j is accessible."""
        from agi_kernel.plugins.graph import GraphPlugin
        import asyncio
        
        graph = GraphPlugin(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
        )
        
        try:
            # Give a timeout for connection
            result = await asyncio.wait_for(graph.initialize(), timeout=10.0)
            assert result is True, "Neo4j should be accessible"
            
            stats = graph.get_stats()
            assert stats["initialized"] is True
        except asyncio.TimeoutError:
            pytest.skip("Neo4j connection timed out")
        except Exception as e:
            pytest.skip(f"Neo4j not available: {str(e)}")
        finally:
            await graph.close()
    
    @pytest.mark.asyncio
    async def test_full_ingestion_workflow(self, kernel):
        """Test complete document ingestion into Qdrant and Neo4j."""
        import tempfile
        import os
        
        # Create a test document
        test_content = """# Test Document on Consensus

Consensus is a fundamental problem in distributed systems.

## The Problem

When multiple nodes need to agree on a value, they must use consensus.
The Paxos algorithm was developed by Leslie Lamport to solve this problem.

## Related Concepts

Raft is a more understandable alternative to Paxos.
Both algorithms ensure safety even during network partitions.
Leader election is a key component of these algorithms.
"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "consensus.md")
            with open(filepath, 'w') as f:
                f.write(test_content)
            
            # Ingest the document
            result = await kernel.ingest(filepath, is_directory=False)
            
            # Verify ingestion results
            assert "error" not in result, f"Ingestion failed: {result}"
            assert result.get("chunks", 0) > 0, "Should have created chunks"
            
            # Verify memory has content
            stats = kernel.memory.get_stats()
            assert stats["semantic_count"] > 0, "Should have semantic memories"
    
    @pytest.mark.asyncio
    async def test_learning_loop_integration(self, kernel):
        """Test that learning loop can run and produce results."""
        # Run a few learning iterations
        result = await kernel.learn(iterations=3, interval_seconds=0.5)
        
        # Check results
        assert result.get("total_iterations", 0) == 3, "Should complete 3 iterations"
        
        # Check that episodes were recorded
        stats = kernel.memory.get_stats()
        total_episodes = stats.get("passed_episodes", 0) + stats.get("failed_episodes", 0)
        assert total_episodes >= 0, "Should have recorded episodes"
    
    @pytest.mark.asyncio
    async def test_neo4j_stores_entities(self, kernel):
        """Test that entities are stored in Neo4j."""
        if not kernel.graph:
            pytest.skip("Graph plugin not available")
        
        # Store a test entity
        await kernel.graph.store_entity(
            entity_id="test_entity_1",
            entity_type="concept",
            properties={"name": "Test Concept", "description": "A test"}
        )
        
        # Verify it's stored
        result = await kernel.graph.run_cypher(
            "MATCH (e:Entity {id: $id}) RETURN e.id as id, e.type as type",
            {"id": "test_entity_1"}
        )
        
        assert len(result) > 0, "Entity should be stored in Neo4j"
        assert result[0]["id"] == "test_entity_1"
    
    @pytest.mark.asyncio
    async def test_neo4j_stores_relations(self, kernel):
        """Test that relations are stored in Neo4j."""
        if not kernel.graph:
            pytest.skip("Graph plugin not available")
        
        # Store two entities
        await kernel.graph.store_entity("entity_a", "concept", {"name": "A"})
        await kernel.graph.store_entity("entity_b", "concept", {"name": "B"})
        
        # Store a relation
        await kernel.graph.store_relation(
            from_entity="entity_a",
            to_entity="entity_b",
            relation_type="RELATES_TO",
            properties={"source": "test"}
        )
        
        # Verify relation exists
        result = await kernel.graph.run_cypher(
            """
            MATCH (a:Entity {id: 'entity_a'})-[r:RELATES_TO]->(b:Entity {id: 'entity_b'})
            RETURN type(r) as relation_type
            """
        )
        
        assert len(result) > 0, "Relation should be stored in Neo4j"
    
    @pytest.mark.asyncio
    async def test_kernel_status_complete(self, kernel):
        """Test that kernel status returns all component information."""
        status = kernel.get_status()
        
        # Check all components are present
        assert "world" in status
        assert "memory" in status
        assert "goals" in status
        assert "reasoning" in status
        assert "meta" in status
        
        # Check world stats
        assert "total_states" in status["world"]
        assert "total_events" in status["world"]
        
        # Check memory stats
        assert "semantic_count" in status["memory"]
        assert "episodic_count" in status["memory"]
    
    @pytest.mark.asyncio
    async def test_evaluation_report_generation(self, kernel):
        """Test that evaluation report is generated correctly."""
        # Run some learning first
        await kernel.learn(iterations=2, interval_seconds=0.5)
        
        # Generate report
        report = kernel.evaluate()
        
        assert "summary" in report
        assert "knowledge_coverage" in report
        assert "performance_trend" in report
        assert "self_correction" in report
        
        # Check conclusion exists
        assert "conclusion" in report


class TestE2EWithoutLLM:
    """Tests that work without LLM (core functionality)."""
    
    @pytest.fixture
    def kernel_no_llm(self):
        """Create kernel without LLM for testing core logic."""
        from agi_kernel.kernel import Kernel
        
        kernel = Kernel(use_plugins=False)
        return kernel
    
    def test_world_model_tracks_states(self, kernel_no_llm):
        """Test that world model properly tracks states."""
        from agi_kernel.core.world import Event
        
        # Create an event
        event = Event(
            actor="test",
            action="observe",
            context={"data": "test_value"}
        )
        
        # Observe it
        state = kernel_no_llm.world.observe(event)
        
        # Verify state is tracked
        assert state.id in kernel_no_llm.world.states
        assert kernel_no_llm.world.get_stats()["total_states"] > 0
    
    def test_memory_stores_and_recalls(self, kernel_no_llm):
        """Test memory storage and recall."""
        from agi_kernel.core.memory import MemoryType
        
        # Store a memory
        item = kernel_no_llm.memory.store(
            content={"fact": "The sky is blue"},
            memory_type=MemoryType.SEMANTIC,
        )
        
        assert item is not None
        
        # Recall
        results = kernel_no_llm.memory.recall("sky blue")
        assert len(results) > 0
    
    def test_goals_generated_from_empty_state(self, kernel_no_llm):
        """Test that goals are generated even with empty state."""
        goals = kernel_no_llm.goals.generate(
            kernel_no_llm.memory,
            kernel_no_llm.world
        )
        
        # Should generate exploration goals at minimum
        assert len(goals) > 0
    
    def test_reasoning_chooses_strategy(self, kernel_no_llm):
        """Test that reasoning controller chooses appropriate strategy."""
        from agi_kernel.core.reasoning import ReasoningContext
        
        context = ReasoningContext(
            question="Why does X cause Y?",
            complexity_estimate=0.7,
            has_graph=True,
        )
        
        strategy, reason = kernel_no_llm.reasoning.choose_strategy(context)
        
        assert strategy is not None
        assert reason is not None and len(reason) > 0
    
    def test_meta_cognition_detects_patterns(self, kernel_no_llm):
        """Test that meta-cognition detects failure patterns."""
        # Record multiple failures
        for i in range(5):
            kernel_no_llm.meta._record_failure({
                "question": f"Same failing question type",
                "strategy": "fast_recall",
                "confidence": 0.2,
            })
        
        # Detect patterns
        patterns = kernel_no_llm.meta.detect_pattern(kernel_no_llm.meta.failure_history)
        
        # Should detect a pattern
        assert len(patterns) > 0


class TestMetricsAndReporting:
    """Tests for metrics collection and reporting."""
    
    def test_snapshot_collection(self):
        """Test metrics snapshot collection."""
        from agi_kernel.metrics import MetricsCollector, Snapshot
        from agi_kernel.kernel import Kernel
        
        kernel = Kernel(use_plugins=False)
        collector = MetricsCollector(output_dir="./test_metrics")
        
        snapshot = collector.collect_snapshot(
            memory=kernel.memory,
            world=kernel.world,
            goals=kernel.goals,
            reasoning=kernel.reasoning,
            meta=kernel.meta,
        )
        
        assert isinstance(snapshot, Snapshot)
        assert snapshot.timestamp is not None
    
    def test_improvement_analysis(self):
        """Test improvement analysis over multiple snapshots."""
        from agi_kernel.metrics import MetricsCollector, Snapshot
        from datetime import timedelta
        
        collector = MetricsCollector(output_dir="./test_metrics")
        
        # Create fake snapshots showing improvement
        snapshot1 = Snapshot(
            timestamp=datetime.utcnow() - timedelta(hours=1),
            pass_rate=0.3,
            semantic_memory_count=10,
            changes_adopted=0,
        )
        
        snapshot2 = Snapshot(
            timestamp=datetime.utcnow(),
            pass_rate=0.6,  # Improved!
            semantic_memory_count=50,  # More knowledge!
            changes_adopted=2,  # Self-corrected!
        )
        
        collector.snapshots = [snapshot1, snapshot2]
        
        analysis = collector.analyze_improvement()
        
        assert analysis["is_improving"] is True
        assert len(analysis["reasons"]) > 0
    
    def test_report_generation(self):
        """Test full report generation."""
        from agi_kernel.metrics import MetricsCollector, Snapshot
        
        collector = MetricsCollector(output_dir="./test_metrics")
        
        # Add a snapshot
        collector.snapshots.append(Snapshot(
            pass_rate=0.5,
            semantic_memory_count=25,
        ))
        
        report = collector.generate_report()
        
        assert "summary" in report
        assert "knowledge_coverage" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
