"""
Integration tests for the AGI Kernel.
"""

import pytest
import asyncio


@pytest.fixture
def kernel():
    """Create a kernel instance without plugins."""
    from agi_kernel.kernel import Kernel
    
    kernel = Kernel(use_plugins=False)
    yield kernel


class TestKernelIntegration:
    """Integration tests for the full kernel."""
    
    def test_kernel_initialization(self, kernel):
        """Test that kernel initializes all components."""
        assert kernel.world is not None
        assert kernel.memory is not None
        assert kernel.goals is not None
        assert kernel.reasoning is not None
        assert kernel.meta is not None
    
    def test_get_status(self, kernel):
        """Test that status returns all component stats."""
        status = kernel.get_status()
        
        assert "world" in status
        assert "memory" in status
        assert "goals" in status
        assert "reasoning" in status
        assert "meta" in status
    
    def test_components_communicate(self, kernel):
        """Test that components can communicate."""
        # Generate goals based on current state
        goals = kernel.goals.generate(kernel.memory, kernel.world)
        
        # Some goals should be generated
        assert len(goals) > 0
        
        # Goals can be prioritized
        selected = kernel.goals.prioritize(goals)
        assert selected is not None


class TestLearningLoopIntegration:
    """Integration tests for the learning loop."""
    
    def test_learning_loop_exists(self, kernel):
        """Test that learning loop is initialized."""
        assert kernel.learning_loop is not None
    
    @pytest.mark.asyncio
    async def test_learning_loop_can_step(self, kernel):
        """Test that learning loop can execute a step."""
        # This will fail without LLM, but should not error
        iteration = await kernel.learning_loop.step()
        
        assert iteration is not None
        assert iteration.id == 1
    
    def test_metrics_collection(self, kernel):
        """Test that metrics can be collected."""
        snapshot = kernel.metrics.collect_snapshot(
            memory=kernel.memory,
            world=kernel.world,
            goals=kernel.goals,
            reasoning=kernel.reasoning,
            meta=kernel.meta,
        )
        
        assert snapshot is not None
        assert snapshot.timestamp is not None


class TestIngestionIntegration:
    """Integration tests for the ingestion pipeline."""
    
    def test_ingestion_exists(self, kernel):
        """Test that ingestion pipeline is initialized."""
        assert kernel.ingestion is not None
    
    def test_chunking(self, kernel):
        """Test that text can be chunked."""
        text = """# Test Document
        
This is a test paragraph with some content.

## Section 1

More content here.

## Section 2

Even more content.
"""
        chunks = kernel.ingestion._hierarchical_chunk(text, source="test.md")
        
        assert len(chunks) > 0
        # Should have document level chunk
        assert any(c.level == 0 for c in chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
