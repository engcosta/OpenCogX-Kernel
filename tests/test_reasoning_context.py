
import pytest
from agi_kernel.core.reasoning import ReasoningController, ReasoningStrategy, ReasoningContext
from agi_kernel.core.memory import Memory, MemoryType

@pytest.mark.asyncio
async def test_reasoning_returns_context_memories():
    # Setup
    memory = Memory()
    memory.store(
        content={"fact": "The sky is blue"},
        memory_type=MemoryType.SEMANTIC,
        source="test"
    )
    
    controller = ReasoningController() # No plugins
    
    context = ReasoningContext(
        question="sky",
        available_memory_types=["semantic"],
        has_vector=False
    )
    
    # Executing FAST_RECALL
    result = await controller.execute(
        strategy=ReasoningStrategy.FAST_RECALL,
        question="sky",
        context=context.__dict__,
        memory=memory
    )
    
    assert result["success"] is True
    assert "context" in result
    assert "memories" in result["context"]
    assert len(result["context"]["memories"]) > 0
    assert result["context"]["memories"][0]["content"]["fact"] == "The sky is blue"

@pytest.mark.asyncio
async def test_hybrid_reasoning_returns_context_memories():
    # Setup
    memory = Memory()
    memory.store(
        content={"fact": "The grass is green"},
        memory_type=MemoryType.SEMANTIC,
        source="test"
    )
    
    controller = ReasoningController() # No plugins, so hybrid will just fallback to recall results
    
    context = ReasoningContext(
        question="grass",
        available_memory_types=["semantic"],
        has_vector=False
    )
    
    # Executing HYBRID
    result = await controller.execute(
        strategy=ReasoningStrategy.HYBRID,
        question="grass",
        context=context.__dict__,
        memory=memory
    )
    
    assert result["success"] is True
    assert "context" in result
    assert "memories" in result["context"]
    assert len(result["context"]["memories"]) > 0
    assert result["context"]["memories"][0]["content"]["fact"] == "The grass is green"
