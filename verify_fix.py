
import asyncio
from agi_kernel.core.reasoning import ReasoningController, ReasoningStrategy, ReasoningContext
from agi_kernel.core.memory import Memory, MemoryType

async def verify_fix():
    print("Verifying fix...")
    
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
    
    print("Testing FAST_RECALL...")
    # Executing FAST_RECALL
    result = await controller.execute(
        strategy=ReasoningStrategy.FAST_RECALL,
        question="sky",
        context=context.__dict__,
        memory=memory
    )
    
    if "context" not in result:
        print("FAIL: 'context' key missing in result")
    elif "memories" not in result["context"]:
        print("FAIL: 'memories' key missing in result['context']")
    elif len(result["context"]["memories"]) == 0:
        print("FAIL: 'memories' list is empty")
    else:
        print("PASS: FAST_RECALL returned memories")
        print(f"Memory content: {result['context']['memories'][0]['content']}")

    print("\nTesting HYBRID...")
    # Executing HYBRID
    result = await controller.execute(
        strategy=ReasoningStrategy.HYBRID,
        question="sky",
        context=context.__dict__,
        memory=memory
    )
    
    if "context" not in result:
        print("FAIL: 'context' key missing in result")
    elif "memories" not in result["context"]:
        print("FAIL: 'memories' key missing in result['context']")
    elif len(result["context"]["memories"]) == 0:
        print("FAIL: 'memories' list is empty")
    else:
        print("PASS: HYBRID returned memories")
        print(f"Memory content: {result['context']['memories'][0]['content']}")

if __name__ == "__main__":
    asyncio.run(verify_fix())
