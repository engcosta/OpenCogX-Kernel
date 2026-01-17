"""Quick test script for infrastructure."""
import asyncio

async def test_infrastructure():
    """Test Qdrant and Neo4j connections."""
    print("=" * 50)
    print("Testing AGI Kernel Infrastructure")
    print("=" * 50)
    
    # Test Qdrant
    print("\n1. Testing Qdrant...")
    try:
        from agi_kernel.plugins.vector import VectorPlugin
        vector = VectorPlugin()
        result = await vector.initialize()
        print(f"   ✅ Qdrant connected: {result}")
        stats = vector.get_stats()
        print(f"   Collection: {stats.get('collection')}")
        await vector.close()
    except Exception as e:
        print(f"   ❌ Qdrant error: {e}")
    
    # Test Neo4j
    print("\n2. Testing Neo4j...")
    try:
        from agi_kernel.plugins.graph import GraphPlugin
        graph = GraphPlugin()
        result = await asyncio.wait_for(graph.initialize(), timeout=10.0)
        print(f"   ✅ Neo4j connected: {result}")
        stats = graph.get_stats()
        print(f"   URI: {stats.get('uri')}")
        await graph.close()
    except asyncio.TimeoutError:
        print("   ⚠️ Neo4j connection timed out")
    except Exception as e:
        print(f"   ❌ Neo4j error: {e}")
    
    print("\n" + "=" * 50)
    print("Infrastructure test complete!")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(test_infrastructure())
