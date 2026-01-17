"""
Clear and reingest script for improved graph structure.
"""
import asyncio


async def main():
    """Clear old data and show what's needed to re-ingest."""
    print("=" * 60)
    print("🔄 CLEAR NEO4J DATA FOR FRESH RE-INGESTION")
    print("=" * 60)
    
    # Clear Neo4j
    print("\n📊 Clearing Neo4j...")
    try:
        from agi_kernel.plugins.graph import GraphPlugin
        
        graph = GraphPlugin()
        await asyncio.wait_for(graph.initialize(), timeout=10.0)
        
        # Delete all nodes and relationships
        result = await graph.run_cypher("MATCH (n) DETACH DELETE n")
        print("   ✅ Neo4j cleared - all nodes and relationships deleted")
        
        await graph.close()
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Clear Qdrant collection
    print("\n📊 Clearing Qdrant...")
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(host="localhost", port=6333)
        collections = client.get_collections().collections
        
        for col in collections:
            client.delete_collection(col.name)
            print(f"   ✅ Deleted collection: {col.name}")
        
        client.close()
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ DATABASES CLEARED!")
    print("=" * 60)
    print("\nNow restart the server and re-ingest:")
    print("  1. Restart: python -m agi_kernel serve --port 8000")
    print("  2. Ingest: POST http://localhost:8000/ingest")
    print('     Body: {"path": "D:/Playground/advanced-agent/corpus", "is_directory": true}')
    print("\nThe new graph will have:")
    print("  • Readable entity names (e.g., 'cap_theorem', 'paxos')")
    print("  • Type-specific labels (:Concept, :Technology, :Pattern)")
    print("  • Proper name property for display")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
