"""View data stored in Neo4j and Qdrant."""
import asyncio


async def view_stored_data():
    """View what's stored in the databases."""
    print("=" * 60)
    print("🔍 DATABASE CONTENT VIEWER")
    print("=" * 60)
    
    # Check Qdrant
    print("\n📊 Qdrant Vector Database:")
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(host="localhost", port=6333)
        collections = client.get_collections().collections
        
        print(f"   Collections: {len(collections)}")
        for col in collections:
            info = client.get_collection(col.name)
            print(f"   - {col.name}: {info.points_count} points, dim={info.config.params.vectors.size}")
        
        client.close()
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Check Neo4j
    print("\n🔗 Neo4j Graph Database:")
    try:
        from agi_kernel.plugins.graph import GraphPlugin
        
        graph = GraphPlugin()
        await asyncio.wait_for(graph.initialize(), timeout=10.0)
        
        # Count entities
        result = await graph.run_cypher(
            "MATCH (e:Entity) RETURN count(e) as count, collect(e.id)[0..5] as sample_ids"
        )
        if result:
            print(f"   Entities: {result[0].get('count', 0)}")
            print(f"   Sample IDs: {result[0].get('sample_ids', [])}")
        
        # Count states
        result = await graph.run_cypher(
            "MATCH (s:State) RETURN count(s) as count"
        )
        if result:
            print(f"   States: {result[0].get('count', 0)}")
        
        # Count events
        result = await graph.run_cypher(
            "MATCH (e:Event) RETURN count(e) as count"
        )
        if result:
            print(f"   Events: {result[0].get('count', 0)}")
        
        # Count relations
        result = await graph.run_cypher(
            "MATCH ()-[r]->() RETURN count(r) as count"
        )
        if result:
            print(f"   Relations: {result[0].get('count', 0)}")
        
        await graph.close()
    except asyncio.TimeoutError:
        print("   ⚠️ Connection timeout")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(view_stored_data())
