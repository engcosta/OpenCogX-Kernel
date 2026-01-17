import asyncio
import os
from agi_kernel.kernel import Kernel

async def inspect_graph():
    print("Initializing Kernel...")
    kernel = Kernel()
    await kernel.initialize_plugins()
    
    if not kernel.graph:
        print("Graph plugin not available.")
        return

    print("Querying usage of 'name' property and node types...")
    
    # Check for nodes that look like they might be the ones in the screenshot
    try:
        # 1. Check State nodes
        print("\n--- State Nodes ---")
        states = await kernel.graph.run_cypher(
            "MATCH (n:State) RETURN n.id, n.timestamp, n.name LIMIT 5"
        )
        for s in states:
            print(f"State: ID={s.get('n.id')} | Timestamp={s.get('n.timestamp')} | Name={s.get('n.name')}")

        # 2. Check Event nodes
        print("\n--- Event Nodes ---")
        events = await kernel.graph.run_cypher(
            "MATCH (n:Event) RETURN n.id, n.timestamp, n.name LIMIT 5"
        )
        for e in events:
            print(f"Event: ID={e.get('n.id')} | Timestamp={e.get('n.timestamp')} | Name={e.get('n.name')}")

        # 3. Check Entity nodes
        print("\n--- Entity Nodes ---")
        entities = await kernel.graph.run_cypher(
            "MATCH (n:Entity) RETURN n.id, n.type, n.name LIMIT 5"
        )
        for e in entities:
            print(f"Entity: ID={e.get('n.id')} | Type={e.get('n.type')} | Name={e.get('n.name')}")

    except Exception as e:
        print(f"Error querying graph: {e}")

    await kernel.close()

if __name__ == "__main__":
    asyncio.run(inspect_graph())
