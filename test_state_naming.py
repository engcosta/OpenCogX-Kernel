import asyncio
from datetime import datetime
from agi_kernel.core.world import State, Event
from agi_kernel.plugins.graph import GraphPlugin

async def test_naming():
    print("Testing State/Event Naming in Graph...")
    graph = GraphPlugin()
    await graph.initialize()
    
    # 1. Store Event
    event = Event(
        action="test_action_naming",
        actor="test_user",
        context={"details": "verifying graph labels"}
    )
    print(f"Storing Event: {event.action}")
    await graph.store_event(event)
    
    # Verify
    res = await graph.run_cypher(f"MATCH (n:Event {{id: '{event.id}'}}) RETURN n.name")
    print(f"Event Name in Graph: {res[0]['n.name']}")
    
    # 2. Store State
    state = State(
        features={"action": "test_state_outcome", "value": 123},
        source="test"
    )
    print(f"Storing State: {state.features['action']}")
    await graph.store_state(state)
    
    # Verify
    res = await graph.run_cypher(f"MATCH (n:State {{id: '{state.id}'}}) RETURN n.name")
    print(f"State Name in Graph: {res[0]['n.name']}")
    
    await graph.close()

if __name__ == "__main__":
    asyncio.run(test_naming())
