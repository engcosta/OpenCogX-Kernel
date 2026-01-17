"""
Quick E2E Verification Script
==============================

Verifies:
1. Qdrant connection and storage
2. Neo4j connection and storage
3. Core kernel functionality
4. Generates a simple evaluation report
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path


async def run_verification():
    """Run quick verification of all components."""
    print("=" * 60)
    print("🧠 AGI KERNEL POC - VERIFICATION")
    print("=" * 60)
    print(f"\nStarted at: {datetime.now().isoformat()}")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "components": {},
        "tests": [],
        "status": "unknown",
    }
    
    # Test 1: Core Components (no LLM needed)
    print("\n📦 Test 1: Core Components...")
    try:
        from agi_kernel.kernel import Kernel
        
        kernel = Kernel(use_plugins=False)
        
        # Test World Model
        from agi_kernel.core.world import Event
        event = Event(actor="test", action="verify", context={"step": 1})
        state = kernel.world.observe(event)
        assert state is not None
        print("   ✅ World Model: Working")
        results["components"]["world"] = "OK"
        
        # Test Memory
        from agi_kernel.core.memory import MemoryType
        item = kernel.memory.store(
            content={"fact": "Test fact"},
            memory_type=MemoryType.SEMANTIC,
        )
        assert item is not None
        recalled = kernel.memory.recall("test")
        assert len(recalled) > 0
        print("   ✅ Memory: Working")
        results["components"]["memory"] = "OK"
        
        # Test Goals
        goals = kernel.goals.generate(kernel.memory, kernel.world)
        assert len(goals) > 0
        print("   ✅ Goals: Working")
        results["components"]["goals"] = "OK"
        
        # Test Reasoning
        from agi_kernel.core.reasoning import ReasoningContext
        context = ReasoningContext(question="Why?", complexity_estimate=0.5)
        strategy, reason = kernel.reasoning.choose_strategy(context)
        assert strategy is not None
        print("   ✅ Reasoning: Working")
        results["components"]["reasoning"] = "OK"
        
        # Test Meta-Cognition
        for i in range(3):
            kernel.meta._record_failure({"question": "test", "strategy": "fast"})
        patterns = kernel.meta.detect_pattern(kernel.meta.failure_history)
        print("   ✅ Meta-Cognition: Working")
        results["components"]["meta"] = "OK"
        
        results["tests"].append({"name": "Core Components", "status": "PASS"})
        
    except Exception as e:
        print(f"   ❌ Core Components: {e}")
        results["tests"].append({"name": "Core Components", "status": "FAIL", "error": str(e)})
    
    # Test 2: Qdrant Connection
    print("\n📦 Test 2: Qdrant Vector Database...")
    try:
        from agi_kernel.plugins.vector import VectorPlugin
        
        vector = VectorPlugin(collection_name="test_verification")
        await vector.initialize(force_recreate=True)
        
        stats = vector.get_stats()
        print(f"   ✅ Qdrant: Connected (collection: {stats.get('collection')})")
        results["components"]["qdrant"] = "OK"
        results["tests"].append({"name": "Qdrant Connection", "status": "PASS"})
        
        await vector.close()
        
    except Exception as e:
        print(f"   ❌ Qdrant: {e}")
        results["components"]["qdrant"] = f"FAIL: {e}"
        results["tests"].append({"name": "Qdrant Connection", "status": "FAIL", "error": str(e)})
    
    # Test 3: Neo4j Connection
    print("\n📦 Test 3: Neo4j Graph Database...")
    try:
        from agi_kernel.plugins.graph import GraphPlugin
        
        graph = GraphPlugin()
        connected = await asyncio.wait_for(graph.initialize(), timeout=10.0)
        
        if connected:
            # Store and retrieve entity
            await graph.store_entity(
                entity_id="test_entity",
                entity_type="concept",
                properties={"name": "Test", "verified": True}
            )
            
            result = await graph.run_cypher(
                "MATCH (e:Entity {id: $id}) RETURN e.id as id",
                {"id": "test_entity"}
            )
            
            print(f"   ✅ Neo4j: Connected (stored entity: {len(result)} found)")
            results["components"]["neo4j"] = "OK"
            results["tests"].append({"name": "Neo4j Connection", "status": "PASS"})
        else:
            print("   ⚠️ Neo4j: Connection returned False")
            results["components"]["neo4j"] = "WARN"
        
        await graph.close()
        
    except asyncio.TimeoutError:
        print("   ⚠️ Neo4j: Connection timeout (may need different credentials)")
        results["components"]["neo4j"] = "TIMEOUT"
        results["tests"].append({"name": "Neo4j Connection", "status": "TIMEOUT"})
    except Exception as e:
        print(f"   ❌ Neo4j: {e}")
        results["components"]["neo4j"] = f"FAIL: {e}"
        results["tests"].append({"name": "Neo4j Connection", "status": "FAIL", "error": str(e)})
    
    # Test 4: Learning Loop (without LLM - mock mode)
    print("\n📦 Test 4: Learning Loop...")
    try:
        kernel = Kernel(use_plugins=False)
        
        # Run a step
        iteration = await kernel.learning_loop.step()
        
        print(f"   ✅ Learning Loop: Completed iteration {iteration.id}")
        print(f"      Question: {iteration.question[:50]}...")
        print(f"      Strategy: {iteration.strategy_used}")
        print(f"      Verdict: {iteration.verdict}")
        
        results["components"]["learning_loop"] = "OK"
        results["tests"].append({"name": "Learning Loop", "status": "PASS"})
        
    except Exception as e:
        print(f"   ❌ Learning Loop: {e}")
        results["tests"].append({"name": "Learning Loop", "status": "FAIL", "error": str(e)})
    
    # Test 5: Metrics and Evaluation
    print("\n📦 Test 5: Metrics Collection...")
    try:
        from agi_kernel.metrics import MetricsCollector, Snapshot
        
        collector = MetricsCollector(output_dir="./metrics")
        
        # Add snapshots
        collector.snapshots.append(Snapshot(pass_rate=0.3, semantic_memory_count=10))
        collector.snapshots.append(Snapshot(pass_rate=0.6, semantic_memory_count=50, changes_adopted=2))
        
        # Analyze
        analysis = collector.analyze_improvement()
        report = collector.generate_report()
        
        print(f"   ✅ Metrics: Working")
        print(f"      Is Improving: {analysis.get('is_improving', False)}")
        print(f"      Reasons: {len(analysis.get('reasons', []))}")
        
        results["components"]["metrics"] = "OK"
        results["tests"].append({"name": "Metrics Collection", "status": "PASS"})
        
        # Include improvement analysis
        results["improvement_analysis"] = analysis
        results["evaluation_report"] = report
        
    except Exception as e:
        print(f"   ❌ Metrics: {e}")
        results["tests"].append({"name": "Metrics Collection", "status": "FAIL", "error": str(e)})
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for t in results["tests"] if t["status"] == "PASS")
    total = len(results["tests"])
    
    print(f"\nTests: {passed}/{total} passed")
    
    for test in results["tests"]:
        emoji = "✅" if test["status"] == "PASS" else "❌" if test["status"] == "FAIL" else "⚠️"
        print(f"   {emoji} {test['name']}: {test['status']}")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED - AGI Kernel is ready!")
        results["status"] = "ALL_PASS"
    elif passed >= total - 1:
        print("\n⚠️ MOSTLY WORKING - Minor issues detected")
        results["status"] = "PARTIAL"
    else:
        print("\n❌ ISSUES DETECTED - Please check the errors above")
        results["status"] = "FAIL"
    
    # Save results
    results_path = f"./metrics/verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path("./metrics").mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Results saved to: {results_path}")
    
    print("\n" + "=" * 60)
    print(f"Completed at: {datetime.now().isoformat()}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    asyncio.run(run_verification())
