"""
Full Kernel Demo Script with Evaluation Report
===============================================

This script runs the complete AGI Kernel workflow:
1. Initialize kernel with all plugins
2. Ingest the sample corpus
3. Run the learning loop
4. Collect metrics over time
5. Generate comprehensive evaluation report
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path


async def run_full_demo():
    """Run the complete demonstration."""
    from agi_kernel.kernel import Kernel
    
    print("=" * 60)
    print("🧠 AGI KERNEL POC - FULL DEMONSTRATION")
    print("=" * 60)
    print(f"\nStarted at: {datetime.now().isoformat()}")
    
    # Initialize kernel
    print("\n📦 Phase 0: Initializing Kernel...")
    kernel = Kernel(use_plugins=True)
    plugin_status = await kernel.initialize_plugins()
    
    print(f"   Qdrant: {'✅' if plugin_status.get('vector') else '❌'}")
    print(f"   Neo4j: {'✅' if plugin_status.get('graph') else '⚠️ (optional)'}")
    print(f"   LLM: {'✅' if plugin_status.get('llm') else '⚠️ (demo mode)'}")
    
    try:
        # Phase 1: Ingestion
        print("\n📥 Phase 1: Ingesting Corpus...")
        corpus_path = Path("./corpus")
        
        if corpus_path.exists():
            result = await kernel.ingest(str(corpus_path), is_directory=True)
            print(f"   Files processed: {result.get('files_processed', 0)}")
            print(f"   Chunks created: {result.get('total_chunks', 0)}")
            print(f"   Entities extracted: {result.get('total_entities', 0)}")
            print(f"   Relations found: {result.get('total_relations', 0)}")
        else:
            print("   ⚠️ Corpus directory not found, using demo mode")
        
        # Collect initial snapshot
        kernel.metrics.collect_snapshot(
            memory=kernel.memory,
            world=kernel.world,
            goals=kernel.goals,
            reasoning=kernel.reasoning,
            meta=kernel.meta,
        )
        
        # Phase 2: Learning Loop
        print("\n🔁 Phase 2: Running Learning Loop...")
        iterations = 5
        
        for i in range(iterations):
            print(f"\n   Iteration {i + 1}/{iterations}...")
            
            iteration = await kernel.learning_loop.step()
            
            # Show progress
            verdict = iteration.verdict
            emoji = "✅" if verdict == "PASS" else "❌" if verdict == "FAIL" else "⚠️"
            print(f"   {emoji} Question: {iteration.question[:50]}...")
            print(f"      Strategy: {iteration.strategy_used}, Confidence: {iteration.confidence:.2f}")
            print(f"      Verdict: {verdict}")
            
            # Short pause between iterations
            await asyncio.sleep(0.5)
        
        # Collect final snapshot
        kernel.metrics.collect_snapshot(
            memory=kernel.memory,
            world=kernel.world,
            goals=kernel.goals,
            reasoning=kernel.reasoning,
            meta=kernel.meta,
            learning_loop=kernel.learning_loop,
        )
        
        # Phase 3: Evaluation
        print("\n📊 Phase 3: Generating Evaluation Report...")
        report = kernel.evaluate()
        
        # Display results
        print("\n" + "=" * 60)
        print("📋 EVALUATION REPORT")
        print("=" * 60)
        
        # Summary
        summary = report.get("summary", {})
        is_improving = summary.get("is_system_improving", False)
        print(f"\n🎯 System Status: {'IMPROVING' if is_improving else 'STABLE/NEEDS MORE DATA'}")
        
        if summary.get("improvement_reasons"):
            print("\n✅ Improvement Signals:")
            for reason in summary["improvement_reasons"]:
                print(f"   • {reason}")
        
        if summary.get("concerns"):
            print("\n⚠️ Concerns:")
            for concern in summary["concerns"]:
                print(f"   • {concern}")
        
        # Knowledge Coverage
        coverage = report.get("knowledge_coverage", {})
        print(f"\n📚 Knowledge Coverage:")
        print(f"   Semantic Memories: {coverage.get('final_semantic_memory', 0)}")
        print(f"   Episodic Memories: {coverage.get('final_episodic_memory', 0)}")
        print(f"   Multi-hop Relations: {coverage.get('final_multi_hop_relations', 0)}")
        
        # Performance
        perf = report.get("performance_trend", {})
        print(f"\n📈 Performance:")
        print(f"   Initial Pass Rate: {perf.get('initial_pass_rate', 0):.1%}")
        print(f"   Final Pass Rate: {perf.get('final_pass_rate', 0):.1%}")
        print(f"   Improvement: {perf.get('pass_rate_improvement', 0):.1%}")
        
        # Strategy Evolution
        strat = report.get("strategy_evolution", {})
        print(f"\n🧠 Strategy Evolution:")
        print(f"   Initial Dominant: {strat.get('initial_dominant', 'N/A')}")
        print(f"   Final Dominant: {strat.get('final_dominant', 'N/A')}")
        print(f"   Shift Detected: {strat.get('shift_detected', False)}")
        
        # Self-Correction
        correction = report.get("self_correction", {})
        print(f"\n🔧 Self-Correction:")
        print(f"   Patterns Detected: {correction.get('patterns_detected', 0)}")
        print(f"   Changes Proposed: {correction.get('changes_proposed', 0)}")
        print(f"   Changes Adopted: {correction.get('changes_adopted', 0)}")
        
        # Conclusion
        print(f"\n📌 Conclusion:")
        print(f"   {report.get('conclusion', 'No conclusion available')}")
        
        # Save report
        report_path = f"./metrics/evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path("./metrics").mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n💾 Full report saved to: {report_path}")
        
        # Show kernel status
        print("\n" + "=" * 60)
        print("🔍 KERNEL STATUS")
        print("=" * 60)
        status = kernel.get_status()
        
        print(f"\n🌎 World Model:")
        print(f"   States: {status['world'].get('total_states', 0)}")
        print(f"   Events: {status['world'].get('total_events', 0)}")
        print(f"   Transitions: {status['world'].get('total_transitions', 0)}")
        
        print(f"\n💾 Memory:")
        print(f"   Semantic: {status['memory'].get('semantic_count', 0)}")
        print(f"   Episodic: {status['memory'].get('episodic_count', 0)}")
        print(f"   Contradictions: {status['memory'].get('contradiction_count', 0)}")
        
        print(f"\n🎯 Goals:")
        print(f"   Active: {len(status['goals'].get('active_goals', []))}")
        print(f"   Completed: {status['goals'].get('completed_count', 0)}")
        print(f"   Failed: {status['goals'].get('failed_count', 0)}")
        
        print(f"\n🧠 Meta-Cognition:")
        meta_stats = kernel.meta.get_stats()
        print(f"   Total Patterns: {meta_stats.get('total_patterns', 0)}")
        print(f"   Changes Adopted: {meta_stats.get('total_adopted', 0)}")
        
        # Self-knowledge
        self_knowledge = kernel.meta.get_self_knowledge()
        if self_knowledge.get("blind_spots"):
            print(f"   Blind Spots: {len(self_knowledge['blind_spots'])}")
        if self_knowledge.get("detected_biases"):
            print(f"   Detected Biases: {len(self_knowledge['detected_biases'])}")
        
        print("\n" + "=" * 60)
        print("✅ DEMONSTRATION COMPLETE")
        print("=" * 60)
        print(f"\nCompleted at: {datetime.now().isoformat()}")
        
    finally:
        await kernel.close()


if __name__ == "__main__":
    asyncio.run(run_full_demo())
