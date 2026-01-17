"""
📊 Metrics Module
=================

Phase 3 of the POC: Evaluation without cheating.

What we measure:
1. Knowledge Coverage (entities, relations, multi-hop)
2. Failure Rate Over Time (does FAIL rate decrease?)
3. Reasoning Strategy Shift (more Graph, less Vector?)
4. Self-Correction Events (ontology/strategy/memory changes)

Key Principle:
If these numbers don't change → it's NOT AGI-like.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import structlog

if TYPE_CHECKING:
    from agi_kernel.core.memory import Memory
    from agi_kernel.core.world import WorldModel
    from agi_kernel.core.goals import GoalEngine
    from agi_kernel.core.reasoning import ReasoningController
    from agi_kernel.core.meta import MetaCognition
    from agi_kernel.learning_loop import LearningLoop

logger = structlog.get_logger()


@dataclass
class Snapshot:
    """A snapshot of system metrics at a point in time."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Knowledge Coverage
    entity_count: int = 0
    relation_count: int = 0
    multi_hop_relations: int = 0
    semantic_memory_count: int = 0
    episodic_memory_count: int = 0
    
    # Performance
    pass_rate: float = 0.0
    average_confidence: float = 0.0
    
    # Strategy Distribution
    strategy_usage: dict = field(default_factory=dict)
    strategy_success_rates: dict = field(default_factory=dict)
    
    # Self-Correction
    patterns_detected: int = 0
    changes_proposed: int = 0
    changes_adopted: int = 0
    
    # Goals
    goals_completed: int = 0
    goals_failed: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "knowledge_coverage": {
                "entities": self.entity_count,
                "relations": self.relation_count,
                "multi_hop": self.multi_hop_relations,
                "semantic_memory": self.semantic_memory_count,
                "episodic_memory": self.episodic_memory_count,
            },
            "performance": {
                "pass_rate": self.pass_rate,
                "avg_confidence": self.average_confidence,
            },
            "strategies": {
                "usage": self.strategy_usage,
                "success_rates": self.strategy_success_rates,
            },
            "self_correction": {
                "patterns": self.patterns_detected,
                "proposed": self.changes_proposed,
                "adopted": self.changes_adopted,
            },
            "goals": {
                "completed": self.goals_completed,
                "failed": self.goals_failed,
            },
        }


class MetricsCollector:
    """
    Collects and analyzes metrics for POC evaluation.
    
    This answers the central question:
    "Does the system show measurable cumulative cognitive improvement?"
    """
    
    def __init__(
        self,
        output_dir: str = "./metrics",
    ):
        """
        Initialize the metrics collector.
        
        Args:
            output_dir: Directory to save metric logs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.snapshots: list[Snapshot] = []
        
        logger.info("metrics_collector_initialized", output_dir=str(self.output_dir))
    
    def collect_snapshot(
        self,
        memory: Optional[Memory] = None,
        world: Optional[WorldModel] = None,
        goals: Optional[GoalEngine] = None,
        reasoning: Optional[ReasoningController] = None,
        meta: Optional[MetaCognition] = None,
        learning_loop: Optional[LearningLoop] = None,
        graph_plugin=None,
    ) -> Snapshot:
        """
        Collect a snapshot of current metrics.
        
        Args:
            memory: Memory system
            world: World model
            goals: Goal engine
            reasoning: Reasoning controller
            meta: Meta-cognition
            learning_loop: Learning loop
            graph_plugin: Graph plugin for entity/relation counts
            
        Returns:
            The collected snapshot
        """
        snapshot = Snapshot()
        
        # Memory stats
        if memory:
            stats = memory.get_stats()
            snapshot.semantic_memory_count = stats.get("semantic_count", 0)
            snapshot.episodic_memory_count = stats.get("episodic_count", 0)
            
            # Calculate pass rate from episodes
            passed = stats.get("passed_episodes", 0)
            failed = stats.get("failed_episodes", 0)
            total = passed + failed
            snapshot.pass_rate = passed / total if total > 0 else 0
        
        # World stats
        if world:
            stats = world.get_stats()
            snapshot.multi_hop_relations = stats.get("total_transitions", 0)
        
        # Goal stats
        if goals:
            stats = goals.get_stats()
            snapshot.goals_completed = len(goals.completed_goals)
            snapshot.goals_failed = len(goals.failed_goals)
        
        # Reasoning stats
        if reasoning:
            stats = reasoning.get_stats()
            snapshot.average_confidence = 0
            
            strategy_stats = stats.get("strategy_stats", {})
            for strategy, s_stats in strategy_stats.items():
                snapshot.strategy_usage[strategy] = s_stats.get("attempts", 0)
                
                attempts = s_stats.get("attempts", 0)
                successes = s_stats.get("successes", 0)
                if attempts > 0:
                    snapshot.strategy_success_rates[strategy] = successes / attempts
        
        # Meta-cognition stats
        if meta:
            stats = meta.get_stats()
            snapshot.patterns_detected = stats.get("total_patterns", 0)
            snapshot.changes_proposed = stats.get("total_proposed", 0) + stats.get("total_adopted", 0) + stats.get("total_rejected", 0)
            snapshot.changes_adopted = stats.get("total_adopted", 0)
        
        # Learning loop metrics
        if learning_loop:
            metrics = learning_loop.get_metrics()
            if metrics.get("iterations", 0) > 0:
                snapshot.pass_rate = metrics.get("pass_rate", 0)
                snapshot.average_confidence = metrics.get("average_confidence", 0)
                
                strategies = metrics.get("strategies_used", {})
                for strategy, s_stats in strategies.items():
                    snapshot.strategy_usage[strategy] = s_stats.get("count", 0)
                    if s_stats.get("count", 0) > 0:
                        snapshot.strategy_success_rates[strategy] = (
                            s_stats.get("passed", 0) / s_stats["count"]
                        )
        
        # Store snapshot
        self.snapshots.append(snapshot)
        
        logger.info(
            "snapshot_collected",
            pass_rate=snapshot.pass_rate,
            entities=snapshot.entity_count,
            changes_adopted=snapshot.changes_adopted,
        )
        
        return snapshot
    
    def analyze_improvement(self) -> dict:
        """
        Analyze whether the system is improving.
        
        This is the key POC evaluation:
        - Is failure rate decreasing?
        - Is knowledge coverage increasing?
        - Are strategy shifts occurring?
        - Are self-corrections happening?
        
        Returns:
            Analysis report
        """
        if len(self.snapshots) < 2:
            return {"status": "insufficient_data", "snapshots": len(self.snapshots)}
        
        first = self.snapshots[0]
        last = self.snapshots[-1]
        
        # Calculate deltas
        pass_rate_delta = last.pass_rate - first.pass_rate
        entity_delta = last.entity_count - first.entity_count
        relation_delta = last.relation_count - first.relation_count
        multi_hop_delta = last.multi_hop_relations - first.multi_hop_relations
        memory_delta = (
            (last.semantic_memory_count + last.episodic_memory_count) -
            (first.semantic_memory_count + first.episodic_memory_count)
        )
        
        # Analyze strategy shifts
        strategy_shift = False
        first_dominant = max(first.strategy_usage.items(), key=lambda x: x[1])[0] if first.strategy_usage else None
        last_dominant = max(last.strategy_usage.items(), key=lambda x: x[1])[0] if last.strategy_usage else None
        if first_dominant and last_dominant and first_dominant != last_dominant:
            strategy_shift = True
        
        # Determine overall assessment
        is_improving = False
        reasons = []
        
        if pass_rate_delta > 0.05:
            is_improving = True
            reasons.append(f"Pass rate improved by {pass_rate_delta:.1%}")
        
        if memory_delta > 5:
            is_improving = True
            reasons.append(f"Knowledge grew by {memory_delta} memories")
        
        if last.changes_adopted > 0:
            is_improving = True
            reasons.append(f"Self-corrected {last.changes_adopted} times")
        
        if strategy_shift:
            is_improving = True
            reasons.append(f"Strategy shifted from {first_dominant} to {last_dominant}")
        
        # Check for concerning patterns
        concerns = []
        if pass_rate_delta < -0.1:
            concerns.append("Pass rate is declining")
        if last.pass_rate == 0 and len(self.snapshots) > 3:
            concerns.append("No successful answers")
        
        report = {
            "is_improving": is_improving,
            "reasons": reasons,
            "concerns": concerns,
            "deltas": {
                "pass_rate": pass_rate_delta,
                "entities": entity_delta,
                "relations": relation_delta,
                "multi_hop": multi_hop_delta,
                "memory": memory_delta,
            },
            "strategy_shift": strategy_shift,
            "self_corrections": last.changes_adopted,
            "snapshots_analyzed": len(self.snapshots),
            "time_span": {
                "from": first.timestamp.isoformat(),
                "to": last.timestamp.isoformat(),
            },
        }
        
        logger.info(
            "improvement_analysis_complete",
            is_improving=is_improving,
            reasons=len(reasons),
            concerns=len(concerns),
        )
        
        return report
    
    def generate_report(self) -> dict:
        """
        Generate a comprehensive evaluation report.
        
        This is the final POC deliverable.
        """
        if not self.snapshots:
            return {"error": "No data collected"}
        
        improvement = self.analyze_improvement()
        
        # Summary statistics
        pass_rates = [s.pass_rate for s in self.snapshots]
        confidences = [s.average_confidence for s in self.snapshots]
        
        report = {
            "summary": {
                "total_snapshots": len(self.snapshots),
                "is_system_improving": improvement.get("is_improving", False),
                "improvement_reasons": improvement.get("reasons", []),
                "concerns": improvement.get("concerns", []),
            },
            "knowledge_coverage": {
                "final_semantic_memory": self.snapshots[-1].semantic_memory_count,
                "final_episodic_memory": self.snapshots[-1].episodic_memory_count,
                "final_multi_hop_relations": self.snapshots[-1].multi_hop_relations,
            },
            "performance_trend": {
                "initial_pass_rate": pass_rates[0] if pass_rates else 0,
                "final_pass_rate": pass_rates[-1] if pass_rates else 0,
                "pass_rate_improvement": improvement.get("deltas", {}).get("pass_rate", 0),
            },
            "strategy_evolution": {
                "initial_dominant": (
                    max(self.snapshots[0].strategy_usage.items(), key=lambda x: x[1])[0]
                    if self.snapshots[0].strategy_usage else None
                ),
                "final_dominant": (
                    max(self.snapshots[-1].strategy_usage.items(), key=lambda x: x[1])[0]
                    if self.snapshots[-1].strategy_usage else None
                ),
                "shift_detected": improvement.get("strategy_shift", False),
            },
            "self_correction": {
                "patterns_detected": self.snapshots[-1].patterns_detected,
                "changes_proposed": self.snapshots[-1].changes_proposed,
                "changes_adopted": self.snapshots[-1].changes_adopted,
                "adoption_rate": (
                    self.snapshots[-1].changes_adopted / self.snapshots[-1].changes_proposed
                    if self.snapshots[-1].changes_proposed > 0 else 0
                ),
            },
            "goals": {
                "completed": self.snapshots[-1].goals_completed,
                "failed": self.snapshots[-1].goals_failed,
                "success_rate": (
                    self.snapshots[-1].goals_completed / 
                    (self.snapshots[-1].goals_completed + self.snapshots[-1].goals_failed)
                    if (self.snapshots[-1].goals_completed + self.snapshots[-1].goals_failed) > 0 else 0
                ),
            },
            "conclusion": self._generate_conclusion(improvement),
        }
        
        return report
    
    def _generate_conclusion(self, improvement: dict) -> str:
        """Generate a human-readable conclusion."""
        if improvement.get("is_improving"):
            reasons = " | ".join(improvement.get("reasons", []))
            return f"✅ POSITIVE: System shows cumulative cognitive improvement. {reasons}"
        elif improvement.get("concerns"):
            concerns = " | ".join(improvement.get("concerns", []))
            return f"⚠️ CONCERNS: {concerns}"
        else:
            return "📊 NEUTRAL: More data needed to determine improvement. Continue learning loop."
    
    def save_logs(self, filename: str = None) -> str:
        """
        Save metrics logs to file.
        
        Args:
            filename: Optional filename (default: auto-generated)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.output_dir / filename
        
        data = {
            "snapshots": [s.to_dict() for s in self.snapshots],
            "report": self.generate_report(),
            "generated_at": datetime.utcnow().isoformat(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info("metrics_saved", filepath=str(filepath))
        return str(filepath)
    
    def export_for_visualization(self) -> dict:
        """
        Export data in format suitable for visualization.
        
        Returns dict with arrays for plotting.
        """
        return {
            "timestamps": [s.timestamp.isoformat() for s in self.snapshots],
            "pass_rates": [s.pass_rate for s in self.snapshots],
            "confidences": [s.average_confidence for s in self.snapshots],
            "entity_counts": [s.entity_count for s in self.snapshots],
            "relation_counts": [s.relation_count for s in self.snapshots],
            "memory_counts": [s.semantic_memory_count + s.episodic_memory_count for s in self.snapshots],
            "patterns_detected": [s.patterns_detected for s in self.snapshots],
            "changes_adopted": [s.changes_adopted for s in self.snapshots],
        }
