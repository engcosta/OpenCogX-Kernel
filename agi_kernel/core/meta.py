"""
🔮 Meta-Cognition
=================

The most critical component - enables self-evolution.

Responsibilities:
- Analyze failures
- Detect error patterns
- Propose structural changes

What is ALLOWED:
- Modify ontology
- Change default strategy
- Adjust memory weights

What is FORBIDDEN:
- ❌ Self-modifying code
- ❌ Changing learning laws
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional
import structlog

if TYPE_CHECKING:
    from agi_kernel.core.memory import Memory
    from agi_kernel.core.reasoning import ReasoningController, ReasoningStrategy
    from agi_kernel.core.goals import GoalEngine
    from agi_kernel.core.world import WorldModel

logger = structlog.get_logger()


class ChangeType(Enum):
    """Types of structural changes that can be proposed."""
    ONTOLOGY_UPDATE = "ontology_update"
    STRATEGY_WEIGHT = "strategy_weight"
    MEMORY_WEIGHT = "memory_weight"
    GOAL_PRIORITY = "goal_priority"
    DECAY_RATE = "decay_rate"
    CONFIDENCE_THRESHOLD = "confidence_threshold"


@dataclass
class FailurePattern:
    """
    A detected pattern of repeated failures.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pattern_type: str = ""  # e.g., "repeated_topic", "strategy_mismatch"
    description: str = ""
    occurrences: int = 0
    affected_topics: list[str] = field(default_factory=list)
    affected_strategies: list[str] = field(default_factory=list)
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict:
        """Serialize pattern for storage."""
        return {
            "id": self.id,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "occurrences": self.occurrences,
            "affected_topics": self.affected_topics,
            "affected_strategies": self.affected_strategies,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
        }


@dataclass
class ProposedChange:
    """
    A structural change proposed by meta-cognition.
    
    Changes must be tested before adoption.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    change_type: ChangeType = ChangeType.STRATEGY_WEIGHT
    reason: str = ""  # WHY this change is proposed
    pattern_id: Optional[str] = None  # Which pattern triggered this
    
    # Change details
    target: str = ""  # What to change
    current_value: Any = None
    proposed_value: Any = None
    
    # Status
    status: str = "proposed"  # proposed, testing, adopted, rejected
    created_at: datetime = field(default_factory=datetime.utcnow)
    tested_at: Optional[datetime] = None
    decided_at: Optional[datetime] = None
    
    # Test results
    test_improvement: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Serialize change proposal for storage."""
        return {
            "id": self.id,
            "change_type": self.change_type.value,
            "reason": self.reason,
            "pattern_id": self.pattern_id,
            "target": self.target,
            "current_value": self.current_value,
            "proposed_value": self.proposed_value,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "tested_at": self.tested_at.isoformat() if self.tested_at else None,
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "test_improvement": self.test_improvement,
        }


@dataclass
class SelfEvaluation:
    """
    Result of self-evaluation process.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Performance metrics
    success_rate: float = 0.0
    average_confidence: float = 0.0
    contradiction_rate: float = 0.0
    
    # Identified issues
    failure_patterns: list[str] = field(default_factory=list)
    blind_spots: list[str] = field(default_factory=list)
    biases: list[str] = field(default_factory=list)
    
    # Proposed actions
    proposed_changes: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Serialize evaluation for storage."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "success_rate": self.success_rate,
            "average_confidence": self.average_confidence,
            "contradiction_rate": self.contradiction_rate,
            "failure_patterns": self.failure_patterns,
            "blind_spots": self.blind_spots,
            "biases": self.biases,
            "proposed_changes": self.proposed_changes,
        }


class MetaCognition:
    """
    Self-monitoring and structural adaptation.
    
    Core API (Mandatory):
    - evaluate(outcome): Analyze an outcome
    - detect_pattern(history): Find error patterns
    - propose_change(): Suggest structural modifications
    
    Allowed Changes:
    - Ontology modifications
    - Strategy default changes
    - Memory type weighting
    
    FORBIDDEN:
    - ❌ Self-modifying code
    - ❌ Changing learning laws
    
    Philosophy:
    - Repeated failure = wrong thinking approach, not hard question
    - The system must be able to change how it thinks
    """
    
    def __init__(
        self,
        min_pattern_occurrences: int = 3,
        improvement_threshold: float = 0.1,
    ):
        """
        Initialize Meta-Cognition.
        
        Args:
            min_pattern_occurrences: Minimum failures to detect a pattern
            improvement_threshold: Minimum improvement to adopt a change
        """
        self.patterns: dict[str, FailurePattern] = {}
        self.proposed_changes: dict[str, ProposedChange] = {}
        self.adopted_changes: list[ProposedChange] = []
        self.rejected_changes: list[ProposedChange] = []
        self.evaluations: list[SelfEvaluation] = []
        
        self.min_pattern_occurrences = min_pattern_occurrences
        self.improvement_threshold = improvement_threshold
        
        # Failure history for pattern detection
        self.failure_history: list[dict] = []
        
        # Blind spot tracking
        self.blind_spots: set[str] = set()
        
        # Bias tracking
        self.detected_biases: list[dict] = []
        
        logger.info(
            "meta_cognition_initialized",
            min_pattern=min_pattern_occurrences,
            threshold=improvement_threshold,
        )
    
    def evaluate(
        self,
        outcome: dict,
        memory: Optional[Memory] = None,
        reasoning: Optional[ReasoningController] = None,
    ) -> SelfEvaluation:
        """
        Evaluate an outcome and update self-knowledge.
        
        Args:
            outcome: The outcome to evaluate (success, answer, confidence, etc.)
            memory: Memory system for historical analysis
            reasoning: Reasoning controller for strategy analysis
            
        Returns:
            Self-evaluation result
        """
        now = datetime.utcnow()
        
        # Track failure
        if not outcome.get("success", False):
            self._record_failure(outcome)
        
        # Calculate metrics
        success_rate = self._calculate_success_rate(memory)
        avg_confidence = self._calculate_avg_confidence(reasoning)
        contradiction_rate = self._calculate_contradiction_rate(memory)
        
        # Detect patterns
        patterns = self.detect_pattern(self.failure_history)
        
        # Identify blind spots
        blind_spots = self._identify_blind_spots()
        
        # Detect biases
        biases = self._detect_biases(reasoning)
        
        # Generate proposed changes
        proposed = []
        for pattern in patterns:
            changes = self.propose_change(pattern)
            proposed.extend([c.id for c in changes])
        
        evaluation = SelfEvaluation(
            success_rate=success_rate,
            average_confidence=avg_confidence,
            contradiction_rate=contradiction_rate,
            failure_patterns=[p.id for p in patterns],
            blind_spots=list(blind_spots),
            biases=[b["type"] for b in biases],
            proposed_changes=proposed,
        )
        
        self.evaluations.append(evaluation)
        
        logger.info(
            "self_evaluation_complete",
            success_rate=success_rate,
            patterns_found=len(patterns),
            changes_proposed=len(proposed),
        )
        
        return evaluation
    
    def _record_failure(self, outcome: dict) -> None:
        """Record a failure for pattern detection."""
        self.failure_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "question": outcome.get("question", ""),
            "strategy": outcome.get("strategy", ""),
            "confidence": outcome.get("confidence", 0),
            "error": outcome.get("error", ""),
            "context": outcome.get("context", {}),
        })
        
        # Keep last 100 failures
        if len(self.failure_history) > 100:
            self.failure_history = self.failure_history[-100:]
    
    def _calculate_success_rate(self, memory: Optional[Memory]) -> float:
        """Calculate recent success rate."""
        if not memory:
            return 0.0
        
        stats = memory.get_stats()
        passed = stats.get("passed_episodes", 0)
        failed = stats.get("failed_episodes", 0)
        total = passed + failed
        
        return passed / total if total > 0 else 0.0
    
    def _calculate_avg_confidence(
        self,
        reasoning: Optional[ReasoningController],
    ) -> float:
        """Calculate average confidence across strategies."""
        if not reasoning:
            return 0.0
        
        stats = reasoning.get_stats()
        strategy_stats = stats.get("strategy_stats", {})
        
        confidences = [
            s["avg_confidence"] 
            for s in strategy_stats.values() 
            if s["attempts"] > 0
        ]
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def _calculate_contradiction_rate(self, memory: Optional[Memory]) -> float:
        """Calculate rate of contradictions."""
        if not memory:
            return 0.0
        
        stats = memory.get_stats()
        contradictions = stats.get("contradiction_count", 0)
        total = stats.get("semantic_count", 1)
        
        return contradictions / total if total > 0 else 0.0
    
    def _identify_blind_spots(self) -> set[str]:
        """Identify topics we consistently fail on."""
        topic_failures: dict[str, int] = {}
        
        for failure in self.failure_history:
            # Extract topic from question (simplified)
            question = failure.get("question", "")[:30]
            topic_failures[question] = topic_failures.get(question, 0) + 1
        
        # Topics with repeated failures are blind spots
        blind_spots = {
            topic for topic, count in topic_failures.items()
            if count >= self.min_pattern_occurrences
        }
        
        self.blind_spots.update(blind_spots)
        return blind_spots
    
    def _detect_biases(
        self,
        reasoning: Optional[ReasoningController],
    ) -> list[dict]:
        """Detect reasoning biases."""
        biases = []
        
        if not reasoning:
            return biases
        
        stats = reasoning.get_stats()
        strategy_stats = stats.get("strategy_stats", {})
        
        # Detect over-reliance on one strategy
        total_attempts = sum(s["attempts"] for s in strategy_stats.values())
        if total_attempts > 10:
            for strategy, s_stats in strategy_stats.items():
                usage_rate = s_stats["attempts"] / total_attempts
                if usage_rate > 0.5:  # Over 50% usage
                    biases.append({
                        "type": "strategy_over_reliance",
                        "strategy": strategy,
                        "usage_rate": usage_rate,
                    })
                elif usage_rate < 0.05 and s_stats["successes"] > 0:
                    biases.append({
                        "type": "strategy_under_utilization",
                        "strategy": strategy,
                        "usage_rate": usage_rate,
                    })
        
        self.detected_biases.extend(biases)
        return biases
    
    def detect_pattern(
        self,
        history: list[dict],
    ) -> list[FailurePattern]:
        """
        Detect patterns in failure history.
        
        Args:
            history: List of failure records
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        # Pattern 1: Same topic failing repeatedly
        topic_counts: dict[str, list[dict]] = {}
        for failure in history:
            topic = failure.get("question", "")[:30]
            if topic not in topic_counts:
                topic_counts[topic] = []
            topic_counts[topic].append(failure)
        
        for topic, failures in topic_counts.items():
            if len(failures) >= self.min_pattern_occurrences:
                pattern_id = f"topic_{hash(topic) % 10000}"
                
                if pattern_id not in self.patterns:
                    pattern = FailurePattern(
                        id=pattern_id,
                        pattern_type="repeated_topic_failure",
                        description=f"Repeated failures on topic: {topic}",
                        occurrences=len(failures),
                        affected_topics=[topic],
                    )
                    self.patterns[pattern_id] = pattern
                    patterns.append(pattern)
                else:
                    # Update existing pattern
                    self.patterns[pattern_id].occurrences = len(failures)
                    self.patterns[pattern_id].last_seen = datetime.utcnow()
        
        # Pattern 2: Strategy consistently failing
        strategy_failures: dict[str, int] = {}
        for failure in history:
            strategy = failure.get("strategy", "unknown")
            strategy_failures[strategy] = strategy_failures.get(strategy, 0) + 1
        
        for strategy, count in strategy_failures.items():
            if count >= self.min_pattern_occurrences:
                pattern_id = f"strategy_{strategy}"
                
                if pattern_id not in self.patterns:
                    pattern = FailurePattern(
                        id=pattern_id,
                        pattern_type="strategy_mismatch",
                        description=f"Strategy {strategy} frequently failing",
                        occurrences=count,
                        affected_strategies=[strategy],
                    )
                    self.patterns[pattern_id] = pattern
                    patterns.append(pattern)
        
        logger.debug(
            "patterns_detected",
            count=len(patterns),
            types=[p.pattern_type for p in patterns],
        )
        
        return patterns
    
    def propose_change(
        self,
        pattern: Optional[FailurePattern] = None,
    ) -> list[ProposedChange]:
        """
        Propose structural changes based on patterns.
        
        Args:
            pattern: The failure pattern triggering the change
            
        Returns:
            List of proposed changes
        """
        changes = []
        
        if not pattern:
            return changes
        
        # Based on pattern type, propose appropriate changes
        if pattern.pattern_type == "repeated_topic_failure":
            # Propose increasing memory weight for this topic
            change = ProposedChange(
                change_type=ChangeType.MEMORY_WEIGHT,
                reason=f"Repeated failures suggest insufficient knowledge on: {pattern.affected_topics}",
                pattern_id=pattern.id,
                target="memory_strength_multiplier",
                current_value=1.0,
                proposed_value=1.5,
            )
            changes.append(change)
            self.proposed_changes[change.id] = change
        
        elif pattern.pattern_type == "strategy_mismatch":
            # Propose reducing weight of failing strategy
            for strategy in pattern.affected_strategies:
                change = ProposedChange(
                    change_type=ChangeType.STRATEGY_WEIGHT,
                    reason=f"Strategy {strategy} has high failure rate",
                    pattern_id=pattern.id,
                    target=strategy,
                    current_value=1.0,
                    proposed_value=0.7,
                )
                changes.append(change)
                self.proposed_changes[change.id] = change
        
        logger.info(
            "changes_proposed",
            count=len(changes),
            types=[c.change_type.value for c in changes],
        )
        
        return changes
    
    def test_change(
        self,
        change_id: str,
        test_results: dict,
    ) -> bool:
        """
        Test a proposed change and decide whether to adopt.
        
        Args:
            change_id: The change to test
            test_results: Results from testing the change
            
        Returns:
            True if change should be adopted
        """
        if change_id not in self.proposed_changes:
            logger.warning("change_not_found", change_id=change_id)
            return False
        
        change = self.proposed_changes[change_id]
        change.tested_at = datetime.utcnow()
        change.status = "testing"
        
        # Calculate improvement
        improvement = test_results.get("improvement", 0.0)
        change.test_improvement = improvement
        
        # Decide adoption
        should_adopt = improvement >= self.improvement_threshold
        
        change.decided_at = datetime.utcnow()
        
        if should_adopt:
            change.status = "adopted"
            self.adopted_changes.append(change)
            del self.proposed_changes[change_id]
            
            logger.info(
                "change_adopted",
                change_id=change_id,
                improvement=improvement,
            )
        else:
            change.status = "rejected"
            self.rejected_changes.append(change)
            del self.proposed_changes[change_id]
            
            logger.info(
                "change_rejected",
                change_id=change_id,
                improvement=improvement,
            )
        
        return should_adopt
    
    def apply_change(
        self,
        change: ProposedChange,
        reasoning: Optional[ReasoningController] = None,
        memory: Optional[Memory] = None,
    ) -> bool:
        """
        Apply an adopted change to the system.
        
        Args:
            change: The change to apply
            reasoning: Reasoning controller (if strategy change)
            memory: Memory system (if memory change)
            
        Returns:
            True if successfully applied
        """
        if change.status != "adopted":
            logger.warning("change_not_adopted", change_id=change.id)
            return False
        
        try:
            if change.change_type == ChangeType.STRATEGY_WEIGHT and reasoning:
                from agi_kernel.core.reasoning import ReasoningStrategy
                strategy = ReasoningStrategy(change.target)
                adjustment = change.proposed_value - change.current_value
                reasoning.adjust_strategy_weight(strategy, adjustment)
                
            elif change.change_type == ChangeType.DECAY_RATE and memory:
                memory.decay_rate = change.proposed_value
                
            elif change.change_type == ChangeType.MEMORY_WEIGHT:
                # Apply memory weight change (store for later use)
                pass
            
            logger.info(
                "change_applied",
                change_id=change.id,
                type=change.change_type.value,
            )
            return True
            
        except Exception as e:
            logger.error("change_apply_failed", change_id=change.id, error=str(e))
            return False
    
    def get_pending_changes(self) -> list[ProposedChange]:
        """Get all pending proposed changes."""
        return list(self.proposed_changes.values())
    
    def get_self_knowledge(self) -> dict:
        """
        Get current self-knowledge.
        
        Returns understanding of:
        - Strengths
        - Weaknesses  
        - Blind spots
        - Biases
        """
        return {
            "blind_spots": list(self.blind_spots),
            "detected_biases": self.detected_biases[-10:],
            "active_patterns": [p.to_dict() for p in self.patterns.values()],
            "pending_changes": len(self.proposed_changes),
            "adopted_changes": len(self.adopted_changes),
            "rejected_changes": len(self.rejected_changes),
            "recent_evaluations": [e.to_dict() for e in self.evaluations[-5:]],
        }
    
    def get_stats(self) -> dict:
        """Get statistics about meta-cognition."""
        return {
            "total_patterns": len(self.patterns),
            "total_proposed": len(self.proposed_changes),
            "total_adopted": len(self.adopted_changes),
            "total_rejected": len(self.rejected_changes),
            "total_evaluations": len(self.evaluations),
            "blind_spot_count": len(self.blind_spots),
            "bias_count": len(self.detected_biases),
            "adoption_rate": (
                len(self.adopted_changes) / (len(self.adopted_changes) + len(self.rejected_changes))
                if (self.adopted_changes or self.rejected_changes) else 0
            ),
        }
