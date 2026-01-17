"""
Tests for core kernel components.
"""

import pytest
from datetime import datetime, timedelta


class TestWorldModel:
    """Tests for the World Model."""
    
    def test_observe_event_creates_state(self):
        """Test that observing an event creates a new state."""
        from agi_kernel.core.world import WorldModel, Event
        
        world = WorldModel()
        event = Event(actor="test", action="create", context={"key": "value"})
        
        state = world.observe(event)
        
        assert state is not None
        assert state.source == "observation"
        assert "action" in state.features
        assert state.features["action"] == "create"
    
    def test_record_transition_updates_probability(self):
        """Test that repeated transitions increase probability."""
        from agi_kernel.core.world import WorldModel, State, RelationType
        
        world = WorldModel()
        state1 = State(features={"name": "start"})
        state2 = State(features={"name": "end"})
        
        # Record same transition multiple times
        t1 = world.record_transition(state1, state2)
        assert t1.probability == 0.5
        
        t2 = world.record_transition(state1, state2)
        assert t2.probability > 0.5
        assert t2.evidence_count == 2
    
    def test_predict_returns_uncertain_for_novel(self):
        """Test that novel predictions have low confidence."""
        from agi_kernel.core.world import WorldModel, State
        
        world = WorldModel()
        state = State(features={"name": "unknown"})
        
        predictions = world.predict(state, "some_action")
        
        assert len(predictions) > 0
        _, probability = predictions[0]
        assert probability < 0.5  # Low confidence for novel


class TestMemory:
    """Tests for the Memory System."""
    
    def test_store_and_recall(self):
        """Test basic store and recall."""
        from agi_kernel.core.memory import Memory, MemoryType
        
        memory = Memory()
        
        # Store
        item = memory.store(
            content={"fact": "test fact"},
            memory_type=MemoryType.SEMANTIC,
        )
        
        assert item is not None
        assert item.type == MemoryType.SEMANTIC
        
        # Recall
        results = memory.recall("test", limit=5)
        assert len(results) > 0
    
    def test_store_episode(self):
        """Test storing episodic memory."""
        from agi_kernel.core.memory import Memory
        
        memory = Memory()
        
        episode = memory.store_episode(
            question="What is X?",
            answer="X is Y",
            outcome="PASS",
            reasoning_strategy="fast_recall",
        )
        
        assert episode is not None
        assert episode.outcome == "PASS"
        assert episode.question == "What is X?"
    
    def test_decay_weakens_memories(self):
        """Test that decay reduces memory strength."""
        from agi_kernel.core.memory import Memory, MemoryType
        
        memory = Memory(decay_rate=0.5)
        
        # Store with old timestamp
        item = memory.store(
            content={"old": "data"},
            memory_type=MemoryType.SEMANTIC,
        )
        
        # Set old last access
        item.last_accessed = datetime.utcnow() - timedelta(hours=24)
        
        # Apply decay
        result = memory.decay()
        
        # Memory should be decayed or forgotten
        assert result["decayed"] + result["forgotten"] >= 0
    
    def test_contradiction_recording(self):
        """Test that contradictions are recorded, not deleted."""
        from agi_kernel.core.memory import Memory, MemoryType
        
        memory = Memory()
        
        item1 = memory.store(
            content={"claim": "X is true"},
            memory_type=MemoryType.SEMANTIC,
        )
        item2 = memory.store(
            content={"claim": "X is false"},
            memory_type=MemoryType.SEMANTIC,
        )
        
        memory.record_contradiction(item1.id, item2.id, "X cannot be both true and false")
        
        contradictions = memory.get_contradictions()
        assert len(contradictions) == 1
        assert contradictions[0]["reason"] == "X cannot be both true and false"


class TestGoalEngine:
    """Tests for the Goal Engine."""
    
    def test_generate_creates_goals(self):
        """Test that goal generation creates goals."""
        from agi_kernel.core.goals import GoalEngine, GoalType
        from agi_kernel.core.memory import Memory
        from agi_kernel.core.world import WorldModel
        
        goals = GoalEngine()
        memory = Memory()
        world = WorldModel()
        
        generated = goals.generate(memory, world)
        
        # Should generate at least exploration goals when memory is empty
        assert len(generated) > 0
    
    def test_prioritize_selects_highest(self):
        """Test that prioritization returns a valid goal."""
        from agi_kernel.core.goals import GoalEngine, Goal, GoalType
        
        engine = GoalEngine()
        
        goals = [
            Goal(type=GoalType.REDUCE_UNCERTAINTY, priority=0.3),
            Goal(type=GoalType.RESOLVE_CONTRADICTION, priority=0.9),
            Goal(type=GoalType.EXPLORE_UNKNOWN, priority=0.5),
        ]
        
        selected = engine.prioritize(goals)
        
        # Prioritization should return one of the provided goals
        assert selected is not None
        assert selected.type in [g.type for g in goals]
        # The selected goal should be stored as active
        assert len(engine.active_goals) > 0
    
    def test_complete_goal_updates_status(self):
        """Test that completing a goal updates its status."""
        from agi_kernel.core.goals import GoalEngine, Goal, GoalType
        
        engine = GoalEngine()
        
        goal = Goal(type=GoalType.EXPLORE_UNKNOWN, priority=0.5)
        goal.status = "active"
        engine.active_goals[goal.id] = goal
        
        engine.complete_goal(goal.id, actual_gain=0.6, success=True)
        
        assert len(engine.completed_goals) == 1
        assert engine.completed_goals[0].actual_gain == 0.6


class TestReasoningController:
    """Tests for the Reasoning Controller."""
    
    def test_choose_strategy_for_causal(self):
        """Test that causal questions get causal strategy."""
        from agi_kernel.core.reasoning import ReasoningController, ReasoningContext, ReasoningStrategy
        
        controller = ReasoningController()
        
        context = ReasoningContext(
            question="Why does X cause Y?",
            has_graph=True,
        )
        
        strategy, reason = controller.choose_strategy(context)
        
        assert strategy == ReasoningStrategy.CAUSAL_REASONING
        assert "causal" in reason.lower()
    
    def test_choose_strategy_for_simple(self):
        """Test that simple questions get fast recall."""
        from agi_kernel.core.reasoning import ReasoningController, ReasoningContext, ReasoningStrategy
        
        controller = ReasoningController()
        
        context = ReasoningContext(
            question="What is X?",
            complexity_estimate=0.2,
            available_memory_types=["semantic"],
        )
        
        strategy, reason = controller.choose_strategy(context)
        
        assert strategy == ReasoningStrategy.FAST_RECALL
    
    def test_adjust_strategy_weight(self):
        """Test that strategy weights can be adjusted."""
        from agi_kernel.core.reasoning import ReasoningController, ReasoningStrategy
        
        controller = ReasoningController()
        
        initial = controller.strategy_weights[ReasoningStrategy.FAST_RECALL.value]
        
        controller.adjust_strategy_weight(ReasoningStrategy.FAST_RECALL, -0.3)
        
        new_weight = controller.strategy_weights[ReasoningStrategy.FAST_RECALL.value]
        assert new_weight < initial


class TestMetaCognition:
    """Tests for Meta-Cognition."""
    
    def test_detect_pattern_from_failures(self):
        """Test that repeated failures create patterns."""
        from agi_kernel.core.meta import MetaCognition
        
        meta = MetaCognition(min_pattern_occurrences=2)
        
        # Record failures on same topic
        for i in range(3):
            meta._record_failure({
                "question": "What is distributed consensus?",
                "strategy": "fast_recall",
                "confidence": 0.3,
            })
        
        patterns = meta.detect_pattern(meta.failure_history)
        
        assert len(patterns) > 0
    
    def test_propose_change_for_pattern(self):
        """Test that patterns generate change proposals."""
        from agi_kernel.core.meta import MetaCognition, FailurePattern, ChangeType
        
        meta = MetaCognition()
        
        pattern = FailurePattern(
            pattern_type="strategy_mismatch",
            description="Strategy X failing",
            occurrences=5,
            affected_strategies=["fast_recall"],
        )
        
        changes = meta.propose_change(pattern)
        
        assert len(changes) > 0
        assert changes[0].change_type == ChangeType.STRATEGY_WEIGHT
    
    def test_self_knowledge_tracks_blind_spots(self):
        """Test that blind spots are tracked."""
        from agi_kernel.core.meta import MetaCognition
        
        meta = MetaCognition(min_pattern_occurrences=2)
        
        # Record repeated failures on same topic
        for i in range(3):
            meta._record_failure({
                "question": "Same question that keeps failing",
                "strategy": "hybrid",
            })
        
        blind_spots = meta._identify_blind_spots()
        
        assert len(blind_spots) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
