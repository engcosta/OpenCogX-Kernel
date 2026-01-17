"""
🌍 World Model
==============

Represents the world as states → actions → outcomes with uncertainty.

Responsibilities:
- Represent reality as state transitions
- Track causal relations with probability
- Enable simulation, prediction, and planning

Key Properties:
- Causal relations
- Probabilistic outcomes
- Temporal validity
- Action consequences

Constraints:
- ❌ No raw text
- ❌ No embeddings
- ✅ Everything must be explainable
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import structlog

logger = structlog.get_logger()


class RelationType(Enum):
    """Types of causal relationships between states."""
    LEADS_TO = "leads_to"       # State A leads to State B
    CAUSES = "causes"           # Event causes State
    PRECEDES = "precedes"       # Temporal ordering
    ENABLES = "enables"         # A enables B (but doesn't guarantee)
    PREVENTS = "prevents"       # A prevents B
    CORRELATES = "correlates"   # Statistical correlation


@dataclass
class State:
    """
    Represents a world state at a point in time.
    
    A state is a snapshot of the world with features that describe it.
    States are connected through causal relations with uncertainty.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    features: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 1.0  # How certain we are about this state
    source: str = "observation"  # observation, prediction, inference
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def to_dict(self) -> dict:
        """Serialize state for storage."""
        return {
            "id": self.id,
            "features": self.features,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "source": self.source,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> State:
        """Deserialize state from storage."""
        return cls(
            id=data["id"],
            features=data["features"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            confidence=data.get("confidence", 1.0),
            source=data.get("source", "observation"),
        )


@dataclass
class Event:
    """
    Represents an event that causes state transitions.
    
    Events are actions or occurrences that change the world state.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    actor: str = "system"
    action: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def to_dict(self) -> dict:
        """Serialize event for storage."""
        return {
            "id": self.id,
            "actor": self.actor,
            "action": self.action,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> Event:
        """Deserialize event from storage."""
        return cls(
            id=data["id"],
            actor=data["actor"],
            action=data["action"],
            context=data.get("context", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


@dataclass
class StateTransition:
    """
    Represents a causal link between states with probability.
    
    State --LEADS_TO(probability)--> State
    """
    from_state_id: str
    to_state_id: str
    relation: RelationType
    probability: float = 1.0  # P(to_state | from_state)
    evidence_count: int = 1   # How many times observed
    
    def to_dict(self) -> dict:
        """Serialize transition for storage."""
        return {
            "from_state_id": self.from_state_id,
            "to_state_id": self.to_state_id,
            "relation": self.relation.value,
            "probability": self.probability,
            "evidence_count": self.evidence_count,
        }


class WorldModel:
    """
    The World Model maintains a representation of reality.
    
    Core API (Mandatory):
    - observe(event) -> State
    - predict(state, action) -> list[State]
    - confidence(state) -> float
    
    Philosophy:
    - The world is represented as state transitions with uncertainty
    - No raw text, no embeddings - everything is structured and explainable
    - Enables simulation, prediction, and planning
    """
    
    def __init__(self, graph_plugin=None):
        """
        Initialize the World Model.
        
        Args:
            graph_plugin: Optional Neo4j plugin for persistent storage
        """
        self.states: dict[str, State] = {}
        self.events: dict[str, Event] = {}
        self.transitions: list[StateTransition] = []
        self.graph_plugin = graph_plugin
        
        logger.info("world_model_initialized")
    
    async def observe(self, event: Event) -> State:
        """
        Process an event and return the resulting state.
        
        This is the primary way the world model learns about changes.
        
        Args:
            event: The event that occurred
            
        Returns:
            The new state after the event
        """
        # Store the event
        self.events[event.id] = event
        
        # Create a new state from the event
        new_state = State(
            features={
                "action": event.action,
                "actor": event.actor,
                **event.context,
            },
            source="observation",
        )
        
        self.states[new_state.id] = new_state
        
        logger.info(
            "world_state_observed",
            event_id=event.id,
            state_id=new_state.id,
            action=event.action,
        )
        
        # Persist to graph if available
        # Persist to graph if available
        if self.graph_plugin:
            await self.graph_plugin.store_state(new_state)
            await self.graph_plugin.store_event(event)
        
        return new_state
    
    def predict(
        self, 
        state: State, 
        action: str,
        context: Optional[dict] = None,
    ) -> list[tuple[State, float]]:
        """
        Predict possible future states given current state and action.
        
        Uses learned transition probabilities to predict outcomes.
        
        Args:
            state: Current state
            action: Proposed action
            context: Additional context for prediction
            
        Returns:
            List of (predicted_state, probability) tuples
        """
        predictions: list[tuple[State, float]] = []
        
        # Find historical transitions from similar states
        for transition in self.transitions:
            if transition.from_state_id == state.id:
                to_state = self.states.get(transition.to_state_id)
                if to_state:
                    predictions.append((to_state, transition.probability))
        
        # If no historical data, create an uncertain prediction
        if not predictions:
            predicted_state = State(
                features={
                    "action": action,
                    "based_on": state.id,
                    **(context or {}),
                },
                confidence=0.3,  # Low confidence for novel predictions
                source="prediction",
            )
            predictions.append((predicted_state, 0.3))
            
            logger.debug(
                "world_prediction_novel",
                from_state=state.id,
                action=action,
            )
        
        return predictions
    
    async def record_transition(
        self,
        from_state: State,
        to_state: State,
        relation: RelationType = RelationType.LEADS_TO,
    ) -> StateTransition:
        """
        Record a causal transition between states.
        
        Updates probability based on evidence accumulation.
        
        Args:
            from_state: Starting state
            to_state: Resulting state
            relation: Type of causal relation
            
        Returns:
            The recorded transition
        """
        # Check if this transition already exists
        for transition in self.transitions:
            if (transition.from_state_id == from_state.id and 
                transition.to_state_id == to_state.id):
                # Update evidence count
                transition.evidence_count += 1
                # Recalculate probability (more evidence = higher confidence)
                transition.probability = min(
                    0.95, 
                    0.5 + (transition.evidence_count * 0.1)
                )
                
                logger.debug(
                    "world_transition_updated",
                    from_state=from_state.id,
                    to_state=to_state.id,
                    evidence=transition.evidence_count,
                )
                
                return transition
        
        # Create new transition
        transition = StateTransition(
            from_state_id=from_state.id,
            to_state_id=to_state.id,
            relation=relation,
            probability=0.5,  # Start with medium confidence
        )
        self.transitions.append(transition)
        
        # Store states if not already tracked
        if from_state.id not in self.states:
            self.states[from_state.id] = from_state
        if to_state.id not in self.states:
            self.states[to_state.id] = to_state
        
        # Persist to graph if available
        # Persist to graph if available
        if self.graph_plugin:
            await self.graph_plugin.store_transition(transition)
        
        logger.info(
            "world_transition_recorded",
            from_state=from_state.id,
            to_state=to_state.id,
            relation=relation.value,
        )
        
        return transition
    
    def get_confidence(self, state: State) -> float:
        """
        Get the confidence level for a state.
        
        Confidence depends on:
        - Source (observation > inference > prediction)
        - Age (newer = more confident)
        - Evidence (more observations = more confident)
        
        Args:
            state: The state to evaluate
            
        Returns:
            Confidence score between 0 and 1
        """
        base_confidence = state.confidence
        
        # Adjust for source
        source_weights = {
            "observation": 1.0,
            "inference": 0.8,
            "prediction": 0.5,
        }
        source_weight = source_weights.get(state.source, 0.5)
        
        # Adjust for age (decay over time)
        age_hours = (datetime.utcnow() - state.timestamp).total_seconds() / 3600
        age_decay = max(0.5, 1.0 - (age_hours * 0.01))  # 1% decay per hour
        
        return base_confidence * source_weight * age_decay
    
    def find_causal_chain(
        self,
        start_state: State,
        end_state: State,
        max_depth: int = 5,
    ) -> list[StateTransition] | None:
        """
        Find a causal chain between two states.
        
        Uses BFS to find the shortest path.
        
        Args:
            start_state: Starting state
            end_state: Target state  
            max_depth: Maximum chain length
            
        Returns:
            List of transitions forming the chain, or None if not found
        """
        from collections import deque
        
        visited = set()
        queue: deque[tuple[str, list[StateTransition]]] = deque()
        queue.append((start_state.id, []))
        
        while queue:
            current_id, path = queue.popleft()
            
            if current_id == end_state.id:
                return path
            
            if current_id in visited or len(path) >= max_depth:
                continue
            
            visited.add(current_id)
            
            for transition in self.transitions:
                if transition.from_state_id == current_id:
                    queue.append((
                        transition.to_state_id,
                        path + [transition]
                    ))
        
        return None
    
    def get_stats(self) -> dict:
        """Get statistics about the world model."""
        return {
            "total_states": len(self.states),
            "total_events": len(self.events),
            "total_transitions": len(self.transitions),
            "observation_states": sum(
                1 for s in self.states.values() if s.source == "observation"
            ),
            "prediction_states": sum(
                1 for s in self.states.values() if s.source == "prediction"
            ),
            "inference_states": sum(
                1 for s in self.states.values() if s.source == "inference"
            ),
        }
