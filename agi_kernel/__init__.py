"""
🧠 Open AGI Kernel
==================

An open-source, self-evolving AGI kernel focused on system-level intelligence.

Core Components:
- WorldModel: Represents reality as states → actions → outcomes with uncertainty
- Memory: Semantic, episodic, and temporal memory with decay
- GoalEngine: Intrinsic motivation without user input
- ReasoningController: Dynamic strategy selection
- MetaCognition: Self-monitoring and structural adaptation
"""

__version__ = "0.1.0"

from agi_kernel.core.world import WorldModel, State, Event
from agi_kernel.core.memory import Memory, MemoryType
from agi_kernel.core.goals import GoalEngine, Goal, GoalType
from agi_kernel.core.reasoning import ReasoningController, ReasoningStrategy
from agi_kernel.core.meta import MetaCognition

__all__ = [
    "WorldModel",
    "State", 
    "Event",
    "Memory",
    "MemoryType",
    "GoalEngine",
    "Goal",
    "GoalType",
    "ReasoningController",
    "ReasoningStrategy",
    "MetaCognition",
]
