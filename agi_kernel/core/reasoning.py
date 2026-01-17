"""
🧩 Reasoning Controller
=======================

Controls HOW the system thinks, not WHAT it thinks.

Strategies:
- FAST_RECALL: Quick memory lookup
- CAUSAL_REASONING: Follow cause-effect chains
- SIMULATION: Predict outcomes
- VERIFICATION: Cross-check answers
- MULTI_HOP: Chain multiple inferences

Golden Rule:
- Every reasoning decision must log WHY it was made
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
    from agi_kernel.core.world import WorldModel, State

logger = structlog.get_logger()


class ReasoningStrategy(Enum):
    """Available reasoning strategies."""
    FAST_RECALL = "fast_recall"          # Quick memory lookup
    CAUSAL_REASONING = "causal_reasoning" # Follow cause-effect
    SIMULATION = "simulation"             # Predict outcomes
    VERIFICATION = "verification"         # Cross-check answers
    MULTI_HOP = "multi_hop"              # Chain inferences
    ANALOGICAL = "analogical"            # Find similar cases
    ABDUCTIVE = "abductive"              # Best explanation
    HYBRID = "hybrid"                    # Combine strategies
    AUTO = "auto"                        # Automatically select strategy


@dataclass
class ReasoningDecision:
    """
    Record of a reasoning strategy choice.
    
    Every decision must be logged with its reason.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy: ReasoningStrategy = ReasoningStrategy.FAST_RECALL
    reason: str = ""  # WHY this strategy was chosen
    context: dict[str, Any] = field(default_factory=dict)
    
    # Timing
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: Optional[float] = None
    
    # Outcome
    success: bool = False
    confidence: float = 0.0
    output: Optional[Any] = None
    
    def to_dict(self) -> dict:
        """Serialize decision for storage."""
        return {
            "id": self.id,
            "strategy": self.strategy.value,
            "reason": self.reason,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "success": self.success,
            "confidence": self.confidence,
        }


@dataclass
class ReasoningContext:
    """
    Context for reasoning strategy selection.
    """
    question: str = ""
    goal_type: Optional[str] = None
    available_memory_types: list[str] = field(default_factory=list)
    has_graph: bool = False
    has_vector: bool = False
    complexity_estimate: float = 0.5  # 0 = simple, 1 = complex
    time_constraint: Optional[float] = None  # Max seconds allowed
    time_constraint: Optional[float] = None  # Max seconds allowed
    previous_attempts: int = 0
    previous_strategy: Optional[ReasoningStrategy] = None
    strict_mode: bool = False  # If True, deny using outside knowledge


class ReasoningController:
    """
    Controls the reasoning process.
    
    Core API (Mandatory):
    - choose_strategy(context, self_model) -> ReasoningStrategy
    - execute(strategy, context) -> result
    
    Golden Rule:
    - Every reasoning decision must log WHY it was made
    
    Philosophy:
    - Reasoning is not fixed
    - Before each task, decide: how to reason, how deeply, how confidently
    - Minimize hallucination and unnecessary computation
    """
    
    def __init__(
        self,
        llm_plugin=None,
        vector_plugin=None,
        graph_plugin=None,
    ):
        """
        Initialize the Reasoning Controller.
        
        Args:
            llm_plugin: LLM adapter for generation
            vector_plugin: Vector DB for semantic search
            graph_plugin: Graph DB for relation traversal
        """
        self.llm_plugin = llm_plugin
        self.vector_plugin = vector_plugin
        self.graph_plugin = graph_plugin
        
        # Decision history for learning
        self.decisions: list[ReasoningDecision] = []
        
        # Strategy performance tracking
        self.strategy_stats: dict[str, dict] = {
            s.value: {"attempts": 0, "successes": 0, "avg_confidence": 0.0}
            for s in ReasoningStrategy
        }
        
        # Default strategy weights (can be modified by meta-cognition)
        self.strategy_weights: dict[str, float] = {
            ReasoningStrategy.FAST_RECALL.value: 1.0,
            ReasoningStrategy.CAUSAL_REASONING.value: 1.0,
            ReasoningStrategy.SIMULATION.value: 1.0,
            ReasoningStrategy.VERIFICATION.value: 1.0,
            ReasoningStrategy.MULTI_HOP.value: 1.0,
            ReasoningStrategy.ANALOGICAL.value: 1.0,
            ReasoningStrategy.ABDUCTIVE.value: 1.0,
            ReasoningStrategy.HYBRID.value: 1.0,
        }
        
        logger.info("reasoning_controller_initialized")
    
    def choose_strategy(
        self,
        context: ReasoningContext,
        self_model: Optional[dict] = None,
    ) -> tuple[ReasoningStrategy, str]:
        """
        Choose the best reasoning strategy for the context.
        """
        # Analyze the question complexity
        complexity = context.complexity_estimate
        
        # Check available resources
        has_memory = len(context.available_memory_types) > 0
        has_graph = context.has_graph
        has_vector = context.has_vector
        
        # Default strategy selection logic
        if complexity < 0.3 and has_memory:
            strategy = ReasoningStrategy.FAST_RECALL
            reason = "Low complexity question with available memory"
        elif has_graph and self._looks_causal(context.question):
            strategy = ReasoningStrategy.CAUSAL_REASONING
            reason = "Question appears causal and graph is available"
        elif self._looks_predictive(context.question):
            strategy = ReasoningStrategy.SIMULATION
            reason = "Question asks about future outcomes"
        elif complexity > 0.7 or self._looks_multi_hop(context.question):
            strategy = ReasoningStrategy.MULTI_HOP
            reason = "Complex question requiring multiple inference steps"
        elif context.previous_attempts > 0 and context.previous_strategy:
            strategy = self._pick_alternative(context.previous_strategy)
            reason = f"Previous attempt with {context.previous_strategy.value} failed"
        else:
            strategy = ReasoningStrategy.HYBRID
            reason = "Default hybrid approach for balanced reasoning"
        
        # Apply strategy weights
        weight = self.strategy_weights.get(strategy.value, 1.0)
        if weight < 0.5:
            old_strategy = strategy
            strategy = self._pick_highest_weight()
            reason = f"Strategy {old_strategy.value} discouraged, switching to {strategy.value}"
        
        # Rich console log
        from rich.console import Console
        from rich.text import Text
        console = Console()
        
        strategy_color = {
            "fast_recall": "blue",
            "causal_reasoning": "magenta",
            "simulation": "cyan",
            "verification": "green",
            "multi_hop": "yellow",
            "hybrid": "white",
        }.get(strategy.value, "white")
        
        console.print(f"\n[bold]🧠 Reasoning Strategy Selected:[/bold] [{strategy_color}]{strategy.value.upper()}[/{strategy_color}]")
        console.print(f"   [dim]Reason: {reason}[/dim]")
        
        logger.info(
            "reasoning_strategy_chosen",
            strategy=strategy.value,
            reason=reason,
            complexity=complexity,
        )
        
        return strategy, reason
    
    def _looks_causal(self, question: str) -> bool:
        """Check if question is about causality."""
        causal_words = ["why", "cause", "because", "leads to", "results in", "effect"]
        return any(w in question.lower() for w in causal_words)
    
    def _looks_predictive(self, question: str) -> bool:
        """Check if question is about prediction."""
        pred_words = ["will", "would", "predict", "expect", "future", "if"]
        return any(w in question.lower() for w in pred_words)
    
    def _looks_multi_hop(self, question: str) -> bool:
        """Check if question requires multi-hop reasoning."""
        multi_words = ["and", "also", "through", "via", "chain", "path"]
        return any(w in question.lower() for w in multi_words)
    
    def _pick_alternative(self, previous: ReasoningStrategy) -> ReasoningStrategy:
        """Pick a different strategy than before."""
        alternatives = [s for s in ReasoningStrategy if s != previous]
        # Sort by weight
        alternatives.sort(
            key=lambda s: self.strategy_weights.get(s.value, 1.0),
            reverse=True
        )
        return alternatives[0] if alternatives else ReasoningStrategy.HYBRID
    
    def _pick_highest_weight(self) -> ReasoningStrategy:
        """Pick strategy with highest weight."""
        best = max(
            self.strategy_weights.items(),
            key=lambda x: x[1]
        )
        return ReasoningStrategy(best[0])

    async def execute(
        self,
        strategy: ReasoningStrategy,
        question: str,
        context: dict,
        memory: Optional[Memory] = None,
        world: Optional[WorldModel] = None,
    ) -> dict:
        """
        Execute reasoning with the chosen strategy.
        """
        start_time = datetime.utcnow()
        reason = context.get("strategy_reason", "unknown")
        
        # Resolve AUTO strategy
        if strategy == ReasoningStrategy.AUTO:
            # Reconstruct context object for selection
            ctx_obj = ReasoningContext(
                question=question,
                available_memory_types=context.get("available_memory_types", []),
                has_graph=context.get("has_graph", False),
                has_vector=context.get("has_vector", False),
                previous_attempts=context.get("previous_attempts", 0),
                strict_mode=context.get("strict_mode", False),
            )
            strategy, reason = self.choose_strategy(ctx_obj)
            context["strategy_reason"] = reason

        # Create decision record
        decision = ReasoningDecision(
            strategy=strategy,
            reason=reason,
            context=context,
        )
        
        # Rich console log
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        
        console.print(Panel(
            f"[bold {strategy.value == 'fast_recall' and 'blue' or 'magenta'}]{question}[/]",
            title=f"🤔 Executing: {strategy.value}",
            border_style="dim",
            expand=False
        ))
        
        try:
            # Execute based on strategy
            if strategy == ReasoningStrategy.FAST_RECALL:
                result = await self._execute_fast_recall(question, memory, context)
            elif strategy == ReasoningStrategy.CAUSAL_REASONING:
                result = await self._execute_causal(question, context, world)
            elif strategy == ReasoningStrategy.SIMULATION:
                result = await self._execute_simulation(question, context, world)
            elif strategy == ReasoningStrategy.VERIFICATION:
                result = await self._execute_verification(question, context, memory)
            elif strategy == ReasoningStrategy.MULTI_HOP:
                result = await self._execute_multi_hop(question, context, memory, world)
            elif strategy == ReasoningStrategy.ANALOGICAL:
                result = await self._execute_analogical(question, memory)
            elif strategy == ReasoningStrategy.ABDUCTIVE:
                result = await self._execute_abductive(question, context)
            else:  # HYBRID
                result = await self._execute_hybrid(question, context, memory, world)
            
            # Update decision with outcome
            decision.success = result.get("success", False)
            decision.confidence = result.get("confidence", 0.0)
            decision.output = result.get("answer")
            
            # Rich log result
            if decision.success:
                console.print(f"   ✅ [green]Answer Generated[/green] (Conf: {decision.confidence:.2f})")
                console.print(f"      [dim]{str(decision.output)[:100]}...[/dim]")
            else:
                console.print(f"   ❌ [red]Failed to Answer[/red]")
            
        except Exception as e:
            logger.error("reasoning_execution_failed", error=str(e))
            result = {
                "success": False,
                "answer": None,
                "confidence": 0.0,
                "error": str(e),
            }
            decision.success = False
            console.print(f"   ❌ [red]Error:[/red] {str(e)}")
        
        # Calculate duration
        end_time = datetime.utcnow()
        decision.duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Store decision for learning
        self.decisions.append(decision)
        
        # Update strategy stats
        self._update_stats(strategy, decision.success, decision.confidence)
        
        logger.info(
            "reasoning_executed",
            strategy=strategy.value,
            success=decision.success,
            confidence=decision.confidence,
            duration_ms=decision.duration_ms,
        )
        
        if result:
            result["strategy"] = strategy.value
        
        return result
    
    async def _execute_fast_recall(
        self,
        question: str,
        memory: Optional[Memory],
        context: Optional[dict] = None,
    ) -> dict:
        """Execute fast memory recall."""
        if not memory:
            return {"success": False, "answer": None, "confidence": 0.0}
        
        #Recall relevant memories
        memories = await memory.recall(question, limit=5)
        
        if not memories:
            # If strict mode, fail immediately if no memories found
            if context and context.get("strict_mode"):
                return {
                    "success": True, 
                    "answer": "I do not have enough information in my database to answer this question.", 
                    "confidence": 1.0, 
                    "source": "strict_compliance",
                    "context": {"memories": []}
                }
            return {"success": False, "answer": None, "confidence": 0.0, "context": {"memories": []}}
        
        memory_dicts = [m.to_dict() if hasattr(m, 'to_dict') else str(m) for m in memories]

        # Use LLM to synthesize answer from memories
        if self.llm_plugin:
            context_text = "\n".join([
                str(m.content if hasattr(m, 'content') else m.answer)
                for m in memories
            ])
            
            # Use stricter prompt if requested
            is_strict = context and context.get("strict_mode")
            if is_strict:
                prompt = f"""You are a STRICT knowledge engine.
answer ONLY using the provided context below. 
Do NOT use ANY outside knowledge. 
If the answer is not in the context, say "I do not have enough information."

CONTEXT:
{context_text}

QUESTION: 
{question}

ANSWER (Strictly from context):"""
            else:
                prompt = f"Based on this context:\n{context_text}\n\nAnswer: {question}"

            answer = await self.llm_plugin.generate(
                prompt=prompt,
                model_type="solver",
            )
            return {
                "success": True,
                "answer": answer,
                "confidence": 0.7,
                "source": "memory_recall",
                "context": {"memories": memory_dicts}
            }
        
        # Return first memory as answer
        return {
            "success": True,
            "answer": memories[0].content if hasattr(memories[0], 'content') else memories[0].answer,
            "confidence": 0.5,
            "source": "direct_recall",
            "context": {"memories": memory_dicts}
        }
    
    async def _execute_causal(
        self,
        question: str,
        context: dict,
        world: Optional[WorldModel],
    ) -> dict:
        """Execute causal reasoning."""
        if not world:
            return {"success": False, "answer": None, "confidence": 0.0}
        
        # Find causal chains in world model
        # For now, use LLM with world context
        if self.llm_plugin:
            world_context = f"World has {len(world.states)} states and {len(world.transitions)} transitions."
            answer = await self.llm_plugin.generate(
                prompt=f"Using causal reasoning.\nContext: {world_context}\n\nQuestion: {question}",
                model_type="reasoning",
            )
            return {
                "success": True,
                "answer": answer,
                "confidence": 0.6,
                "source": "causal_reasoning",
            }
        
        return {"success": False, "answer": None, "confidence": 0.0}
    
    async def _execute_simulation(
        self,
        question: str,
        context: dict,
        world: Optional[WorldModel],
    ) -> dict:
        """Execute predictive simulation."""
        if not world:
            return {"success": False, "answer": None, "confidence": 0.0}
        
        # Use world model predictions
        if self.llm_plugin:
            answer = await self.llm_plugin.generate(
                prompt=f"Simulate the outcome.\n\nQuestion: {question}",
                model_type="reasoning",
            )
            return {
                "success": True,
                "answer": answer,
                "confidence": 0.5,  # Lower confidence for predictions
                "source": "simulation",
            }
        
        return {"success": False, "answer": None, "confidence": 0.0}
    
    async def _execute_verification(
        self,
        question: str,
        context: dict,
        memory: Optional[Memory],
    ) -> dict:
        """Execute verification/cross-checking."""
        if not self.llm_plugin:
            return {"success": False, "answer": None, "confidence": 0.0}
        
        # Generate answer
        initial_answer = await self.llm_plugin.generate(
            prompt=f"Answer this question:\n{question}",
            model_type="solver",
        )
        
        # Verify with critic
        verification = await self.llm_plugin.generate(
            prompt=f"Question: {question}\nAnswer: {initial_answer}\n\nIs this answer correct? Explain.",
            model_type="critic",
        )
        
        # Parse verification result
        is_correct = "yes" in verification.lower() or "correct" in verification.lower()
        
        return {
            "success": is_correct,
            "answer": initial_answer,
            "confidence": 0.8 if is_correct else 0.3,
            "verification": verification,
            "source": "verification",
        }
    
    async def _execute_multi_hop(
        self,
        question: str,
        context: dict,
        memory: Optional[Memory],
        world: Optional[WorldModel],
    ) -> dict:
        """Execute multi-hop reasoning."""
        if not self.llm_plugin:
            return {"success": False, "answer": None, "confidence": 0.0}
        
        # Decompose into steps
        decomposition = await self.llm_plugin.generate(
            prompt=f"Break this question into 2-3 simpler steps:\n{question}",
            model_type="reasoning",
        )
        
        # Execute each step
        intermediate_results = []
        for step in decomposition.split("\n")[:3]:
            if step.strip():
                step_result = await self.llm_plugin.generate(
                    prompt=f"Previous context: {intermediate_results}\n\nAnswer: {step}",
                    model_type="solver",
                )
                intermediate_results.append(step_result)
        
        # Synthesize final answer
        final_answer = await self.llm_plugin.generate(
            prompt=f"Question: {question}\nSteps: {intermediate_results}\n\nFinal answer:",
            model_type="solver",
        )
        
        return {
            "success": True,
            "answer": final_answer,
            "confidence": 0.7,
            "steps": intermediate_results,
            "source": "multi_hop",
        }
    
    async def _execute_analogical(
        self,
        question: str,
        memory: Optional[Memory],
    ) -> dict:
        """Execute analogical reasoning."""
        if not memory or not self.llm_plugin:
            return {"success": False, "answer": None, "confidence": 0.0}
        
        # Find similar past episodes
        episodes = await memory.recall(question)
        
        if not episodes:
            return {"success": False, "answer": None, "confidence": 0.0, "context": {"memories": []}}
        
        memory_dicts = [e.to_dict() if hasattr(e, 'to_dict') else str(e) for e in episodes]

        # Use analogy from similar case
        similar_context = "\n".join([
            f"Q: {e.question}\nA: {e.answer}" 
            for e in episodes 
            if hasattr(e, 'question')
        ][:3])
        
        answer = await self.llm_plugin.generate(
            prompt=f"Similar cases:\n{similar_context}\n\nBy analogy, answer: {question}",
            model_type="solver",
        )
        
        return {
            "success": True,
            "answer": answer,
            "confidence": 0.6,
            "source": "analogical",
            "context": {"memories": memory_dicts}
        }
    
    async def _execute_abductive(
        self,
        question: str,
        context: dict,
    ) -> dict:
        """Execute abductive reasoning (best explanation)."""
        if not self.llm_plugin:
            return {"success": False, "answer": None, "confidence": 0.0}
        
        # Generate possible explanations
        explanations = await self.llm_plugin.generate(
            prompt=f"List 3 possible explanations for:\n{question}",
            model_type="reasoning",
        )
        
        # Pick best explanation
        best = await self.llm_plugin.generate(
            prompt=f"Explanations:\n{explanations}\n\nWhich is most likely and why?",
            model_type="reasoning",
        )
        
        return {
            "success": True,
            "answer": best,
            "confidence": 0.5,
            "explanations": explanations,
            "source": "abductive",
        }
    
    async def _execute_hybrid(
        self,
        question: str,
        context: dict,
        memory: Optional[Memory],
        world: Optional[WorldModel],
    ) -> dict:
        """Execute hybrid reasoning combining multiple strategies."""
        results = []
        
        # Try fast recall first
        recall_result = await self._execute_fast_recall(question, memory, context)
        if recall_result["success"]:
            results.append(recall_result)
        
        # Try causal if applicable
        if world and self._looks_causal(question):
            causal_result = await self._execute_causal(question, context, world)
            if causal_result["success"]:
                results.append(causal_result)
        
        # Return strict compliance immediately if strictly enforced
        if context.get("strict_mode") and results:
             for r in results:
                 if r.get("source") == "strict_compliance":
                     return r

        # Collect all memories from results
        all_memories = []
        for r in results:
             if "context" in r and "memories" in r["context"]:
                 all_memories.extend(r["context"]["memories"])

        # Verify the results
        if results and self.llm_plugin:
            answers = [r["answer"] for r in results]
            
            is_strict = context.get("strict_mode", False)
            if is_strict:
                prompt = f"""You are a STRICT knowledge aggregator.
Synthesize the provided answers into a final answer.
Do NOT use ANY outside knowledge.
If the provided answers say "I do not have enough information", then your answer must be "I do not have enough information."

POSSIBLE ANSWERS FROM DATABASE:
{answers}

QUESTION:
{question}

FINAL ANSWER (Strictly from inputs):"""
            else:
                prompt = f"Question: {question}\nPossible answers: {answers}\n\nBest answer:"
                
            final = await self.llm_plugin.generate(
                prompt=prompt,
                model_type="solver",
            )
            return {
                "success": True,
                "answer": final,
                "confidence": max(r["confidence"] for r in results),
                "sources": [r["source"] for r in results],
                "source": "hybrid",
                "context": {"memories": all_memories}
            }
        
        if results:
            return results[0]
        
        return {"success": False, "answer": None, "confidence": 0.0}
    
    def _update_stats(
        self,
        strategy: ReasoningStrategy,
        success: bool,
        confidence: float,
    ) -> None:
        """Update strategy performance statistics."""
        stats = self.strategy_stats[strategy.value]
        stats["attempts"] += 1
        if success:
            stats["successes"] += 1
        
        # Running average of confidence
        n = stats["attempts"]
        stats["avg_confidence"] = (
            (stats["avg_confidence"] * (n - 1) + confidence) / n
        )
    
    def adjust_strategy_weight(
        self,
        strategy: ReasoningStrategy,
        adjustment: float,
    ) -> None:
        """
        Adjust the weight of a strategy.
        
        Called by meta-cognition to tune strategy preferences.
        
        Args:
            strategy: Strategy to adjust
            adjustment: Amount to add/subtract (-1 to +1)
        """
        current = self.strategy_weights.get(strategy.value, 1.0)
        new_weight = max(0.1, min(2.0, current + adjustment))
        self.strategy_weights[strategy.value] = new_weight
        
        logger.info(
            "strategy_weight_adjusted",
            strategy=strategy.value,
            old_weight=current,
            new_weight=new_weight,
        )
    
    def get_stats(self) -> dict:
        """Get statistics about reasoning performance."""
        return {
            "total_decisions": len(self.decisions),
            "strategy_stats": self.strategy_stats,
            "strategy_weights": self.strategy_weights,
            "recent_strategies": [
                d.strategy.value for d in self.decisions[-10:]
            ],
        }
