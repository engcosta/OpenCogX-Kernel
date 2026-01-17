"""
🔁 Learning Loop
================

The heart of the POC - autonomous learning without user input.

Loop Structure:
1. Goal Generation (identify uncertainty/contradictions)
2. Question Proposal (multi-hop reasoning questions)
3. Answer Attempt (hybrid retrieval)
4. Critic Evaluation
5. Outcome Recording

Key Principle:
- This loop runs WITHOUT user interaction
- Failure is recorded and used, not hidden
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import structlog

from agi_kernel.core.world import WorldModel, Event
from agi_kernel.core.memory import Memory, MemoryType
from agi_kernel.core.goals import GoalEngine, Goal, GoalType
from agi_kernel.core.reasoning import ReasoningController, ReasoningContext, ReasoningStrategy
from agi_kernel.core.meta import MetaCognition
from agi_kernel.plugins.llm import LLMPlugin
from agi_kernel.plugins.vector import VectorPlugin
from agi_kernel.plugins.graph import GraphPlugin

logger = structlog.get_logger()


@dataclass
class LoopIteration:
    """Record of a single learning loop iteration."""
    id: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Goal
    goal_type: Optional[str] = None
    goal_description: str = ""
    
    # Question
    question: str = ""
    question_type: str = ""  # e.g., "multi_hop", "causal", "factual"
    
    # Answer
    answer: str = ""
    strategy_used: str = ""
    confidence: float = 0.0
    
    # Critique
    verdict: str = ""  # PASS, FAIL, PARTIAL
    issues: list[str] = field(default_factory=list)
    
    # Outcome
    knowledge_gained: bool = False
    gap_recorded: bool = False
    
    # Duration
    duration_ms: float = 0.0
    
    def to_dict(self) -> dict:
        """Serialize iteration for logging."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "goal_type": self.goal_type,
            "question": self.question[:100],
            "strategy": self.strategy_used,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "duration_ms": self.duration_ms,
        }


class LearningLoop:
    """
    The Learning Loop is the autonomous learning engine.
    
    It runs continuously (or on-demand) to:
    1. Identify what the system doesn't know
    2. Generate questions to fill gaps
    3. Attempt to answer using hybrid reasoning
    4. Critique answers for validity
    5. Record outcomes for future learning
    
    This is the core of the POC - proving the system can
    improve without user input or model retraining.
    """
    
    def __init__(
        self,
        world: WorldModel,
        memory: Memory,
        goals: GoalEngine,
        reasoning: ReasoningController,
        meta: MetaCognition,
        llm_plugin: Optional[LLMPlugin] = None,
        vector_plugin: Optional[VectorPlugin] = None,
        graph_plugin: Optional[GraphPlugin] = None,
        max_retries: int = 3,
    ):
        """
        Initialize the Learning Loop.
        
        Args:
            world: World model instance
            memory: Memory system
            goals: Goal engine
            reasoning: Reasoning controller
            meta: Meta-cognition
            llm_plugin: LLM for generation
            vector_plugin: Vector DB for semantic search
            graph_plugin: Graph DB for relations
            max_retries: Max retries per question
        """
        self.world = world
        self.memory = memory
        self.goals = goals
        self.reasoning = reasoning
        self.meta = meta
        
        self.llm = llm_plugin
        self.vector = vector_plugin
        self.graph = graph_plugin
        
        self.max_retries = max_retries
        
        # Loop state
        self.iteration_count = 0
        self.iterations: list[LoopIteration] = []
        self.running = False
        
        logger.info("learning_loop_initialized")
    
    async def step(self) -> LoopIteration:
        """
        Execute a single learning iteration.
        """
        start_time = datetime.utcnow()
        self.iteration_count += 1
        
        iteration = LoopIteration(id=self.iteration_count)
        
        # Rich console log
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        
        console.print(f"\n[bold white]🔄 Learning Iteration #{self.iteration_count}[/bold white]")
        
        try:
            # Step A: Goal Generation
            # logger.info("step_a_goal_generation")
            generated_goals = await self.goals.generate(self.memory, self.world)
            
            if not generated_goals:
                generated_goals = [Goal(
                    type=GoalType.EXPLORE_UNKNOWN,
                    description="Explore knowledge domain",
                    priority=0.5,
                )]
            
            # Step B: Goal Prioritization
            goal = self.goals.prioritize(generated_goals)
            
            if goal:
                iteration.goal_type = goal.type.value
                iteration.goal_description = goal.description
                
                logger.info("goal_selected", goal_type=goal.type.value)
                console.print(f"   [yellow]🎯 Goal:[/yellow] {goal.description} [dim]({goal.type.value})[/dim]")
            
            # Step C: Question Generation
            # logger.info("step_c_question_generation")
            question = await self._generate_question(goal)
            iteration.question = question
            iteration.question_type = self._classify_question(question)
            
            # console.print(f"   [cyan]❓ Question:[/cyan] {question}")
            
            # Step D: Answer Attempt
            # logger.info("step_d_answer_attempt")
            result = await self._attempt_answer(question, goal)
            
            iteration.answer = result.get("answer", "")
            iteration.strategy_used = result.get("strategy", "unknown")
            iteration.confidence = result.get("confidence", 0.0)
            
            # Step E: Critique
            # logger.info("step_e_critique")
            critique = await self._critique_answer(
                question=question,
                answer=result.get("answer", ""),
                context=result.get("context", ""),
            )
            
            iteration.verdict = critique.get("verdict", "FAIL")
            iteration.issues = critique.get("issues", [])
            
            verdict_color = "green" if iteration.verdict == "PASS" else "red"
            console.print(f"   [{verdict_color}]⚖️  Critique Verdict: {iteration.verdict}[/{verdict_color}]")
            if iteration.issues:
                for issue in iteration.issues:
                    console.print(f"      [dim]- {issue}[/dim]")
            
            # Step F: Record Outcome
            await self._record_outcome(iteration, goal)
            
            if iteration.knowledge_gained:
                console.print(f"   [bold green]📚 Knowledge Gained![/bold green]")
            if iteration.gap_recorded:
                console.print(f"   [bold orange3]🕳️  Knowledge Gap Recorded[/bold orange3]")
            
            # Step G: Meta-Cognition Evaluation
            outcome = {
                "success": iteration.verdict == "PASS",
                "question": question,
                "strategy": iteration.strategy_used,
                "confidence": iteration.confidence,
            }
            self.meta.evaluate(outcome, self.memory, self.reasoning)
            
        except Exception as e:
            logger.error("learning_step_failed", error=str(e))
            iteration.verdict = "ERROR"
            iteration.issues = [str(e)]
            console.print(f"   [bold red]💥 Error in Learning Loop:[/bold red] {e}")
        
        # Calculate duration
        end_time = datetime.utcnow()
        iteration.duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Store iteration
        self.iterations.append(iteration)
        
        logger.info(
            "learning_iteration_complete",
            iteration=self.iteration_count,
            verdict=iteration.verdict,
            duration_ms=iteration.duration_ms,
        )
        
        return iteration
    
    async def _generate_question(self, goal: Optional[Goal]) -> str:
        """
        Generate a question based on the goal.
        
        Questions are NOT for users - they are for the system itself.
        """
        if not self.llm:
            return "What is the main concept in this domain?"
        
        # Build prompt based on goal type
        goal_context = ""
        if goal:
            if goal.type == GoalType.REDUCE_UNCERTAINTY:
                goal_context = f"Focus on clarifying: {goal.target_entity or 'uncertain areas'}"
            elif goal.type == GoalType.RESOLVE_CONTRADICTION:
                goal_context = f"Focus on resolving: {goal.description}"
            elif goal.type == GoalType.FILL_KNOWLEDGE_GAP:
                goal_context = f"Focus on learning about: {goal.target_entity}"
            elif goal.type == GoalType.IMPROVE_PREDICTION:
                goal_context = "Focus on understanding causal relationships"
            else:
                goal_context = goal.description
        
        # Get context from memory
        memory_context = ""
        memories = await self.memory.recall("domain knowledge", limit=3)
        if memories:
            memory_context = "Known facts:\n" + "\n".join([
                str(m.content if hasattr(m, 'content') else m.answer)[:100]
                for m in memories
            ])
        
        prompt = f"""Generate a thoughtful, multi-hop reasoning question.

{goal_context}

{memory_context}

Generate ONE question that:
- Requires connecting multiple pieces of information
- Explores relationships or causality
- Helps fill knowledge gaps

Question:"""
        
        question = await self.llm.generate(
            prompt=prompt,
            model_type="reasoning",
            temperature=0.8,
        )
        
        return question.strip()
    
    def _classify_question(self, question: str) -> str:
        """Classify the type of question."""
        question_lower = question.lower()
        
        if "why" in question_lower or "cause" in question_lower:
            return "causal"
        elif "how" in question_lower:
            return "procedural"
        elif "what" in question_lower and "relationship" in question_lower:
            return "relational"
        elif "and" in question_lower or "through" in question_lower:
            return "multi_hop"
        else:
            return "factual"
    
    async def _attempt_answer(
        self,
        question: str,
        goal: Optional[Goal],
    ) -> dict:
        """
        Attempt to answer the question using hybrid reasoning.
        
        Combines:
        - Vector search (facts from Qdrant)
        - Graph traversal (relations from Neo4j)
        - LLM reasoning (synthesis)
        """
        # Build reasoning context
        context = ReasoningContext(
            question=question,
            goal_type=goal.type.value if goal else None,
            available_memory_types=["semantic", "episodic"],
            has_graph=self.graph is not None,
            has_vector=self.vector is not None,
            complexity_estimate=self._estimate_complexity(question),
        )
        
        # Choose strategy
        strategy, reason = self.reasoning.choose_strategy(context)
        
        # Execute reasoning
        result = await self.reasoning.execute(
            strategy=strategy,
            question=question,
            context={"strategy_reason": reason},
            memory=self.memory,
            world=self.world,
        )
        
        result["strategy"] = strategy.value
        return result
    
    def _estimate_complexity(self, question: str) -> float:
        """Estimate question complexity (0-1)."""
        complexity = 0.3  # Base
        
        # Multi-part increases complexity
        if " and " in question:
            complexity += 0.2
        
        # Causal questions are more complex
        if "why" in question.lower() or "how" in question.lower():
            complexity += 0.2
        
        # Longer questions tend to be more complex
        if len(question) > 100:
            complexity += 0.1
        
        return min(1.0, complexity)
    
    async def _critique_answer(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> dict:
        """
        Critique the answer for validity.
        
        Checks:
        - Completeness
        - Consistency
        - Support from knowledge base
        """
        if not self.llm or not answer:
            return {
                "verdict": "FAIL",
                "confidence": 0.0,
                "issues": ["No answer generated"],
            }
        
        return await self.llm.critique(
            question=question,
            answer=answer,
            context=context,
        )
    
    async def _record_outcome(
        self,
        iteration: LoopIteration,
        goal: Optional[Goal],
    ) -> None:
        """
        Record the outcome of this iteration.
        
        - PASS: Store in episodic memory as success
        - FAIL: Record as knowledge gap, retry later
        - PARTIAL: Store with low confidence
        """
        if iteration.verdict == "PASS":
            # Store successful learning
            self.memory.store_episode(
                question=iteration.question,
                answer=iteration.answer,
                outcome="PASS",
                reasoning_strategy=iteration.strategy_used,
                confidence=iteration.confidence,
            )
            iteration.knowledge_gained = True
            
            # CRITICAL: Consolidate knowledge into long-term memory (Graph & Vector)
            # This ensures the graph evolves over time.
            if self.graph and self.llm:
                await self._consolidate_knowledge(iteration)
            
            # Complete the goal
            if goal:
                self.goals.complete_goal(goal.id, actual_gain=0.5, success=True)
                
        elif iteration.verdict == "FAIL":
            # Record failure for learning
            self.memory.store_episode(
                question=iteration.question,
                answer=iteration.answer,
                outcome="FAIL",
                reasoning_strategy=iteration.strategy_used,
                confidence=iteration.confidence,
                failure_reason=str(iteration.issues),
            )
            iteration.gap_recorded = True
            
            # Mark goal as needing retry
            if goal:
                goal.attempts += 1
                if goal.attempts >= goal.max_attempts:
                    self.goals.complete_goal(goal.id, actual_gain=0.0, success=False)
                    
        else:  # PARTIAL
            self.memory.store_episode(
                question=iteration.question,
                answer=iteration.answer,
                outcome="PARTIAL",
                reasoning_strategy=iteration.strategy_used,
                confidence=iteration.confidence * 0.5,  # Lower confidence
            )
            iteration.knowledge_gained = True
            
            # Also attempt to consolidate partial knowledge if high enough quality
            if iteration.confidence > 0.6 and self.graph and self.llm:
                await self._consolidate_knowledge(iteration)

    async def _consolidate_knowledge(self, iteration: LoopIteration) -> None:
        """
        Extract entities and relations from the learned answer and persist to Graph & Vector.
        """
        try:
            from rich import print as rprint
            rprint("[dim]   ... Consolidating new knowledge into Graph & VectorDB ...[/dim]")
            
            # 1. Extract Triples using LLM
            prompt = f"""Analyze this Q&A pair and extract new knowledge as strict triples.
            
Question: {iteration.question}
Answer: {iteration.answer}

Extract 3-5 key relations that explain this answer.
Format: SUBJECT | RELATION | OBJECT | TYPE
- SUBJECT/OBJECT: The entity names (e.g., "Leaderless Replication", "Quorum")
- RELATION: The relationship (e.g., "uses", "requires", "improves", "avoids")
- TYPE: The type of the subject entity (e.g., "CONCEPT", "TECHNOLOGY", "ALGORITHM")

Examples:
Leaderless Replication | uses | Read Repair | CONCEPT
Dynamo | implements | Leaderless Replication | TECHNOLOGY

Return ONLY the triples, one per line.
"""
            extraction = await self.llm.generate(
                prompt=prompt,
                model_type="solver",
            )
            
            triples = []
            for line in extraction.strip().split('\n'):
                parts = line.split('|')
                if len(parts) >= 4:
                    subj = parts[0].strip()
                    rel = parts[1].strip().upper().replace(' ', '_')
                    obj = parts[2].strip()
                    subj_type = parts[3].strip().upper()
                    
                    # Create IDs
                    subj_id = subj.lower().replace(' ', '_')
                    obj_id = obj.lower().replace(' ', '_')
                    
                    triples.append({
                        "from_id": subj_id,
                        "from_name": subj,
                        "from_type": subj_type,
                        "rel": rel,
                        "to_id": obj_id,
                        "to_name": obj
                    })

            # 2. Store in Graph (Neo4j)
            count_nodes = 0
            count_rels = 0
            for t in triples:
                # Ensure entities exist
                await self.graph.store_entity(
                    entity_id=t["from_id"],
                    entity_type=t["from_type"],
                    properties={"name": t["from_name"], "source": "learning_loop"}
                )
                await self.graph.store_entity(
                    entity_id=t["to_id"],
                    entity_type="CONCEPT", # Default type for object if unknown
                    properties={"name": t["to_name"], "source": "learning_loop"}
                )
                
                # Store relation
                await self.graph.store_relation(
                    from_entity=t["from_id"],
                    to_entity=t["to_id"],
                    relation_type=t["rel"],
                    properties={"confidence": iteration.confidence, "source": "learning_loop"}
                )
                count_nodes += 2
                count_rels += 1
                
            rprint(f"[dim]   ... Graph Updated: +{count_rels} relations[/dim]")

            # 3. Store in Vector DB (Qdrant) - Create a specific "insight" memory
            if self.vector:
                from agi_kernel.core.memory import MemoryItem, MemoryType
                
                content_text = f"Question: {iteration.question}\nAnswer: {iteration.answer}"
                embedding = await self.llm.embed(content_text)
                
                item = MemoryItem(
                    type=MemoryType.SEMANTIC,
                    content={"text": content_text, "source": "learning_insight"},
                    source="learning_loop",
                    confidence=iteration.confidence
                )
                
                await self.vector.store_memory(item, embedding=embedding)
                rprint(f"[dim]   ... Vector Updated: +1 insight[/dim]")

        except Exception as e:
            logger.error("knowledge_consolidation_failed", error=str(e))
            rprint(f"[red]   Failed to consolidate knowledge: {e}[/red]")
    
    async def run(
        self,
        iterations: int = 10,
        interval_seconds: float = 5.0,
    ) -> list[LoopIteration]:
        """
        Run the learning loop for multiple iterations.
        
        Args:
            iterations: Number of iterations to run
            interval_seconds: Wait time between iterations
            
        Returns:
            List of iteration records
        """
        logger.info(
            "learning_loop_starting",
            iterations=iterations,
            interval=interval_seconds,
        )
        
        self.running = True
        results = []
        
        for i in range(iterations):
            if not self.running:
                break
            
            try:
                iteration = await self.step()
                results.append(iteration)
                
                # Apply memory decay periodically
                if i % 5 == 0:
                    self.memory.decay()
                
                # Wait before next iteration
                if i < iterations - 1:
                    await asyncio.sleep(interval_seconds)
                    
            except Exception as e:
                logger.error("loop_iteration_error", iteration=i, error=str(e))
        
        self.running = False
        
        logger.info(
            "learning_loop_complete",
            total_iterations=len(results),
            passed=sum(1 for r in results if r.verdict == "PASS"),
            failed=sum(1 for r in results if r.verdict == "FAIL"),
        )
        
        return results
    
    def stop(self) -> None:
        """Stop the learning loop."""
        self.running = False
        logger.info("learning_loop_stopped")
    
    def get_metrics(self) -> dict:
        """
        Get metrics about the learning loop.
        
        This is key for the POC evaluation.
        """
        if not self.iterations:
            return {"iterations": 0}
        
        total = len(self.iterations)
        passed = sum(1 for i in self.iterations if i.verdict == "PASS")
        failed = sum(1 for i in self.iterations if i.verdict == "FAIL")
        partial = sum(1 for i in self.iterations if i.verdict == "PARTIAL")
        
        # Strategy usage
        strategies = {}
        for i in self.iterations:
            s = i.strategy_used
            if s not in strategies:
                strategies[s] = {"count": 0, "passed": 0}
            strategies[s]["count"] += 1
            if i.verdict == "PASS":
                strategies[s]["passed"] += 1
        
        # Calculate pass rate over time (for improvement tracking)
        window_size = 5
        pass_rate_over_time = []
        for i in range(0, total, window_size):
            window = self.iterations[i:i + window_size]
            if window:
                rate = sum(1 for w in window if w.verdict == "PASS") / len(window)
                pass_rate_over_time.append(rate)
        
        return {
            "total_iterations": total,
            "passed": passed,
            "failed": failed,
            "partial": partial,
            "pass_rate": passed / total if total > 0 else 0,
            "average_confidence": sum(i.confidence for i in self.iterations) / total,
            "average_duration_ms": sum(i.duration_ms for i in self.iterations) / total,
            "strategies_used": strategies,
            "pass_rate_over_time": pass_rate_over_time,
            "knowledge_gained_count": sum(1 for i in self.iterations if i.knowledge_gained),
            "gaps_recorded_count": sum(1 for i in self.iterations if i.gap_recorded),
        }
