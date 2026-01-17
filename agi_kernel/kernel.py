"""
🧬 AGI Kernel
=============

The main Kernel class that orchestrates all components.

This is the minimal execution loop:
    while True:
        event = perceive()
        state = world.observe(event)
        goals = goals.generate(memory, world)
        goal = goals.prioritize(goals)
        strategy = reasoning.choose_strategy(state)
        outcome = reasoning.execute(strategy)
        memory.store(outcome)
        meta.evaluate(outcome)
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional
from dotenv import load_dotenv
import structlog

from agi_kernel.core.world import WorldModel
from agi_kernel.core.memory import Memory
from agi_kernel.core.goals import GoalEngine
from agi_kernel.core.reasoning import ReasoningController
from agi_kernel.core.meta import MetaCognition
from agi_kernel.plugins.llm import LLMPlugin
from agi_kernel.plugins.vector import VectorPlugin
from agi_kernel.plugins.graph import GraphPlugin
from agi_kernel.plugins.persistence import PersistencePlugin
from agi_kernel.learning_loop import LearningLoop
from agi_kernel.ingestion import IngestionPipeline
from agi_kernel.metrics import MetricsCollector

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class Kernel:
    """
    The AGI Kernel orchestrates all cognitive components.
    
    Components:
    - World Model: Represents states, events, causality
    - Memory: Semantic, episodic, temporal storage
    - Goal Engine: Intrinsic motivation
    - Reasoning: Dynamic strategy selection
    - Meta-Cognition: Self-monitoring and adaptation
    
    Plugins:
    - LLM: Language model adapter
    - Vector: Semantic search (Qdrant)
    - Graph: Knowledge graph (Neo4j)
    """
    
    def __init__(
        self,
        env_file: str = ".env",
        use_plugins: bool = True,
    ):
        """
        Initialize the AGI Kernel.
        
        Args:
            env_file: Path to environment file
            use_plugins: Whether to initialize external plugins
        """
        # Load environment variables
        load_dotenv(env_file)
        
        logger.info("kernel_initializing")
        
        # Initialize core components
        self.world = WorldModel()
        self.memory = Memory(
            decay_rate=float(os.getenv("MEMORY_DECAY_RATE", "0.1")),
        )
        self.goals = GoalEngine()
        self.reasoning = ReasoningController()
        self.meta = MetaCognition()
        
        # Initialize plugins if requested
        self.llm: Optional[LLMPlugin] = None
        self.vector: Optional[VectorPlugin] = None
        self.graph: Optional[GraphPlugin] = None
        self.persistence: Optional[PersistencePlugin] = None
        
        if use_plugins:
            self._init_plugins()
        
        # Initialize derived components
        self.learning_loop: Optional[LearningLoop] = None
        self.ingestion: Optional[IngestionPipeline] = None
        self.metrics: Optional[MetricsCollector] = None
        
        self._init_derived_components()
        
        logger.info("kernel_initialized")
    
    def _init_plugins(self) -> None:
        """Initialize external plugins."""
        try:
            # LLM Plugin
            self.llm = LLMPlugin(
                ollama_url=os.getenv("OLLAMA_BASE_URL"),
                lm_studio_url=os.getenv("LM_STUDIO_BASE_URL"),
                reasoning_model=os.getenv("REASONING_MODEL", "qwen3:8b"),
                solver_model=os.getenv("SOLVER_MODEL", "llama3.2:3b"),
                critic_model=os.getenv("CRITIC_MODEL", "llama3.2:3b"),
                embedding_model=os.getenv("EMBEDDING_MODEL", "qwen3-embedding:4b"),
            )
            self.reasoning.llm_plugin = self.llm
            logger.info("llm_plugin_initialized")
        except Exception as e:
            logger.warning("llm_plugin_failed", error=str(e))
        
        try:
            # Vector Plugin
            self.vector = VectorPlugin(
                host=os.getenv("QDRANT_HOST"),
                port=int(os.getenv("QDRANT_PORT", "6333")),
                llm_plugin=self.llm,
            )
            self.memory.vector_plugin = self.vector
            self.reasoning.vector_plugin = self.vector
            logger.info("vector_plugin_initialized")
        except Exception as e:
            logger.warning("vector_plugin_failed", error=str(e))
        
        try:
            # Graph Plugin
            self.graph = GraphPlugin(
                uri=os.getenv("NEO4J_URI"),
                user=os.getenv("NEO4J_USER"),
                password=os.getenv("NEO4J_PASSWORD"),
            )
            self.world.graph_plugin = self.graph
            self.goals.graph_plugin = self.graph
            self.reasoning.graph_plugin = self.graph
            logger.info("graph_plugin_initialized")
        except Exception as e:
            logger.warning("graph_plugin_failed", error=str(e))

        try:
            # Persistence Plugin
            self.persistence = PersistencePlugin(db_path="agi_state.db")
            # We delay goals injection until async init
            logger.info("persistence_plugin_initialized")
        except Exception as e:
            logger.warning("persistence_plugin_failed", error=str(e))
    
    def _init_derived_components(self) -> None:
        """Initialize components that depend on core and plugins."""
        # Learning Loop
        self.learning_loop = LearningLoop(
            world=self.world,
            memory=self.memory,
            goals=self.goals,
            reasoning=self.reasoning,
            meta=self.meta,
            llm_plugin=self.llm,
            vector_plugin=self.vector,
            graph_plugin=self.graph,
            persistence_plugin=self.persistence,
            max_retries=int(os.getenv("MAX_RETRY_ON_FAIL", "3")),
        )
        
        # Ingestion Pipeline
        self.ingestion = IngestionPipeline(
            memory=self.memory,
            world=self.world,
            llm_plugin=self.llm,
            vector_plugin=self.vector,
            graph_plugin=self.graph,
        )
        
        # Metrics Collector
        self.metrics = MetricsCollector(output_dir="./metrics")
    
    async def initialize_plugins(self) -> dict:
        """
        Initialize plugin connections (async).
        
        Returns:
            Status of each plugin
        """
        status = {}
        
        if self.vector:
            status["vector"] = await self.vector.initialize()
        
        if self.graph:
            status["graph"] = await self.graph.initialize()
        
        if self.persistence:
            status["persistence"] = await self.persistence.initialize()
            # Inject into goals and load state
            await self.goals.initialize_persistence(self.persistence)
            
            # Inject into world
            self.world.persistence_plugin = self.persistence

        if self.llm:
            status["llm"] = await self.llm.check_health()
        
        logger.info("plugins_initialized", status=status)
        return status
    
    async def ingest(
        self,
        path: str,
        is_directory: bool = False,
    ) -> dict:
        """
        Ingest documents into the knowledge base.
        
        Args:
            path: File or directory path
            is_directory: Whether path is a directory
            
        Returns:
            Ingestion statistics
        """
        if is_directory:
            return await self.ingestion.ingest_directory(path)
        else:
            return await self.ingestion.ingest_file(path)
    
    async def learn(
        self,
        iterations: int = 10,
        interval_seconds: float = 5.0,
    ) -> dict:
        """
        Run the learning loop.
        
        Args:
            iterations: Number of learning iterations
            interval_seconds: Wait between iterations
            
        Returns:
            Learning metrics
        """
        logger.info("learning_started", iterations=iterations)
        
        # Run learning loop
        results = await self.learning_loop.run(iterations, interval_seconds)
        
        # Collect final snapshot
        self.metrics.collect_snapshot(
            memory=self.memory,
            world=self.world,
            goals=self.goals,
            reasoning=self.reasoning,
            meta=self.meta,
            learning_loop=self.learning_loop,
        )
        
        return self.learning_loop.get_metrics()
    
    async def step(self):
        """Execute a single learning step."""
        return await self.learning_loop.step()
    
    def evaluate(self) -> dict:
        """
        Generate evaluation report.
        
        Returns:
            Comprehensive evaluation report
        """
        # Collect final snapshot
        self.metrics.collect_snapshot(
            memory=self.memory,
            world=self.world,
            goals=self.goals,
            reasoning=self.reasoning,
            meta=self.meta,
            learning_loop=self.learning_loop,
        )
        
        return self.metrics.generate_report()
    
    def get_status(self) -> dict:
        """Get current kernel status."""
        return {
            "world": self.world.get_stats(),
            "memory": self.memory.get_stats(),
            "goals": self.goals.get_stats(),
            "reasoning": self.reasoning.get_stats(),
            "meta": self.meta.get_stats(),
            "learning_loop": (
                self.learning_loop.get_metrics() if self.learning_loop else {}
            ),
            "ingestion": (
                self.ingestion.get_stats() if self.ingestion else {}
            ),
        }
    
    async def close(self) -> None:
        """Clean up resources."""
        if self.llm:
            await self.llm.close()
        if self.vector:
            await self.vector.close()
        if self.graph:
            await self.graph.close()
        
        logger.info("kernel_closed")


async def main():
    """Main entry point for the kernel."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AGI Kernel POC")
    parser.add_argument("--ingest", type=str, help="Path to ingest")
    parser.add_argument("--learn", type=int, default=0, help="Learning iterations")
    parser.add_argument("--evaluate", action="store_true", help="Generate report")
    parser.add_argument("--status", action="store_true", help="Show status")
    
    args = parser.parse_args()
    
    # Initialize kernel
    kernel = Kernel()
    await kernel.initialize_plugins()
    
    try:
        # Ingest if requested
        if args.ingest:
            import os
            is_dir = os.path.isdir(args.ingest)
            result = await kernel.ingest(args.ingest, is_directory=is_dir)
            print(f"Ingestion result: {result}")
        
        # Learn if requested
        if args.learn > 0:
            result = await kernel.learn(iterations=args.learn)
            print(f"Learning result: {result}")
        
        # Evaluate if requested
        if args.evaluate:
            report = kernel.evaluate()
            import json
            print(json.dumps(report, indent=2))
        
        # Status if requested
        if args.status:
            status = kernel.get_status()
            import json
            print(json.dumps(status, indent=2))
        
    finally:
        await kernel.close()


if __name__ == "__main__":
    asyncio.run(main())
