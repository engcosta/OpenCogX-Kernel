"""
🌐 FastAPI Server
=================

HTTP API for interacting with the AGI Kernel.

Endpoints:
- /status: Get kernel status
- /ingest: Ingest documents
- /learn: Run learning loop
- /step: Single learning step
- /evaluate: Get evaluation report
- /memory: Query memory
- /goals: View goals
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Optional
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agi_kernel.kernel import Kernel

logger = structlog.get_logger()

# Global kernel instance
kernel: Optional[Kernel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage kernel lifecycle."""
    global kernel
    
    # Startup
    logger.info("server_starting")
    kernel = Kernel()
    await kernel.initialize_plugins()
    
    yield
    
    # Shutdown
    if kernel:
        await kernel.close()
    logger.info("server_stopped")


app = FastAPI(
    title="AGI Kernel API",
    description="Open AGI Kernel POC - A self-evolving cognitive system",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class IngestRequest(BaseModel):
    path: str
    is_directory: bool = False


class IngestResponse(BaseModel):
    success: bool
    files_processed: Optional[int] = None
    chunks: Optional[int] = None
    entities: Optional[int] = None
    relations: Optional[int] = None
    error: Optional[str] = None


class LearnRequest(BaseModel):
    iterations: int = 10
    interval_seconds: float = 5.0


class LearnResponse(BaseModel):
    success: bool
    total_iterations: int
    passed: int
    failed: int
    pass_rate: float
    average_confidence: float


class StepResponse(BaseModel):
    iteration_id: int
    verdict: str
    question: str
    answer: str
    strategy: str
    confidence: float
    duration_ms: float


class MemoryQuery(BaseModel):
    query: str
    limit: int = 10
    memory_types: Optional[list[str]] = None


class StatusResponse(BaseModel):
    world: dict
    memory: dict
    goals: dict
    reasoning: dict
    meta: dict
    learning_loop: dict
    ingestion: dict



class CleanRequest(BaseModel):
    confirm: bool = False


class AskRequest(BaseModel):
    question: str
    strategy: str = "hybrid"  # hybrid, search, reason
    strict_mode: bool = True  # Default to True per user requirement



# Endpoints
@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "AGI Kernel API",
        "version": "0.1.0",
        "description": "Open AGI Kernel POC - A self-evolving cognitive system",
        "endpoints": [
            "/status",
            "/ingest",
            "/learn",
            "/step",
            "/evaluate",
            "/memory",
            "/goals",
            "/clean",
            "/ask",
        ],
    }


@app.post("/clean")
async def clean_databases(request: CleanRequest):
    """Clear both Neo4j and Qdrant databases."""
    if not request.confirm:
        raise HTTPException(status_code=400, detail="Must set confirm=True to clear databases")
        
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")
    
    try:
        # Clear Graph
        if kernel.graph:
            await kernel.graph.run_cypher("MATCH (n) DETACH DELETE n")
            
        # Clear Vector
        if kernel.vector and kernel.vector._client:
            try:
                kernel.vector._client.delete_collection(kernel.vector.collection_name)
                # Re-init collection immediately
                from qdrant_client.models import VectorParams, Distance
                kernel.vector._client.create_collection(
                    collection_name=kernel.vector.collection_name,
                    vectors_config=VectorParams(size=kernel.vector.vector_size, distance=Distance.COSINE),
                )
            except Exception as v_err:
                logger.warning("vector_clear_failed", error=str(v_err))

        logger.info("databases_cleared_via_api")
        return {"status": "success", "message": "Databases cleared successfully"}
        
    except Exception as e:
        logger.error("clean_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_question(request: AskRequest):
    """Ask a question using the full cognitive architecture."""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")

    try:
        from agi_kernel.core.reasoning import ReasoningContext, ReasoningStrategy
        
        # Determine strategy
        strategy_enum = ReasoningStrategy.HYBRID
        if request.strategy.lower() == "search":
            strategy_enum = ReasoningStrategy.FAST_RECALL
        elif request.strategy.lower() == "reason":
            strategy_enum = ReasoningStrategy.CAUSAL_REASONING
        elif request.strategy.lower() == "auto":
            strategy_enum = ReasoningStrategy.AUTO
            
        # Build Context
        ctx = ReasoningContext(
            question=request.question,
            available_memory_types=["semantic", "episodic"],
            has_vector=kernel.vector is not None,
            has_graph=kernel.graph is not None,
            strict_mode=request.strict_mode,
            # user_intent="user_query"  <-- Removed unknown field
        )
        
        # Execute reasoning
        start_time = asyncio.get_event_loop().time()
        response = await kernel.reasoning.execute(
            strategy=strategy_enum,
            question=request.question,
            context=ctx.__dict__,
            memory=kernel.memory,
            world=kernel.world
        )
        duration = asyncio.get_event_loop().time() - start_time
        
        return {
            "question": request.question,
            "answer": response.get("answer"),
            "confidence": response.get("confidence", 0.0),
            "strategy_used": response.get("strategy", request.strategy),
            "duration_seconds": duration,
            "context_used": response.get("context", {}).get("memories", [])
        }
        
    except Exception as e:
        logger.error("ask_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")
    
    return {
        "status": "healthy",
        "llm": kernel.llm is not None,
        "vector": kernel.vector is not None,
        "graph": kernel.graph is not None,
    }


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current kernel status."""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")
    
    return kernel.get_status()


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """Ingest documents into the knowledge base."""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")
    
    try:
        result = await kernel.ingest(request.path, request.is_directory)
        
        if "error" in result:
            return IngestResponse(success=False, error=result["error"])
        
        return IngestResponse(
            success=True,
            files_processed=result.get("files_processed", 1),
            chunks=result.get("chunks") or result.get("total_chunks", 0),
            entities=result.get("entities") or result.get("total_entities", 0),
            relations=result.get("relations") or result.get("total_relations", 0),
        )
    except Exception as e:
        logger.error("ingest_failed", error=str(e))
        return IngestResponse(success=False, error=str(e))


@app.post("/learn", response_model=LearnResponse)
async def learn(request: LearnRequest, background_tasks: BackgroundTasks):
    """Run the learning loop."""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")
    
    try:
        result = await kernel.learn(
            iterations=request.iterations,
            interval_seconds=request.interval_seconds,
        )
        
        return LearnResponse(
            success=True,
            total_iterations=result.get("total_iterations", 0),
            passed=result.get("passed", 0),
            failed=result.get("failed", 0),
            pass_rate=result.get("pass_rate", 0.0),
            average_confidence=result.get("average_confidence", 0.0),
        )
    except Exception as e:
        logger.error("learn_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResponse)
async def step():
    """Execute a single learning step."""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")
    
    try:
        iteration = await kernel.step()
        
        return StepResponse(
            iteration_id=iteration.id,
            verdict=iteration.verdict,
            question=iteration.question,
            answer=iteration.answer,
            strategy=iteration.strategy_used,
            confidence=iteration.confidence,
            duration_ms=iteration.duration_ms,
        )
    except Exception as e:
        logger.error("step_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluate")
async def evaluate():
    """Generate evaluation report."""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")
    
    return kernel.evaluate()


@app.post("/memory/query")
async def query_memory(query: MemoryQuery):
    """Query the memory system."""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")
    
    from agi_kernel.core.memory import MemoryType
    
    memory_types = None
    if query.memory_types:
        memory_types = [MemoryType(t) for t in query.memory_types]
    
    results = kernel.memory.recall(
        query=query.query,
        limit=query.limit,
        memory_types=memory_types,
    )
    
    return {
        "query": query.query,
        "count": len(results),
        "results": [
            {
                "id": r.id,
                "type": r.type.value if hasattr(r, 'type') else "episodic",
                "content": r.content if hasattr(r, 'content') else {
                    "question": r.question,
                    "answer": r.answer,
                    "outcome": r.outcome,
                },
            }
            for r in results
        ],
    }


@app.get("/memory/stats")
async def memory_stats():
    """Get memory statistics."""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")
    
    return kernel.memory.get_stats()


@app.get("/goals")
async def get_goals():
    """Get current goals."""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")
    
    active = kernel.goals.get_active_goals()
    
    return {
        "active_goals": [g.to_dict() for g in active],
        "completed_count": len(kernel.goals.completed_goals),
        "failed_count": len(kernel.goals.failed_goals),
        "stats": kernel.goals.get_stats(),
    }


@app.get("/world")
async def get_world():
    """Get world model state."""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")
    
    return {
        "stats": kernel.world.get_stats(),
        "recent_states": [
            s.to_dict() for s in list(kernel.world.states.values())[-10:]
        ],
        "recent_events": [
            e.to_dict() for e in list(kernel.world.events.values())[-10:]
        ],
    }


@app.get("/reasoning")
async def get_reasoning():
    """Get reasoning statistics."""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")
    
    return kernel.reasoning.get_stats()


@app.get("/meta")
async def get_meta():
    """Get meta-cognition state."""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")
    
    return {
        "self_knowledge": kernel.meta.get_self_knowledge(),
        "stats": kernel.meta.get_stats(),
        "pending_changes": [
            c.to_dict() for c in kernel.meta.get_pending_changes()
        ],
    }


@app.get("/metrics/export")
async def export_metrics():
    """Export metrics for visualization."""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")
    
    return kernel.metrics.export_for_visualization()


@app.post("/metrics/snapshot")
async def collect_snapshot():
    """Collect a metrics snapshot."""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")
    
    snapshot = kernel.metrics.collect_snapshot(
        memory=kernel.memory,
        world=kernel.world,
        goals=kernel.goals,
        reasoning=kernel.reasoning,
        meta=kernel.meta,
        learning_loop=kernel.learning_loop,
    )
    
    return snapshot.to_dict()


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
