
import asyncio
import os
import sys
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Add current directory to path so we can import agi_kernel
sys.path.append(os.getcwd())

from agi_kernel.kernel import Kernel
from agi_kernel.core.goals import Goal, GoalType

console = Console()

async def get_db_stats(kernel):
    """Get counts from Qdrant and Neo4j."""
    # Qdrant stats (approximate from memory plugin or direct client query if possible)
    # We'll rely on memory container counts which reflect loaded memories
    # But better to query the plugins directly if possible.
    
    # Vector Stats
    vector_count = 0
    if kernel.vector and kernel.vector._client:
        try:
             # Use get_collection in v1.x
            info = kernel.vector._client.get_collection(kernel.vector.collection_name)
            vector_count = info.points_count
        except:
            vector_count = 0

    # Graph Stats
    node_count = 0
    rel_count = 0
    if kernel.graph:
        try:
            async with kernel.graph._driver.session() as session:
                r1 = await session.run("MATCH (n) RETURN count(n) as c")
                node_count = (await r1.single())["c"]
                r2 = await session.run("MATCH ()-[r]->() RETURN count(r) as c")
                rel_count = (await r2.single())["c"]
        except:
            pass
            
    return {
        "vector_points": vector_count,
        "graph_nodes": node_count,
        "graph_relations": rel_count
    }

async def run_eval_test():
    console.print(Panel.fit("🧪 [bold blue]Comprehensive Learning & Evaluation Test[/bold blue]", border_style="blue"))
    
    # Initialize Kernel
    console.print("\n[bold]📦 Initializing Kernel...[/bold]")
    kernel = Kernel(use_plugins=True)
    await kernel.initialize_plugins()
    
    # 1. Ingestion / Baseline
    console.print("\n[bold]📥 Step 1: Ingesting Corpus File[/bold]")
    target_file = os.path.abspath("corpus/database_replication.md")
    if not os.path.exists(target_file):
        console.print(f"[red]File not found: {target_file}[/red]")
        return

    # Measure baseline BEFORE ingestion (to see ingestion impact) OR meaningful baseline?
    # User said: "understand what is already there and how much relation enhanced during learning"
    # So we should probably measure AFTER ingestion but BEFORE learning loop to measure *learning loop* enhancement.
    
    # await kernel.ingest(target_file, is_directory=False)
    
    stats_baseline = await get_db_stats(kernel)
    console.print(f"   [dim]Baseline (Post-Ingest): Vectors={stats_baseline['vector_points']}, Nodes={stats_baseline['graph_nodes']}, Rels={stats_baseline['graph_relations']}[/dim]")

    # 2. Learning Loop (2 Steps)
    console.print("\n[bold]🔄 Step 2: Running Learning Loop (2 Steps)[/bold]")
    
    # Force a relevant goal to ensure we focus on the topic
    # We manually inject a goal into the queue
    target_concept = "leaderless_replication"
    manual_goal = Goal(
        type=GoalType.FILL_KNOWLEDGE_GAP,
        description=f"Explore relations for: {target_concept}",
        priority=1.0,
        expected_gain=1.0,
        target_entity=target_concept
    )
    kernel.goals.add(manual_goal)
    console.print(f"   [yellow]⚠️ Injected Manual Goal: {manual_goal.description}[/yellow]")

    iterations_data = []
    for i in range(2):
        console.print(f"\n   [bold white]Iteration {i+1}...[/bold white]")
        iteration = await kernel.learning_loop.step()
        iterations_data.append(iteration)
    
    stats_after_learning = await get_db_stats(kernel)
    
    # 3. Accuracy Evaluation
    console.print("\n[bold]🎯 Step 3: Knowledge Accuracy Evaluation[/bold]")
    
    eval_questions = [
        {
            "q": "What is the main disadvantage of Single-Leader Replication regarding failure?",
            "keywords": ["leader", "bottleneck", "failover"],
            "ground_truth": "Leader is a bottleneck; Leader failure requires failover"
        },
        {
            "q": "List examples of databases that use Leaderless Replication.",
            "keywords": ["dynamo", "cassandra", "riak"],
            "ground_truth": "Amazon Dynamo, Cassandra, Riak"
        },
        {
            "q": "How does Consistent Hashing minimize work when nodes are added?",
            "keywords": ["data movement", "minimize", "virtual nodes"],
            "ground_truth": "Minimizes data movement; Virtual nodes improve balance"
        }
    ]
    
    results_table = Table(title="Evaluation Results")
    results_table.add_column("Question", style="cyan")
    results_table.add_column("Model Answer", style="white")
    results_table.add_column("Accuracy", style="green")
    
    total_score = 0
    
    for item in eval_questions:
        # Use reasoning controller explicitly to just "ask" the system
        # Strategy: Hybrid to use both Memory (Vector) and Reasoning
        from agi_kernel.core.reasoning import ReasoningContext
        
        ctx = ReasoningContext(
            question=item["q"],
            available_memory_types=["semantic", "episodic"],
            has_vector=True,
            has_graph=True
        )
        
        # We assume hybrid strategy for best result
        from agi_kernel.core.reasoning import ReasoningStrategy
        response = await kernel.reasoning.execute(
            strategy=ReasoningStrategy.HYBRID,
            question=item["q"],
            context=ctx.__dict__,
            memory=kernel.memory,
            world=kernel.world
        )
        
        answer = str(response.get("answer", "No answer"))
        
        # Simple grading
        hits = sum(1 for k in item["keywords"] if k.lower() in answer.lower())
        score = hits / len(item["keywords"])
        total_score += score
        
        accuracy_str = f"{score:.0%}"
        if score > 0.6:
            accuracy_str = f"[bold green]{accuracy_str}[/bold green]"
        elif score > 0.3:
            accuracy_str = f"[yellow]{accuracy_str}[/yellow]"
        else:
            accuracy_str = f"[red]{accuracy_str}[/red]"
            
        results_table.add_row(item["q"], answer[:100]+"...", accuracy_str)

    console.print(results_table)
    
    # 4. Comprehensive Report
    console.print("\n" + "="*60)
    console.print("[bold]📊 COMPREHENSIVE LEARNING REPORT[/bold]")
    console.print("="*60)
    
    # Calculate Enhancements
    vec_diff = stats_after_learning['vector_points'] - stats_baseline['vector_points']
    node_diff = stats_after_learning['graph_nodes'] - stats_baseline['graph_nodes']
    rel_diff = stats_after_learning['graph_relations'] - stats_baseline['graph_relations']
    
    vec_pct = (vec_diff / stats_baseline['vector_points'] * 100) if stats_baseline['vector_points'] else 0
    rel_pct = (rel_diff / stats_baseline['graph_relations'] * 100) if stats_baseline['graph_relations'] else 0
    
    console.print(f"\n[bold underline]Database Enhancement (During Learning Loop)[/bold underline]")
    console.print(f"• [cyan]Qdrant (Vectors):[/cyan] {stats_baseline['vector_points']} -> {stats_after_learning['vector_points']} (+{vec_diff}, {vec_pct:.1f}%)")
    console.print(f"• [magenta]Neo4j (Nodes):[/magenta]   {stats_baseline['graph_nodes']} -> {stats_after_learning['graph_nodes']} (+{node_diff})")
    console.print(f"• [magenta]Neo4j (Relations):[/magenta] {stats_baseline['graph_relations']} -> {stats_after_learning['graph_relations']} (+{rel_diff}, {rel_pct:.1f}%)")
    
    console.print(f"\n[bold underline]Model Accuracy[/bold underline]")
    avg_accuracy = (total_score / len(eval_questions)) * 100
    console.print(f"• Final Score: [bold green]{avg_accuracy:.1f}%[/bold green]")
    
    console.print("\n[bold underline]Conclusion[/bold underline]")
    if rel_diff > 0:
        console.print("✅ [green]The model actively expanded its knowledge graph during the learning loop.[/green]")
    else:
        console.print("⚠️ [yellow]No new relations were discovered. The model might have focused on internal reasoning or failed to bridge gaps.[/yellow]")
        
    await kernel.close()

if __name__ == "__main__":
    asyncio.run(run_eval_test())
