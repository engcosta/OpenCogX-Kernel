"""
🧬 AGI Kernel POC - Main Entry Point
=====================================

Usage:
    python -m agi_kernel [command] [options]

Commands:
    serve       Start the API server
    ingest      Ingest documents
    learn       Run learning loop
    evaluate    Generate evaluation report
    demo        Run a demo with sample data

Examples:
    python -m agi_kernel serve --port 8000
    python -m agi_kernel ingest ./data/corpus
    python -m agi_kernel learn --iterations 20
    python -m agi_kernel evaluate
    python -m agi_kernel demo
"""

import argparse
import asyncio
import json
import sys

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


def print_banner():
    """Print the kernel banner."""
    banner = """
    ╔══════════════════════════════════════════════════════╗
    ║                                                      ║
    ║   🧠 AGI KERNEL POC v0.1.0                          ║
    ║   An open-source, self-evolving cognitive system    ║
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, border_style="cyan"))


def cmd_serve(args):
    """Start the API server."""
    import uvicorn
    
    console.print(f"[green]Starting API server on {args.host}:{args.port}[/green]")
    console.print(f"[cyan]API docs available at: http://{args.host}:{args.port}/docs[/cyan]")
    
    uvicorn.run(
        "agi_kernel.api:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


async def cmd_ingest(args):
    """Ingest documents."""
    from agi_kernel.kernel import Kernel
    import os
    
    kernel = Kernel()
    await kernel.initialize_plugins()
    
    try:
        is_dir = os.path.isdir(args.path)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Ingesting {args.path}...",
                total=None,
            )
            
            result = await kernel.ingest(args.path, is_directory=is_dir)
        
        # Display results
        table = Table(title="Ingestion Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        if "files_processed" in result:
            table.add_row("Files Processed", str(result["files_processed"]))
        table.add_row("Chunks", str(result.get("chunks") or result.get("total_chunks", 0)))
        table.add_row("Entities", str(result.get("entities") or result.get("total_entities", 0)))
        table.add_row("Relations", str(result.get("relations") or result.get("total_relations", 0)))
        
        console.print(table)
        
    finally:
        await kernel.close()


async def cmd_learn(args):
    """Run the learning loop."""
    from agi_kernel.kernel import Kernel
    
    kernel = Kernel()
    await kernel.initialize_plugins()
    
    try:
        console.print(f"[yellow]Running {args.iterations} learning iterations...[/yellow]")
        
        result = await kernel.learn(
            iterations=args.iterations,
            interval_seconds=args.interval,
        )
        
        # Display results
        table = Table(title="Learning Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Iterations", str(result.get("total_iterations", 0)))
        table.add_row("Passed", str(result.get("passed", 0)))
        table.add_row("Failed", str(result.get("failed", 0)))
        table.add_row("Pass Rate", f"{result.get('pass_rate', 0):.1%}")
        table.add_row("Avg Confidence", f"{result.get('average_confidence', 0):.2f}")
        
        console.print(table)
        
        # Show strategy usage
        strategies = result.get("strategies_used", {})
        if strategies:
            strat_table = Table(title="Strategy Usage")
            strat_table.add_column("Strategy", style="cyan")
            strat_table.add_column("Count", style="green")
            strat_table.add_column("Passed", style="green")
            
            for strategy, stats in strategies.items():
                strat_table.add_row(
                    strategy,
                    str(stats.get("count", 0)),
                    str(stats.get("passed", 0)),
                )
            
            console.print(strat_table)
        
    finally:
        await kernel.close()


async def cmd_evaluate(args):
    """Generate evaluation report."""
    from agi_kernel.kernel import Kernel
    
    kernel = Kernel()
    await kernel.initialize_plugins()
    
    try:
        report = kernel.evaluate()
        
        # Display summary
        summary = report.get("summary", {})
        
        is_improving = summary.get("is_system_improving", False)
        status_color = "green" if is_improving else "yellow"
        status_icon = "✅" if is_improving else "⚠️"
        
        console.print(Panel(
            f"{status_icon} System is {'improving' if is_improving else 'stable/unclear'}",
            border_style=status_color,
        ))
        
        # Improvement reasons
        if summary.get("improvement_reasons"):
            console.print("\n[green]Improvement Signals:[/green]")
            for reason in summary["improvement_reasons"]:
                console.print(f"  • {reason}")
        
        # Concerns
        if summary.get("concerns"):
            console.print("\n[yellow]Concerns:[/yellow]")
            for concern in summary["concerns"]:
                console.print(f"  • {concern}")
        
        # Detailed metrics
        console.print("\n[cyan]Detailed Metrics:[/cyan]")
        console.print(json.dumps(report, indent=2))
        
        # Save report
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            console.print(f"\n[green]Report saved to {args.output}[/green]")
        
    finally:
        await kernel.close()


async def cmd_demo(args):
    """Run a demo with sample data."""
    from agi_kernel.kernel import Kernel
    import tempfile
    import os
    
    console.print("[yellow]Running AGI Kernel Demo...[/yellow]\n")
    
    # Create sample corpus
    sample_docs = [
        """# Distributed Systems Fundamentals
        
A distributed system is a collection of autonomous computing elements that appears to its users as a single coherent system.

## Key Concepts

### CAP Theorem
The CAP theorem states that a distributed system can only provide two of three guarantees:
- Consistency: All nodes see the same data at the same time
- Availability: Every request receives a response
- Partition Tolerance: The system continues to operate despite network partitions

### Consensus
Consensus is the process of agreeing on values among a group of processes. The Paxos and Raft algorithms are common solutions.

### Replication
Replication ensures data is copied across multiple nodes for fault tolerance. Types include:
- Synchronous replication
- Asynchronous replication
- Semi-synchronous replication
""",
        """# Fault Tolerance in Distributed Systems

Fault tolerance is the ability of a system to continue operating properly in the event of failures.

## Types of Faults
- Crash faults: A node stops working
- Byzantine faults: A node behaves arbitrarily, including maliciously
- Omission faults: A node fails to send or receive messages

## Recovery Strategies
1. Checkpointing: Periodically saving system state
2. Logging: Recording operations for replay
3. Redundancy: Maintaining backup components

## Relationship to Consensus
Fault tolerance often relies on consensus protocols to ensure all nodes agree on the system state despite failures.
""",
        """# Database Consistency Models

Consistency models define the rules for how updates propagate through a distributed database.

## Strong Consistency
All reads reflect the most recent write. Requires coordination, impacting availability.

## Eventual Consistency
Updates will eventually be visible to all nodes. Used in high-availability systems.

## Causal Consistency
Related operations appear in the same order to all nodes. A middle ground between strong and eventual.

## Real-world Examples
- Strong: Traditional SQL databases with ACID
- Eventual: Amazon DynamoDB, Cassandra
- Causal: COPS (Clusters of Order-Preserving Servers)
"""
    ]
    
    kernel = Kernel()
    await kernel.initialize_plugins()
    
    try:
        # Create temp directory with sample docs
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, doc in enumerate(sample_docs):
                filepath = os.path.join(tmpdir, f"doc_{i}.md")
                with open(filepath, 'w') as f:
                    f.write(doc)
            
            # Ingest
            console.print("[cyan]Phase 1: Ingesting sample corpus...[/cyan]")
            result = await kernel.ingest(tmpdir, is_directory=True)
            console.print(f"  Ingested {result.get('files_processed', 0)} files, "
                         f"{result.get('total_chunks', 0)} chunks, "
                         f"{result.get('total_entities', 0)} entities")
            
            console.print("\n[cyan]Phase 2: Running learning loop (5 iterations)...[/cyan]")
            result = await kernel.learn(iterations=5, interval_seconds=1.0)
            console.print(f"  Completed {result.get('total_iterations', 0)} iterations")
            console.print(f"  Pass rate: {result.get('pass_rate', 0):.1%}")
            
            console.print("\n[cyan]Phase 3: Evaluating...[/cyan]")
            report = kernel.evaluate()
            
            conclusion = report.get("conclusion", "No conclusion available")
            console.print(Panel(conclusion, title="Evaluation Result"))
            
            # Show self-knowledge
            self_knowledge = kernel.meta.get_self_knowledge()
            if self_knowledge.get("blind_spots"):
                console.print(f"\n[yellow]Blind spots detected:[/yellow] {len(self_knowledge['blind_spots'])}")
            if self_knowledge.get("detected_biases"):
                console.print(f"[yellow]Biases detected:[/yellow] {len(self_knowledge['detected_biases'])}")
        
        console.print("\n[green]Demo complete![/green]")
        
    finally:
        await kernel.close()


async def cmd_status(args):
    """Show kernel status."""
    from agi_kernel.kernel import Kernel
    
    kernel = Kernel()
    await kernel.initialize_plugins()
    
    try:
        status = kernel.get_status()
        
        # Display as tables
        for component, stats in status.items():
            if stats:
                table = Table(title=f"{component.title()} Status")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")
                
                for key, value in stats.items():
                    if isinstance(value, dict):
                        table.add_row(key, json.dumps(value, indent=2))
                    else:
                        table.add_row(key, str(value))
                
                console.print(table)
                console.print()
        
    finally:
        await kernel.close()


def main():
    """Main entry point."""
    print_banner()
    
    parser = argparse.ArgumentParser(
        description="AGI Kernel POC - A self-evolving cognitive system",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    
    # ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("path", help="Path to file or directory")
    
    # learn command
    learn_parser = subparsers.add_parser("learn", help="Run learning loop")
    learn_parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    learn_parser.add_argument("--interval", type=float, default=5.0, help="Seconds between iterations")
    
    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Generate evaluation report")
    eval_parser.add_argument("--output", help="Output file for report")
    
    # status command
    status_parser = subparsers.add_parser("status", help="Show kernel status")
    
    # demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo with sample data")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # Run the appropriate command
    if args.command == "serve":
        cmd_serve(args)  # Sync - uvicorn handles its own event loop
    elif args.command == "ingest":
        asyncio.run(cmd_ingest(args))
    elif args.command == "learn":
        asyncio.run(cmd_learn(args))
    elif args.command == "evaluate":
        asyncio.run(cmd_evaluate(args))
    elif args.command == "status":
        asyncio.run(cmd_status(args))
    elif args.command == "demo":
        asyncio.run(cmd_demo(args))


if __name__ == "__main__":
    main()
