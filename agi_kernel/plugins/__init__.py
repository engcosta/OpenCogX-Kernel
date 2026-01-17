"""
Plugins Package
===============

Plugins are external tools that the kernel can use.
They are NOT part of the core kernel.

Available plugins:
- llm/: LLM adapters (Ollama, LM Studio)
- vector/: Vector database (Qdrant)
- graph/: Graph database (Neo4j)
"""

from agi_kernel.plugins.llm import LLMPlugin
from agi_kernel.plugins.vector import VectorPlugin
from agi_kernel.plugins.graph import GraphPlugin

__all__ = ["LLMPlugin", "VectorPlugin", "GraphPlugin"]
