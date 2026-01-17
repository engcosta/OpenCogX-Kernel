"""
🤖 LLM Plugin
=============

Adapter for Language Models (Ollama, LM Studio).

This is a PLUGIN, not part of the kernel.
The kernel can function without any specific LLM.

Supported backends:
- Ollama (local models)
- LM Studio (local models)
- OpenAI-compatible APIs
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional
import httpx
import structlog

logger = structlog.get_logger()


class ModelType(Enum):
    """Types of models for different tasks."""
    REASONING = "reasoning"   # Complex reasoning (qwen3:8b)
    SOLVER = "solver"         # Answer generation (llama3.2:3b)
    CRITIC = "critic"         # Verification (llama3.2:3b)
    EMBEDDING = "embedding"   # Embeddings (qwen3-embedding:4b)


class LLMPlugin:
    """
    LLM Plugin for interacting with language models.
    
    Supports multiple backends:
    - Ollama (default: http://localhost:11434)
    - LM Studio (default: http://localhost:1234)
    
    Model types:
    - reasoning: For complex reasoning tasks
    - solver: For generating answers
    - critic: For verification and criticism
    - embedding: For generating embeddings
    """
    
    def __init__(
        self,
        ollama_url: Optional[str] = None,
        lm_studio_url: Optional[str] = None,
        reasoning_model: str = "qwen3:8b",
        solver_model: str = "llama3.2:3b",
        critic_model: str = "llama3.2:3b",
        embedding_model: str = "qwen3-embedding:4b",
        timeout: float = 120.0,
    ):
        """
        Initialize the LLM Plugin.
        
        Args:
            ollama_url: Ollama base URL
            lm_studio_url: LM Studio base URL
            reasoning_model: Model for reasoning tasks
            solver_model: Model for answer generation
            critic_model: Model for verification
            embedding_model: Model for embeddings
            timeout: Request timeout in seconds
        """
        self.ollama_url = ollama_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.lm_studio_url = lm_studio_url or os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")
        
        self.models = {
            ModelType.REASONING: reasoning_model,
            ModelType.SOLVER: solver_model,
            ModelType.CRITIC: critic_model,
            ModelType.EMBEDDING: embedding_model,
        }
        
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
        
        # Stats tracking
        self.call_count = 0
        self.token_count = 0
        
        logger.info(
            "llm_plugin_initialized",
            ollama_url=self.ollama_url,
            models=self.models,
        )
    
    async def generate(
        self,
        prompt: str,
        model_type: str = "solver",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        use_lm_studio: bool = False,
    ) -> str:
        """
        Generate text using a language model.
        
        Args:
            prompt: The prompt to generate from
            model_type: Type of model to use (reasoning, solver, critic)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            use_lm_studio: Use LM Studio instead of Ollama
            
        Returns:
            Generated text
        """
        model_enum = ModelType(model_type)
        model = self.models.get(model_enum, self.models[ModelType.SOLVER])
        
        if use_lm_studio:
            return await self._generate_lm_studio(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )
        else:
            return await self._generate_ollama(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )
    
    async def _generate_ollama(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
    ) -> str:
        """Generate using Ollama API."""
        url = f"{self.ollama_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            self.call_count += 1
            
            result = data.get("response", "")
            
            logger.debug(
                "ollama_generation_complete",
                model=model,
                prompt_length=len(prompt),
                response_length=len(result),
            )
            
            return result
            
        except httpx.HTTPError as e:
            logger.error("ollama_request_failed", error=str(e))
            raise
    
    async def _generate_lm_studio(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
    ) -> str:
        """Generate using LM Studio OpenAI-compatible API."""
        url = f"{self.lm_studio_url}/v1/chat/completions"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        
        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            self.call_count += 1
            
            result = data["choices"][0]["message"]["content"]
            
            logger.debug(
                "lm_studio_generation_complete",
                model=model,
                response_length=len(result),
            )
            
            return result
            
        except httpx.HTTPError as e:
            logger.error("lm_studio_request_failed", error=str(e))
            raise
    
    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> list[float]:
        """
        Generate embeddings for text.
        
        Args:
            text: Text to embed
            model: Override for embedding model
            
        Returns:
            Embedding vector
        """
        model = model or self.models[ModelType.EMBEDDING]
        url = f"{self.ollama_url}/api/embed"
        
        payload = {
            "model": model,
            "input": text,
        }
        
        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            embeddings_list = data.get("embeddings", [])
            if not embeddings_list:
                logger.warning("no_embeddings_returned", model=model, text=text[:50])
                return []
                
            embeddings = embeddings_list[0]
            
            logger.debug(
                "embedding_generated",
                model=model,
                text_length=len(text),
                embedding_dim=len(embeddings),
            )
            
            return embeddings
            
        except httpx.HTTPError as e:
            logger.error("embedding_request_failed", error=str(e))
            raise
    
    async def critique(
        self,
        question: str,
        answer: str,
        context: Optional[str] = None,
    ) -> dict:
        """
        Critique an answer.
        
        Args:
            question: Original question
            answer: Generated answer
            context: Optional context used
            
        Returns:
            Dict with verdict, issues, and confidence
        """
        critique_prompt = f"""Evaluate this answer critically.

Question: {question}

Answer: {answer}

{f'Context: {context}' if context else ''}

Respond in this exact format:
VERDICT: [PASS/FAIL/PARTIAL]
CONFIDENCE: [0.0-1.0]
ISSUES: [List any issues, or "None"]
REASONING: [Brief explanation]
"""
        
        response = await self.generate(
            prompt=critique_prompt,
            model_type="critic",
            temperature=0.3,  # Lower temp for more consistent criticism
        )
        
        # Parse the response
        verdict = "FAIL"
        confidence = 0.5
        issues = []
        reasoning = ""
        
        for line in response.split("\n"):
            if line.startswith("VERDICT:"):
                verdict = line.split(":")[1].strip().upper()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":")[1].strip())
                except ValueError:
                    confidence = 0.5
            elif line.startswith("ISSUES:"):
                issues_text = line.split(":")[1].strip()
                if issues_text.lower() != "none":
                    issues = [issues_text]
            elif line.startswith("REASONING:"):
                reasoning = line.split(":")[1].strip()
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "issues": issues,
            "reasoning": reasoning,
        }
    
    async def check_health(self) -> dict:
        """Check if LLM services are available."""
        health = {"ollama": False, "lm_studio": False}
        
        try:
            response = await self._client.get(f"{self.ollama_url}/api/tags")
            health["ollama"] = response.status_code == 200
        except:
            pass
        
        try:
            response = await self._client.get(f"{self.lm_studio_url}/v1/models")
            health["lm_studio"] = response.status_code == 200
        except:
            pass
        
        return health
    
    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "call_count": self.call_count,
            "token_count": self.token_count,
            "models": {mt.value: m for mt, m in self.models.items()},
        }
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
