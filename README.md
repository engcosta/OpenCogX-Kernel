# 🧠 Open AGI Kernel POC

> An open-source, self-evolving AGI kernel focused on **system-level intelligence** rather than model-scale intelligence.

## 🎯 Core Hypothesis

**General intelligence emerges when a system can:**
- Model the world
- Model itself
- Set goals
- Reason under uncertainty
- Learn from failure
- Modify its own cognitive strategies over time

## 🏗️ Architecture

The kernel is composed of **5 irreducible layers**:

```
┌─────────────────────────────────────────────────────┐
│                    META LAYER                        │
│         (Self-monitoring & Structural Adaptation)    │
├─────────────────────────────────────────────────────┤
│                  REASONING LAYER                     │
│            (Dynamic Strategy Selection)              │
├─────────────────────────────────────────────────────┤
│                    GOAL LAYER                        │
│        (Intrinsic Motivation & Uncertainty)          │
├─────────────────────────────────────────────────────┤
│                   MEMORY LAYER                       │
│       (Semantic, Episodic, Temporal Memory)          │
├─────────────────────────────────────────────────────┤
│                   WORLD LAYER                        │
│       (States, Events, Actions, Causality)           │
└─────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
agi_kernel/
├── core/
│   ├── world.py        # World Model
│   ├── memory.py       # Memory System
│   ├── goals.py        # Goal Engine
│   ├── reasoning.py    # Reasoning Controller
│   └── meta.py         # Meta-Cognition
├── plugins/
│   ├── llm/            # LLM adapters (Ollama, LM Studio)
│   ├── vector/         # Vector DB (Qdrant)
│   └── graph/          # Graph DB (Neo4j)
├── api/                # FastAPI endpoints
├── ingestion/          # Document ingestion
└── metrics/            # Evaluation & logging
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker (for Qdrant and Neo4j)
- Ollama or LM Studio with models

### Setup

1. **Start Infrastructure**
```bash
docker-compose up -d
```

2. **Install Dependencies**
```bash
pip install -e .
```

3. **Configure Environment**
```bash
cp .env.example .env
# Edit .env with your settings
```

4. **Run the Kernel**
```bash
python -m agi_kernel
```

## 🧪 POC Phases

### Phase 1: Ingestion
- Hierarchical chunking
- Vector storage (Qdrant)
- Graph extraction (Neo4j)

### Phase 2: Learning Loop
- Goal Generation (identify uncertainty/contradictions)
- Question Proposal (multi-hop reasoning)
- Answer Attempt (hybrid retrieval)
- Critic Evaluation
- Outcome Recording

### Phase 3: Evaluation
- Knowledge Coverage metrics
- Failure Rate over time
- Reasoning Strategy shifts
- Self-Correction events

## 📊 Metrics We Measure

| Metric | Description |
|--------|-------------|
| Entity Count | Number of entities in knowledge graph |
| Relation Count | Number of relations extracted |
| Multi-hop Relations | Complex reasoning paths discovered |
| Failure Rate | FAIL outcomes over time |
| Strategy Shifts | Changes in reasoning approach |
| Self-Corrections | Ontology/Strategy modifications |

## 🔴 Red Lines (Non-Negotiable)

- ❌ No hardcoded prompts in the kernel
- ❌ No hidden logic inside LLMs
- ❌ No decision without logged reason
- ❌ No learning without critic
- ❌ No feature without philosophical justification

## 📜 Philosophy

> "Intelligence is not a property of a model, but a result of component interactions."

Models are **tools, not minds**. Any model can be replaced. The kernel must remain.

## 📄 License

MIT License - Open source, transparent, forkable, extensible.
