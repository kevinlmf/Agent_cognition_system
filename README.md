# Agent Perception + Memory System

A graph-based cognitive infrastructure for intelligent agents with perception and memory capabilities.

## Overview

This project implements a cognitive engine for AI agents featuring:
- **Perception Layer**: Processes external data into structured world models (WorldGraph)
- **Memory Layer**: Maintains episodic (time-stamped) and semantic (abstract) memories (MemoryGraph)
- **Interfaces**: Supports LLM-based reasoning agents (LangGraph) and RL agents

## Architecture

```
External World -> [Perception Layer] -> [Memory Layer] -> [Cognitive Layer (LLM/RL)]
                   WorldGraph builder    Episodic/Semantic    Query & retrieval
```

## Installation & Quick Start

```bash
# Clone and install
cd Agent_Perception_Memory_System
pip install -r requirements.txt

# Run demos
python main.py

# Run trading agent example
python examples/trading_agent_example.py
```

### System Evaluation

Run comprehensive evaluation tests:

```bash
# Make the script executable (first time only)
chmod +x evaluation.sh

# Run full evaluation suite
./evaluation.sh
```

The evaluation script tests:
- Basic functional correctness
- Perception quality (entity extraction, processing time)
- Memory storage/retrieval performance
- WorldGraph structure quality
- LLM and RL agent integration
- Scalability under load (100+ operations)
- End-to-end trading scenario

Results are saved to `evaluation_results/` with detailed metrics in JSON format.

### Optional Dependencies
```bash
pip install spacy transformers sentence-transformers openai matplotlib plotly
```


## Project Structure

```
Agent_Perception_Memory_System/
├── perception/
│   ├── collectors/          # Web, text, API collectors
│   ├── preprocessors/       # Summarization, entity extraction
│   ├── world_model/         # WorldGraph, event tracking
│   └── perception_main.py
├── memory/
│   ├── episodic_memory.py   # Time-stamped experiences
│   ├── semantic_memory.py   # Abstract knowledge
│   ├── memory_graph.py      # MemoryGraph implementation
│   ├── retrieval.py         # Smart retrieval
│   └── memory_main.py
├── interface/
│   ├── langgraph_adapter.py # LLM interface
│   ├── rl_adapter.py        # RL interface
│   └── schema.py            # Data structures
├── examples/
│   └── trading_agent_example.py
└── main.py                  # Entry point
```

## Advanced Customization

### Custom Entity Extractor
```python
from perception.preprocessors.entity_extractor import EntityExtractor
import spacy

class SpacyEntityExtractor(EntityExtractor):
    def __init__(self):
        super().__init__()
        self.nlp = spacy.load("en_core_web_sm")

    def extract_entities(self, text):
        doc = self.nlp(text)
        return [{"text": ent.text, "type": ent.label_} for ent in doc.ents]
```

### Embeddings Integration
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("The Fed raised rates")

memory_id = memory_engine.store_experience(
    world_snapshot=snapshot,
    perception_result=perception,
    embedding=embedding
)
```

## Design Principles

1. **Graph-First**: World models and memories as graphs
2. **Modular**: Independent, composable components
3. **Scalable**: Memory pruning and graph consolidation
4. **Interface-Agnostic**: Works with LLMs, RL, or hybrid systems
5. **Consolidation**: Automatic pattern extraction

## License

MIT License

---

**Built with:** Python, NetworkX, NumPy
