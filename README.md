# Agent Perception + Memory System

A graph-based cognitive infrastructure for intelligent agents with **universal memory mechanisms** that work across diverse environments: Industrial systems, Healthcare, and Competitive games.





## Why Memory Matters

Memory enables agents to:
- **Learn continuously** from experience without catastrophic forgetting
- **Maintain long-term context** across interactions
- **Personalize** strategies for individual users/opponents
- **Model the world** dynamically, tracking entities and relationships
- **Transfer knowledge** through semantic consolidation

## Core Methods: Universal Across Environments

## Architecture

```
External World → [Perception] → [Memory Layer] → [Optimization] → Action
                WorldGraph      Episodic/Semantic   Bayesian/Graph
```

Our system employs **three core methods** that demonstrate effectiveness across multiple domains:

### 1. **Bayesian Estimation** (State/Hidden Variable Inference)

**Mathematical Foundation:**
```
b_{t+1} = P(σ | a_1:t, a_{t+1})           # Belief update
P(cards | a_1:t) ∝ P(a_t | cards) * P(cards | a_1:t-1)  # Range estimation
P(latent | obs) ∝ P(obs | latent) * P(latent)            # Latent state
```

### 2. **Graph-Based Modeling** (World Model & Memory Structure)

**Structure:**
- **Episodic Memory**: Time-stamped experiences → nodes, temporal relations → edges
- **Semantic Memory**: Concepts → nodes, semantic relations → edges  
- **World Graph**: Entities → nodes, relations → edges


### 3. **Hybrid Optimization** (Exploitation + Exploration)

**Mathematical Foundation:**
```
EV(a) = (1-α)·EV_exploitative + α·EV_GTO      # Weighted combination
a* = argmax_a E_{s ~ b_t} [R(a, s)]          # Best response
```
## Why These Methods Work Universally

1. **Bayesian Estimation**: Provides principled uncertainty quantification, enabling adaptive learning in any domain with hidden states
2. **Graph Structure**: Captures relationships naturally, applicable to any domain with entities and connections
3. **Hybrid Optimization**: Balances exploitation (current best) with exploration (robust baseline), effective across competitive and cooperative settings



## Experimental Validation: Cross-Domain Effectiveness

Our methods have been validated across **three diverse scenarios**, demonstrating **universal applicability**:

| Scenario | Core Task | Our Method | Baseline Comparison | Improvement |
|----------|-----------|------------|---------------------|-------------|
| **Industrial** | System stability & robustness | Graph-based world modeling + Bayesian state estimation | LSTM/Transformer/Memory Networks | **+0.944** stability, **+0.879** robustness |
| **Health** | Latent state estimation & behavior prediction | Bayesian latent state inference + Graph-based pattern learning | LSTM/Transformer/Episodic Memory | **+0.6** prediction, **+0.538** state estimation |
| **Poker** | Opponent modeling & strategy optimization | Bayesian range estimation + Best response optimization | LSTM/Transformer/Memory Networks | **+0.475** hidden state prediction, **+0.3** consistency |



## Quick Start

```bash
# Clone the repository
git clone https://github.com/kevinlmf/Agent_Memory
cd Agent_Memory

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run evaluations (automatically includes baseline comparisons)
python evaluation/evaluate_industrial.py
python evaluation/evaluate_health.py
python evaluation/evaluate_poker.py
```


## Project Structure

```
Agent_Memory/
├── memory/
│   ├── memory_graph.py      # Graph-based storage (universal)
│   ├── episodic_memory.py  # Time-stamped experiences
│   ├── semantic_memory.py  # Abstract knowledge
│   ├── retrieval.py        # Multi-strategy retrieval
│   ├── opponent_model.py   # Bayesian modeling (Poker/Health/Industrial)
│   ├── range_estimator.py  # Bayesian estimation (universal)
│   └── best_response.py    # Optimization (universal)
├── evaluation/
│   ├── evaluate_industrial.py
│   ├── evaluate_health.py
│   └── evaluate_poker.py
└── examples/
```



---
Trying my best to build memories that last and bring light✨
