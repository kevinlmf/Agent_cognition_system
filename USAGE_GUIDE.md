# Usage Guide

This guide provides practical examples for common use cases.

## Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [LangGraph Integration](#langgraph-integration)
4. [RL Agent Integration](#rl-agent-integration)
5. [Custom Extensions](#custom-extensions)
6. [Best Practices](#best-practices)

## Installation

```bash
# Clone and install
cd Agent_Perception_Memory_System
pip install -r requirements.txt

# Optional: Install advanced features
pip install spacy sentence-transformers
python -m spacy download en_core_web_sm
```

## Basic Usage

### 1. Simple Agent

```python
from main import CognitiveAgent

# Create agent
agent = CognitiveAgent(mode="langgraph")

# Perceive world
agent.perceive("The market opened at $100 today.")

# Query memory
memories = agent.query_memory("market")

# Get state
state = agent.get_state()
print(state.to_text_summary())
```

### 2. Continuous Learning Loop

```python
agent = CognitiveAgent(mode="langgraph")

# Observation loop
observations = [
    "Day 1: Market stable at $100",
    "Day 2: News breaks, sentiment positive",
    "Day 3: Price surges to $105",
    "Day 4: Consolidation around $104"
]

for i, obs in enumerate(observations):
    # Perceive
    agent.perceive(obs, source="market_feed")

    # Make decision (your logic)
    decision = make_decision(agent.get_state())

    # Record action
    agent.record_action(
        action_type="trade",
        parameters={"decision": decision},
        reward=calculate_reward(decision)
    )

    # Periodic consolidation
    if i % 10 == 0:
        agent.consolidate()
```

## LangGraph Integration

### Setup

```python
from langgraph.graph import StateGraph, END
from interface.langgraph_adapter import LangGraphCognitiveInterface
from perception.perception_main import PerceptionEngine
from memory.memory_main import MemoryEngine

# Initialize cognitive system
perception = PerceptionEngine()
memory = MemoryEngine()
cognitive = LangGraphCognitiveInterface(perception, memory)
```

### LangGraph Workflow

```python
from typing import TypedDict

class AgentState(TypedDict):
    input: str
    perception: dict
    memory_context: dict
    cognitive_state: dict
    response: str

def perceive_node(state: AgentState) -> AgentState:
    """Process observation"""
    result = cognitive.observe(state["input"])
    state["perception"] = result["perception"]
    return state

def retrieve_node(state: AgentState) -> AgentState:
    """Retrieve relevant memories"""
    context = cognitive.query_memory(state["input"])
    state["memory_context"] = context
    return state

def reason_node(state: AgentState) -> AgentState:
    """LLM reasoning with cognitive context"""
    cognitive_state = cognitive.get_cognitive_state()
    context_text = cognitive_state.to_text_summary()

    # Call LLM with context
    prompt = f"""
Context:
{context_text}

User Input: {state['input']}

Please provide a thoughtful response based on the context.
"""

    # response = llm.invoke(prompt)
    response = "LLM response here"  # Placeholder

    state["response"] = response
    return state

def act_node(state: AgentState) -> AgentState:
    """Record the action"""
    cognitive.record_action(
        action_type="response",
        parameters={"input": state["input"]},
        result=state["response"],
        reward=1.0  # Could be from feedback
    )
    return state

# Build graph
workflow = StateGraph(AgentState)

workflow.add_node("perceive", perceive_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("reason", reason_node)
workflow.add_node("act", act_node)

workflow.set_entry_point("perceive")
workflow.add_edge("perceive", "retrieve")
workflow.add_edge("retrieve", "reason")
workflow.add_edge("reason", "act")
workflow.add_edge("act", END)

app = workflow.compile()

# Run
result = app.invoke({"input": "What's happening with interest rates?"})
print(result["response"])
```

## RL Agent Integration

### Gym Environment

```python
import gym
import numpy as np
from interface.rl_adapter import RLCognitiveInterface

class TradingEnvWithMemory(gym.Env):
    def __init__(self):
        super().__init__()
        self.cognitive = RLCognitiveInterface(
            perception_engine=PerceptionEngine(),
            memory_engine=MemoryEngine(),
            observation_dim=128
        )

        self.action_space = gym.spaces.Discrete(3)  # buy, hold, sell
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(128,), dtype=np.float32
        )

    def reset(self):
        self.state = self._get_market_state()
        obs = self.cognitive.get_observation(self.state)
        return obs

    def step(self, action):
        # Execute action
        next_state = self._execute_action(action)
        reward = self._calculate_reward(action, next_state)
        done = self._is_done()

        # Record in cognitive memory
        text_obs = f"Action {action}, reward {reward:.2f}"
        obs, info = self.cognitive.step(
            next_state, action, reward, done, text_obs
        )

        self.state = next_state
        return obs, reward, done, info

    def _get_market_state(self):
        return {"price": 100.0, "volume": 1000}

    def _execute_action(self, action):
        # Your trading logic
        return {"price": 101.0, "volume": 1100}

    def _calculate_reward(self, action, state):
        # Your reward function
        return 0.5

    def _is_done(self):
        return False
```

### Training Loop

```python
from stable_baselines3 import PPO

# Create environment
env = TradingEnvWithMemory()

# Train agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# During training, cognitive system automatically:
# - Stores experiences in episodic memory
# - Consolidates patterns into semantic memory
# - Provides memory-augmented observations
```

### Memory-Augmented DQN

```python
import torch
import torch.nn as nn

class MemoryAugmentedDQN(nn.Module):
    def __init__(self, obs_dim, action_dim, memory_dim=64):
        super().__init__()

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim - memory_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Memory encoder
        self.memory_encoder = nn.Sequential(
            nn.Linear(memory_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # Combined Q-network
        self.q_network = nn.Sequential(
            nn.Linear(64 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, obs):
        # Split observation into env and memory parts
        env_obs = obs[:, :-64]
        memory_obs = obs[:, -64:]

        # Encode separately
        env_features = self.obs_encoder(env_obs)
        memory_features = self.memory_encoder(memory_obs)

        # Combine
        combined = torch.cat([env_features, memory_features], dim=1)
        q_values = self.q_network(combined)

        return q_values

# Use in training
cognitive = RLCognitiveInterface(perception, memory, observation_dim=128)
dqn = MemoryAugmentedDQN(obs_dim=128, action_dim=3)

# Training loop uses cognitive.get_observation()
```

## Custom Extensions

### 1. Custom Data Collector

```python
from perception.collectors.web_collector import WebCollector
import yfinance as yf

class YahooFinanceCollector(WebCollector):
    def fetch_stock_data(self, symbol, period="1d"):
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)

        return {
            "symbol": symbol,
            "data": hist.to_dict(),
            "info": ticker.info,
            "timestamp": datetime.now().isoformat()
        }

# Use it
collector = YahooFinanceCollector()
data = collector.fetch_stock_data("AAPL")
```

### 2. Custom Entity Extractor with spaCy

```python
from perception.preprocessors.entity_extractor import EntityExtractor
import spacy

class SpacyEntityExtractor(EntityExtractor):
    def __init__(self, model="en_core_web_sm"):
        super().__init__()
        self.nlp = spacy.load(model)

    def extract_entities(self, text):
        doc = self.nlp(text)

        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })

        return entities

    def extract_relations(self, text, entities=None):
        doc = self.nlp(text)

        relations = []
        for token in doc:
            if token.dep_ in ["nsubj", "dobj", "pobj"]:
                head = token.head.text
                rel = token.dep_
                obj = token.text
                relations.append((head, rel, obj))

        return relations

# Integrate
engine = PerceptionEngine()
engine.entity_extractor = SpacyEntityExtractor()
```

### 3. Embeddings for Better Retrieval

```python
from sentence_transformers import SentenceTransformer

class EmbeddingMemoryEngine:
    def __init__(self):
        self.memory_engine = MemoryEngine()
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def store_with_embedding(self, world_snapshot, perception, **kwargs):
        # Generate embedding
        text = perception.get('summary', perception.get('content', ''))
        embedding = self.embedder.encode(text)

        # Store
        return self.memory_engine.store_experience(
            world_snapshot, perception,
            embedding=embedding,
            **kwargs
        )

    def retrieve_similar(self, query_text, top_k=10):
        query_embedding = self.embedder.encode(query_text)

        return self.memory_engine.retrieve(
            query_embedding=query_embedding,
            retrieval_strategy="similar",
            top_k=top_k
        )

# Use
memory = EmbeddingMemoryEngine()
memory.store_with_embedding(snapshot, perception)
similar = memory.retrieve_similar("interest rate policy")
```

## Best Practices

### 1. Memory Management

```python
# Periodic pruning
if len(agent.memory_engine.memory_graph.memories) > 5000:
    agent.memory_engine.prune_memories(
        importance_threshold=0.3,
        keep_recent=1000
    )

# Periodic consolidation
if episode_count % 50 == 0:
    agent.consolidate()
```

### 2. Observation Quality

```python
# Good: Informative observations
agent.perceive(
    "Federal Reserve raised rates by 0.25% citing inflation concerns. "
    "Market reacted with 2% decline."
)

# Bad: Too vague
agent.perceive("Something happened")
```

### 3. Importance Scoring

```python
# Let the system auto-calculate importance
memory_id = memory_engine.store_experience(
    world_snapshot, perception,
    reward=0.8  # System uses this for importance
)

# Or set manually for critical events
memory_id = memory_engine.memory_graph.add_memory(
    memory_type="episodic",
    content="Critical event occurred",
    summary="Critical",
    importance=1.0  # Maximum importance
)
```

### 4. Retrieval Strategy Selection

```python
# For recent context
recent = agent.query_memory(retrieval_strategy="recent")

# For important facts
important = agent.query_memory(retrieval_strategy="important")

# For semantic search (with embeddings)
similar = agent.query_memory(
    query_embedding=embedding,
    retrieval_strategy="similar"
)

# For best overall (recommended)
relevant = agent.query_memory(
    query_embedding=embedding,
    retrieval_strategy="hybrid"
)
```

### 5. State Management

```python
# Always get fresh state before important decisions
state = agent.get_state(query="specific query")
context = state.to_text_summary()

# Use context in decision making
decision = your_decision_function(context)

# Record the action
agent.record_action(
    action_type="decision",
    parameters={"decision": decision},
    result="Executed successfully",
    reward=feedback_score
)
```

## Troubleshooting

### Issue: Too many memories
**Solution**: Increase pruning frequency or lower importance threshold

```python
agent.memory_engine.prune_memories(threshold=0.4, keep_recent=500)
```

### Issue: Poor retrieval quality
**Solution**: Add embeddings for semantic search

```python
# Use sentence transformers for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')
```

### Issue: WorldGraph too large
**Solution**: Consolidate duplicate entities

```python
# Merge similar entities manually
world_graph.merge_graph(other_graph, strategy="weighted")
```

### Issue: Slow consolidation
**Solution**: Reduce consolidation frequency or batch size

```python
memory_engine.consolidation_threshold = 100  # Instead of 50
```

## Advanced Patterns

### Pattern 1: Multi-Source Perception

```python
# Collect from multiple sources
sources = {
    "news": news_collector.fetch_news("market"),
    "data": market_collector.fetch_market_status("SPY"),
    "social": twitter_collector.fetch_tweets("$SPY")
}

for source_name, data in sources.items():
    agent.perceive(str(data), source=source_name)
```

### Pattern 2: Hierarchical Memory

```python
# High-level semantic concepts
agent.memory_engine.semantic_memory.add_concept(
    name="Bull Market Pattern",
    concept_type="pattern",
    properties={"indicators": ["rising_prices", "high_volume"]}
)

# Link to episodes
for episode in relevant_episodes:
    concept.add_example(episode['episode_id'])
```

### Pattern 3: Active Forgetting

```python
# Implement decay-based forgetting
def decay_importance(memory_id, decay_rate=0.99):
    memory = agent.memory_engine.memory_graph.memories[memory_id]
    memory.importance *= decay_rate

# Apply periodically
for memory_id in list(agent.memory_engine.memory_graph.memories.keys()):
    decay_importance(memory_id)
```

---

For more examples, see:
- `main.py` - Complete demos
- `examples/trading_agent_example.py` - Trading agent
- `Quick_Start.ipynb` - Interactive tutorial
