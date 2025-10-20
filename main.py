"""
Main entry point for Agent Perception + Memory System
Demonstrates end-to-end workflow
"""
from perception.perception_main import PerceptionEngine
from memory.memory_main import MemoryEngine
from interface.langgraph_adapter import LangGraphCognitiveInterface
from interface.rl_adapter import RLCognitiveInterface
import numpy as np


class CognitiveAgent:
    """
    Complete cognitive agent with perception and memory
    """

    def __init__(self, mode: str = "langgraph", observation_dim: int = 128):
        """
        Initialize agent
        mode: 'langgraph' for LLM-based reasoning, 'rl' for reinforcement learning
        observation_dim: dimension of observation vector for RL mode
        """
        # Core engines
        self.perception_engine = PerceptionEngine()
        self.memory_engine = MemoryEngine()

        # Interface
        self.mode = mode
        if mode == "langgraph":
            self.interface = LangGraphCognitiveInterface(
                self.perception_engine, self.memory_engine
            )
        elif mode == "rl":
            self.interface = RLCognitiveInterface(
                self.perception_engine, self.memory_engine,
                observation_dim=observation_dim
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def perceive(self, observation: str, source: str = "environment"):
        """Process new observation"""
        if self.mode == "langgraph":
            return self.interface.observe(observation, source)
        elif self.mode == "rl":
            # For RL, need to convert to state dict
            state = {"observation_text": observation}
            obs_vector = self.interface.get_observation(state)
            return {"observation_vector": obs_vector}

    def get_state(self, query: str = None):
        """Get current cognitive state"""
        if self.mode == "langgraph":
            return self.interface.get_cognitive_state(query)
        elif self.mode == "rl":
            return self.interface.get_state_encoding()

    def query_memory(self, query: str):
        """Query memory"""
        if self.mode == "langgraph":
            return self.interface.query_memory(query)
        else:
            # RL mode
            memories = self.memory_engine.retrieve(
                query=query, retrieval_strategy="recent", top_k=5
            )
            return {"memories": memories}

    def query_world(self, entity: str):
        """Query world model"""
        return self.interface.query_world(entity)

    def record_action(self, action_type: str, parameters: dict,
                     result: str = None, reward: float = None):
        """Record action taken"""
        if self.mode == "langgraph":
            return self.interface.record_action(action_type, parameters, result, reward)
        elif self.mode == "rl":
            # Dummy state for RL
            state = {}
            action_id = 0  # Simplified
            obs, info = self.interface.step(state, action_id, reward or 0.0, False, result)
            return info['memory_id']

    def consolidate(self):
        """Consolidate memories"""
        return self.interface.consolidate()

    def get_statistics(self):
        """Get system statistics"""
        return self.interface.get_statistics()


def demo_langgraph_mode():
    """
    Demo: LangGraph mode (for LLM-based reasoning agents)
    """
    print("=" * 60)
    print("DEMO: LangGraph Mode (LLM Reasoning Agent)")
    print("=" * 60)

    agent = CognitiveAgent(mode="langgraph")

    # Step 1: Perceive observations
    print("\n1. Perceiving observations...")

    observations = [
        "The Federal Reserve announced a 0.25% interest rate increase today.",
        "Stock market experienced increased volatility following the Fed announcement.",
        "Treasury bond yields rose to 4.5% after the rate decision.",
        "Market analysts predict further rate increases in the coming months."
    ]

    for obs in observations:
        result = agent.perceive(obs, source="news")
        print(f"   Perceived: {obs[:50]}...")
        print(f"   Entities found: {len(result['perception']['entities'])}")

    # Step 2: Query world model
    print("\n2. Querying world model...")
    world_info = agent.query_world("Federal Reserve")
    print(f"   Entity: Federal Reserve")
    print(f"   Status: {world_info.get('status', world_info.keys())}")

    # Step 3: Retrieve relevant memories
    print("\n3. Retrieving memories about 'interest rates'...")
    memory_context = agent.query_memory("interest rates")
    print(f"   Recent memories: {len(memory_context.get('recent', []))}")
    print(f"   Important memories: {len(memory_context.get('important', []))}")

    # Step 4: Get cognitive state
    print("\n4. Getting cognitive state...")
    state = agent.get_state()
    summary = state.to_text_summary()
    print(summary[:500] + "...")

    # Step 5: Record action
    print("\n5. Recording agent action...")
    agent.record_action(
        action_type="analysis",
        parameters={"topic": "rate_impact"},
        result="Analysis shows rate increases typically lead to short-term volatility.",
        reward=0.8
    )
    print("   Action recorded with reward 0.8")

    # Step 6: Consolidate memories
    print("\n6. Consolidating memories...")
    consolidation = agent.consolidate()
    print(f"   New concepts: {consolidation['new_concepts']}")
    print(f"   New patterns: {consolidation['new_patterns']}")

    # Step 7: Statistics
    print("\n7. System statistics:")
    stats = agent.get_statistics()
    print(f"   Total memories: {stats['memory']['memory_graph']['total_memories']}")
    print(f"   World entities: {stats['perception']['num_entities']}")
    print(f"   Semantic concepts: {stats['memory']['semantic_memory']['total_concepts']}")

    print("\n" + "=" * 60)


def demo_rl_mode():
    """
    Demo: RL mode (for reinforcement learning agents)
    """
    print("=" * 60)
    print("DEMO: RL Mode (Reinforcement Learning Agent)")
    print("=" * 60)

    agent = CognitiveAgent(mode="rl")

    # Simulate RL episode
    print("\n1. Running RL episode...")

    episode_data = [
        ("Market opens, initial price: $100", {"price": 100, "volume": 1000}, 0, "observe"),
        ("Price increases to $102", {"price": 102, "volume": 1200}, 0.5, "buy"),
        ("Price reaches $105", {"price": 105, "volume": 1500}, 1.0, "hold"),
        ("Price drops to $103", {"price": 103, "volume": 1100}, -0.2, "sell"),
    ]

    for step, (obs_text, state, reward, action) in enumerate(episode_data):
        print(f"\n   Step {step}: {action.upper()}")

        # Get observation vector
        obs_vector = agent.interface.get_observation(state)
        print(f"   Observation dim: {len(obs_vector)}")

        # Record experience
        agent.record_action(
            action_type=action,
            parameters=state,
            result=obs_text,
            reward=reward
        )
        print(f"   Reward: {reward:+.2f}")

    # Get state encoding
    print("\n2. Getting state encoding...")
    state_encoding = agent.get_state()
    print(f"   State encoding shape: {state_encoding.shape}")

    # Memory-based value adjustment
    print("\n3. Memory-augmented value estimation...")
    base_value = 0.5
    adjusted_value = agent.interface.get_memory_augmented_value(
        base_value, {"price": 103}
    )
    print(f"   Base value: {base_value:.2f}")
    print(f"   Adjusted value: {adjusted_value:.2f}")

    # Statistics
    print("\n4. System statistics:")
    stats = agent.get_statistics()
    print(f"   Total memories: {stats['memory']['memory_graph']['total_memories']}")
    print(f"   Episodic memories: {stats['memory']['memory_graph']['episodic_memories']}")

    print("\n" + "=" * 60)


def demo_comprehensive():
    """
    Comprehensive demo showing full system capabilities
    """
    print("\n" + "=" * 60)
    print("COMPREHENSIVE SYSTEM DEMO")
    print("=" * 60)

    agent = CognitiveAgent(mode="langgraph")

    # Simulate a trading scenario with rich perception
    print("\n=== Scenario: Market Analysis Agent ===\n")

    # Day 1: Initial observations
    print("Day 1: Market opens")
    agent.perceive("Market opens at $100. Trading volume is normal.")

    # Day 2: News event
    print("\nDay 2: Major news")
    agent.perceive("Federal Reserve announces emergency rate cut. Market sentiment turns positive.")
    agent.record_action("analysis", {"focus": "rate_impact"}, "Positive impact expected", 0.7)

    # Day 3: Market reaction
    print("\nDay 3: Market reacts")
    agent.perceive("Stock prices surge 5% on rate cut news. Volume spikes.")
    agent.record_action("trading", {"action": "buy"}, "Entered long position", 1.0)

    # Day 4: Consolidation
    print("\nDay 4: Price consolidates")
    agent.perceive("Prices stabilize around $105. Market shows signs of consolidation.")

    # Query what we know
    print("\n--- Memory Query ---")
    context = agent.query_memory("Federal Reserve rate")
    print(f"Found {len(context.get('relevant', []))} relevant memories")

    # Check cognitive state
    print("\n--- Cognitive State ---")
    state = agent.get_state()
    print(f"Recent memories: {len(state.recent_memories)}")
    print(f"Key concepts: {len(state.key_concepts)}")

    # Consolidate knowledge
    print("\n--- Consolidation ---")
    result = agent.consolidate()
    print(f"Learned {result['new_concepts']} new concepts")
    print(f"Detected {result['new_patterns']} patterns")

    # Final statistics
    print("\n--- Final Statistics ---")
    stats = agent.get_statistics()
    mem_stats = stats['memory']['memory_graph']
    print(f"Total experiences: {mem_stats['total_memories']}")
    print(f"Episodic: {mem_stats['episodic_memories']}")
    print(f"Semantic: {mem_stats['semantic_memories']}")
    print(f"World entities: {stats['perception']['num_entities']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Run demos
    demo_langgraph_mode()
    print("\n\n")
    demo_rl_mode()
    print("\n\n")
    demo_comprehensive()

    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)
