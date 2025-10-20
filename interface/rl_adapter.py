"""
RL Adapter: Interface for RL agents (DQN, PPO, SAC, etc.)
Provides observation space, memory retrieval, and state encoding
"""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime

from .schema import Observation, Action, AgentState


class RLCognitiveInterface:
    """
    Interface for RL agents to access perception and memory
    Provides structured observation space
    """

    def __init__(self, perception_engine, memory_engine,
                 observation_dim: int = 128,
                 memory_context_size: int = 5):
        self.perception_engine = perception_engine
        self.memory_engine = memory_engine
        self.observation_dim = observation_dim
        self.memory_context_size = memory_context_size

    def get_observation(self, environment_state: Dict,
                       include_memory: bool = True) -> np.ndarray:
        """
        Get observation vector for RL agent
        Combines perception + memory into fixed-size vector
        """
        observation_components = []

        # 1. Environment state encoding (first half of observation)
        env_encoding = self._encode_environment(environment_state)
        observation_components.append(env_encoding)

        # 2. Memory context (second half of observation)
        if include_memory:
            memory_encoding = self._encode_memory_context()
            observation_components.append(memory_encoding)

        # Concatenate and ensure fixed size
        observation = np.concatenate(observation_components)

        # Pad or truncate to observation_dim
        if len(observation) < self.observation_dim:
            observation = np.pad(observation, (0, self.observation_dim - len(observation)))
        else:
            observation = observation[:self.observation_dim]

        return observation

    def _encode_environment(self, environment_state: Dict) -> np.ndarray:
        """
        Encode environment state to vector
        Override this for specific environments
        """
        # Default: extract numeric values
        values = []

        def extract_values(obj):
            if isinstance(obj, (int, float)):
                values.append(float(obj))
            elif isinstance(obj, dict):
                for v in obj.values():
                    extract_values(v)
            elif isinstance(obj, list):
                for item in obj:
                    extract_values(item)

        extract_values(environment_state)

        # Return as array, limit size
        return np.array(values[:self.observation_dim // 2] if values else [0.0])

    def _encode_memory_context(self) -> np.ndarray:
        """
        Encode memory context to vector
        Uses recent and important memories
        """
        # Get recent memories
        recent = self.memory_engine.retrieve(
            retrieval_strategy="recent",
            top_k=self.memory_context_size
        )

        # Encode memory features
        features = []

        for memory in recent:
            # Importance
            features.append(memory.get('importance', 0.5))

            # Recency (time since memory, normalized)
            timestamp = datetime.fromisoformat(memory['timestamp'])
            time_diff = (datetime.now() - timestamp).total_seconds()
            recency = np.exp(-time_diff / 3600)  # 1-hour decay
            features.append(recency)

        # Pad to fixed size
        expected_size = self.memory_context_size * 2
        if len(features) < expected_size:
            features.extend([0.0] * (expected_size - len(features)))

        return np.array(features[:expected_size])

    def step(self, environment_state: Dict,
            action: int,
            reward: float,
            done: bool,
            text_observation: str = None) -> Tuple[np.ndarray, Dict]:
        """
        RL step: process transition and return next observation
        """
        # Perceive text observation if provided
        perception = None
        if text_observation:
            perception = self.perception_engine.perceive_text(
                text_observation, source="rl_environment"
            )

        # Store experience
        memory_id = self.memory_engine.store_experience(
            world_snapshot=self.perception_engine.get_world_snapshot(),
            perception_result=perception or {"summary": "RL step"},
            action={"action": action},
            reward=reward
        )

        # Get next observation
        next_obs = self.get_observation(environment_state)

        # Get memory statistics
        stats = self.memory_engine.get_statistics()

        info = {
            "memory_id": memory_id,
            "perception": perception,
            "total_memories": stats['memory_graph']['total_memories']
        }

        return next_obs, info

    def get_state_encoding(self, include_world_graph: bool = True) -> np.ndarray:
        """
        Get rich state encoding including world graph structure
        For more sophisticated RL agents
        """
        encoding_parts = []

        # World graph statistics
        if include_world_graph:
            stats = self.perception_engine.world_graph.get_statistics()
            graph_features = [
                float(stats.get('num_entities', 0)) / 100.0,  # Normalized
                float(stats.get('num_relations', 0)) / 100.0,
                float(stats.get('density', 0))
            ]
            encoding_parts.append(np.array(graph_features))

        # Memory statistics
        mem_stats = self.memory_engine.get_statistics()
        memory_features = [
            float(mem_stats['memory_graph']['total_memories']) / 1000.0,
            float(mem_stats['memory_graph']['episodic_memories']) / 1000.0,
            float(mem_stats['memory_graph']['semantic_memories']) / 100.0
        ]
        encoding_parts.append(np.array(memory_features))

        # Concatenate
        encoding = np.concatenate(encoding_parts)

        return encoding

    def get_memory_augmented_value(self, base_value: float,
                                  state: Dict) -> float:
        """
        Augment value estimate with memory-based importance
        """
        # Get relevant memories
        relevant = self.memory_engine.retrieve(
            retrieval_strategy="important",
            threshold=0.7,
            top_k=5
        )

        if not relevant:
            return base_value

        # Compute importance-weighted adjustment
        avg_reward = np.mean([
            m.get('metadata', {}).get('reward', 0)
            for m in relevant
            if m.get('metadata', {}).get('reward') is not None
        ])

        # Adjust value based on historical rewards
        adjustment = avg_reward * 0.1  # Small adjustment factor

        return base_value + adjustment

    def consolidate_if_needed(self) -> bool:
        """
        Check if consolidation is needed and trigger it
        """
        stats = self.memory_engine.get_statistics()
        episodes_since = stats.get('episodes_since_consolidation', 0)

        if episodes_since >= 50:
            self.memory_engine.consolidate_memories()
            return True

        return False

    def get_statistics(self) -> Dict:
        """Get system statistics"""
        return {
            "perception": self.perception_engine.world_graph.get_statistics(),
            "memory": self.memory_engine.get_statistics(),
            "observation_dim": self.observation_dim,
            "memory_context_size": self.memory_context_size
        }


class RLMemoryReplayAdapter:
    """
    Adapter for using cognitive memory as experience replay
    """

    def __init__(self, memory_engine):
        self.memory_engine = memory_engine

    def sample_batch(self, batch_size: int = 32,
                    prioritized: bool = False) -> List[Dict]:
        """
        Sample batch of experiences from memory
        """
        if prioritized:
            # Sample based on importance
            memories = self.memory_engine.retrieve(
                retrieval_strategy="important",
                threshold=0.0,
                top_k=batch_size * 2
            )

            # Sample with probability proportional to importance
            importances = np.array([m.get('importance', 0.5) for m in memories])
            probs = importances / importances.sum()
            indices = np.random.choice(len(memories), size=min(batch_size, len(memories)), p=probs, replace=False)

            batch = [memories[i] for i in indices]
        else:
            # Uniform sampling from recent memories
            recent = self.memory_engine.retrieve(
                retrieval_strategy="recent",
                top_k=batch_size * 2
            )

            indices = np.random.choice(len(recent), size=min(batch_size, len(recent)), replace=False)
            batch = [recent[i] for i in indices]

        return batch

    def get_transitions(self, batch: List[Dict]) -> Tuple[np.ndarray, ...]:
        """
        Convert memory batch to RL transitions
        Returns (states, actions, rewards, next_states, dones)
        """
        states = []
        actions = []
        rewards = []
        # Note: next_states would require looking up subsequent memories

        for memory in batch:
            metadata = memory.get('metadata', {})

            # Extract state (would need proper encoding)
            states.append(memory.get('world_graph_snapshot', {}))

            # Extract action
            action = metadata.get('action', {})
            actions.append(action.get('action', 0))

            # Extract reward
            rewards.append(metadata.get('reward', 0.0))

        return (np.array(states), np.array(actions), np.array(rewards))
