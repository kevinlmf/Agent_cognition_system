"""
LangGraph Adapter: Interface for LangGraph-based reasoning agents
Provides perception and memory access for LLM-based agents
"""
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import numpy as np

from .schema import Observation, Action, AgentState, PerceptionMemoryState


class LangGraphCognitiveInterface:
    """
    Interface for LangGraph agents to access perception and memory
    """

    def __init__(self, perception_engine, memory_engine):
        self.perception_engine = perception_engine
        self.memory_engine = memory_engine

    def observe(self, observation_text: str, source: str = "environment") -> Dict[str, Any]:
        """
        Process new observation
        Returns perception result + memory context
        """
        # Perceive the observation
        perception = self.perception_engine.perceive_text(observation_text, source)

        # Store in memory (with dummy embedding - could be from actual embedding model)
        memory_id = self.memory_engine.store_experience(
            world_snapshot=self.perception_engine.get_world_snapshot(),
            perception_result=perception,
            action=None,
            reward=None
        )

        return {
            "perception": perception,
            "memory_id": memory_id,
            "timestamp": datetime.now().isoformat()
        }

    def get_cognitive_state(self, query: str = None,
                          query_embedding: Optional[np.ndarray] = None) -> PerceptionMemoryState:
        """
        Get complete cognitive state for agent reasoning
        """
        # Get world state
        world_snapshot = self.perception_engine.get_world_snapshot()

        # Get current perception (most recent)
        recent_perceptions = self.perception_engine.event_tracker.get_recent_events(1)
        current_perception = recent_perceptions[0].to_dict() if recent_perceptions else {}

        # Retrieve memories
        recent_memories = self.memory_engine.retrieve(
            retrieval_strategy="recent", top_k=5
        )

        important_memories = self.memory_engine.retrieve(
            retrieval_strategy="important", threshold=0.7, top_k=5
        )

        relevant_memories = []
        if query_embedding is not None:
            relevant_memories = self.memory_engine.retrieve(
                query_embedding=query_embedding,
                retrieval_strategy="hybrid",
                top_k=10
            )

        # Get semantic knowledge
        key_concepts = [
            c.to_dict()
            for c in self.memory_engine.semantic_memory.get_strongest_concepts(5)
        ]

        patterns = list(self.memory_engine.semantic_memory.patterns.values())[:5]

        # Build state
        state = PerceptionMemoryState(
            current_perception=current_perception,
            world_graph_summary=world_snapshot,
            recent_memories=recent_memories,
            important_memories=important_memories,
            relevant_memories=relevant_memories,
            key_concepts=key_concepts,
            patterns=patterns,
            statistics={
                "memory": self.memory_engine.get_statistics(),
                "perception": self.perception_engine.world_graph.get_statistics()
            }
        )

        return state

    def query_memory(self, query: str, query_embedding: Optional[np.ndarray] = None,
                    strategy: str = "hybrid") -> Dict[str, Any]:
        """
        Query memory system
        """
        context = self.memory_engine.get_context(query, query_embedding)
        return context

    def query_world(self, entity: str, depth: int = 1) -> Dict[str, Any]:
        """
        Query world model about an entity
        """
        return self.perception_engine.query_world(entity, depth)

    def reflect(self, topic: str, query_embedding: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Reflect on memories about a topic
        """
        return self.memory_engine.reflect(topic, query_embedding)

    def record_action(self, action_type: str, parameters: Dict,
                     result: str = None, reward: float = None):
        """
        Record an action taken by the agent
        """
        action = Action(action_type=action_type, parameters=parameters)

        # If there's a result, perceive it
        if result:
            perception = self.perception_engine.perceive_text(result, source="action_result")
        else:
            perception = {"summary": f"Action taken: {action_type}"}

        # Store experience with action and reward
        memory_id = self.memory_engine.store_experience(
            world_snapshot=self.perception_engine.get_world_snapshot(),
            perception_result=perception,
            action={"type": action_type, **parameters},
            reward=reward
        )

        return memory_id

    def consolidate(self) -> Dict:
        """
        Trigger memory consolidation
        """
        return self.memory_engine.consolidate_memories()

    def get_statistics(self) -> Dict:
        """
        Get system statistics
        """
        return {
            "perception": self.perception_engine.world_graph.get_statistics(),
            "memory": self.memory_engine.get_statistics(),
            "timestamp": datetime.now().isoformat()
        }


class LangGraphNode:
    """
    Example LangGraph node that uses cognitive interface
    """

    def __init__(self, cognitive_interface: LangGraphCognitiveInterface):
        self.cognitive = cognitive_interface

    def perceive_node(self, state: Dict) -> Dict:
        """
        Node that processes new observation
        """
        observation = state.get("observation", "")

        result = self.cognitive.observe(observation)

        state["perception"] = result["perception"]
        state["last_memory_id"] = result["memory_id"]

        return state

    def retrieve_node(self, state: Dict) -> Dict:
        """
        Node that retrieves relevant memories
        """
        query = state.get("query", "")
        query_embedding = state.get("query_embedding")

        memory_context = self.cognitive.query_memory(query, query_embedding)

        state["memory_context"] = memory_context

        return state

    def reason_node(self, state: Dict) -> Dict:
        """
        Node that reasons using cognitive state
        """
        # Get full cognitive state
        cognitive_state = self.cognitive.get_cognitive_state()

        # Convert to text for LLM
        context_text = cognitive_state.to_text_summary()

        state["cognitive_context"] = context_text
        state["cognitive_state"] = cognitive_state.to_dict()

        return state

    def act_node(self, state: Dict) -> Dict:
        """
        Node that records actions
        """
        action_type = state.get("action_type", "unknown")
        parameters = state.get("action_parameters", {})
        result = state.get("action_result")
        reward = state.get("reward")

        memory_id = self.cognitive.record_action(
            action_type, parameters, result, reward
        )

        state["action_memory_id"] = memory_id

        return state
