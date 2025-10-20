"""
Schema: Unified data structures for agent interfaces
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass
class Observation:
    """Standard observation format"""
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class Action:
    """Standard action format"""
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """Agent's cognitive state"""
    current_observation: Optional[Observation] = None
    world_snapshot: Optional[Dict] = None
    relevant_memories: List[Dict] = field(default_factory=list)
    semantic_concepts: List[Dict] = field(default_factory=list)
    last_action: Optional[Action] = None
    last_reward: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "current_observation": {
                "content": self.current_observation.content if self.current_observation else None,
                "source": self.current_observation.source if self.current_observation else None
            } if self.current_observation else None,
            "world_snapshot": self.world_snapshot,
            "num_relevant_memories": len(self.relevant_memories),
            "num_concepts": len(self.semantic_concepts),
            "last_action": {
                "type": self.last_action.action_type,
                "parameters": self.last_action.parameters
            } if self.last_action else None,
            "last_reward": self.last_reward,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class PerceptionMemoryState:
    """
    Complete perception + memory state for agent consumption
    """
    # Current perception
    current_perception: Dict
    world_graph_summary: Dict

    # Memory context
    recent_memories: List[Dict]
    important_memories: List[Dict]
    relevant_memories: List[Dict]

    # Semantic knowledge
    key_concepts: List[Dict]
    patterns: List[Dict]

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    statistics: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "current_perception": self.current_perception,
            "world_graph_summary": self.world_graph_summary,
            "memory_context": {
                "recent": self.recent_memories,
                "important": self.important_memories,
                "relevant": self.relevant_memories
            },
            "semantic_knowledge": {
                "concepts": self.key_concepts,
                "patterns": self.patterns
            },
            "statistics": self.statistics,
            "timestamp": self.timestamp.isoformat()
        }

    def to_text_summary(self) -> str:
        """
        Convert state to text summary for LLM consumption
        """
        summary_parts = []

        # Current situation
        summary_parts.append("=== Current Situation ===")
        summary_parts.append(self.current_perception.get('summary', 'No current perception'))

        # World state
        if self.world_graph_summary:
            summary_parts.append("\n=== World State ===")
            stats = self.world_graph_summary.get('statistics', {})
            summary_parts.append(f"Entities: {stats.get('num_entities', 0)}, Relations: {stats.get('num_relations', 0)}")

        # Recent context
        if self.recent_memories:
            summary_parts.append("\n=== Recent Memories ===")
            for mem in self.recent_memories[:3]:
                summary_parts.append(f"- {mem.get('summary', '')}")

        # Important facts
        if self.important_memories:
            summary_parts.append("\n=== Important Facts ===")
            for mem in self.important_memories[:3]:
                summary_parts.append(f"- {mem.get('summary', '')} (importance: {mem.get('importance', 0):.2f})")

        # Key concepts
        if self.key_concepts:
            summary_parts.append("\n=== Key Concepts ===")
            for concept in self.key_concepts[:5]:
                summary_parts.append(f"- {concept.get('name', '')}")

        return "\n".join(summary_parts)
