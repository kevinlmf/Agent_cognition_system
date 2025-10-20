"""
MemoryGraph: Long-term memory storage using graph structure
Stores episodic memories (events) and semantic memories (concepts)
"""
import numpy as np
import networkx as nx
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import uuid


@dataclass
class MemoryNode:
    """Represents a single memory unit"""
    memory_id: str
    memory_type: str  # 'episodic' or 'semantic'
    timestamp: datetime
    content: str
    summary: str
    embedding: Optional[np.ndarray] = None
    world_graph_snapshot: Optional[Dict] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5  # 0-1 scale
    access_count: int = 0
    last_access: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "summary": self.summary,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "world_graph_snapshot": self.world_graph_snapshot,
            "metadata": self.metadata,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_access": self.last_access.isoformat() if self.last_access else None
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryNode':
        embedding = np.array(data['embedding']) if data.get('embedding') else None
        last_access = datetime.fromisoformat(data['last_access']) if data.get('last_access') else None

        return cls(
            memory_id=data['memory_id'],
            memory_type=data['memory_type'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            content=data['content'],
            summary=data['summary'],
            embedding=embedding,
            world_graph_snapshot=data.get('world_graph_snapshot'),
            metadata=data.get('metadata', {}),
            importance=data.get('importance', 0.5),
            access_count=data.get('access_count', 0),
            last_access=last_access
        )


class MemoryGraph:
    """
    MemoryGraph stores and manages agent's long-term memory
    - Episodic: time-stamped events and experiences
    - Semantic: abstracted concepts and patterns
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.memories: Dict[str, MemoryNode] = {}
        self.episodic_timeline: List[str] = []  # Ordered list of episodic memory IDs

    def add_memory(self, memory_type: str, content: str, summary: str,
                   embedding: Optional[np.ndarray] = None,
                   world_graph_snapshot: Optional[Dict] = None,
                   importance: float = 0.5,
                   metadata: Dict[str, Any] = None) -> str:
        """
        Add a new memory to the graph
        Returns the memory_id
        """
        memory_id = str(uuid.uuid4())

        memory = MemoryNode(
            memory_id=memory_id,
            memory_type=memory_type,
            timestamp=datetime.now(),
            content=content,
            summary=summary,
            embedding=embedding,
            world_graph_snapshot=world_graph_snapshot,
            importance=importance,
            metadata=metadata or {}
        )

        self.memories[memory_id] = memory
        self.graph.add_node(memory_id, **memory.to_dict())

        if memory_type == "episodic":
            self.episodic_timeline.append(memory_id)

            # Connect to previous episodic memory (temporal link)
            if len(self.episodic_timeline) > 1:
                prev_id = self.episodic_timeline[-2]
                self.add_relation(prev_id, "followed_by", memory_id, relation_type="temporal")

        return memory_id

    def add_relation(self, source_id: str, relation: str, target_id: str,
                    relation_type: str = "associative", strength: float = 1.0,
                    metadata: Dict = None):
        """
        Add a relationship between two memories
        relation_type: 'temporal', 'causal', 'associative', 'similarity'
        """
        if source_id not in self.memories or target_id not in self.memories:
            raise ValueError("Both memories must exist before creating relation")

        self.graph.add_edge(
            source_id, target_id,
            relation=relation,
            relation_type=relation_type,
            strength=strength,
            created_at=datetime.now().isoformat(),
            metadata=metadata or {}
        )

    def get_memory(self, memory_id: str) -> Optional[MemoryNode]:
        """Retrieve a memory by ID and update access statistics"""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory.access_count += 1
            memory.last_access = datetime.now()
            return memory
        return None

    def get_episodic_range(self, start_time: datetime, end_time: datetime) -> List[MemoryNode]:
        """Get all episodic memories within a time range"""
        return [
            self.memories[mid]
            for mid in self.episodic_timeline
            if start_time <= self.memories[mid].timestamp <= end_time
        ]

    def get_recent_memories(self, n: int = 10, memory_type: str = None) -> List[MemoryNode]:
        """Get the n most recent memories"""
        if memory_type == "episodic":
            recent_ids = self.episodic_timeline[-n:]
            return [self.memories[mid] for mid in recent_ids]

        # For all or semantic memories, sort by timestamp
        all_memories = [
            m for m in self.memories.values()
            if memory_type is None or m.memory_type == memory_type
        ]
        all_memories.sort(key=lambda x: x.timestamp, reverse=True)
        return all_memories[:n]

    def find_similar_memories(self, query_embedding: np.ndarray,
                            top_k: int = 5,
                            memory_type: str = None,
                            threshold: float = 0.0) -> List[Tuple[MemoryNode, float]]:
        """
        Find memories similar to query embedding using cosine similarity
        Returns list of (memory, similarity_score) tuples
        """
        results = []

        for memory in self.memories.values():
            if memory_type and memory.memory_type != memory_type:
                continue

            if memory.embedding is None:
                continue

            # Cosine similarity
            similarity = np.dot(query_embedding, memory.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(memory.embedding)
            )

            if similarity >= threshold:
                results.append((memory, float(similarity)))

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_connected_memories(self, memory_id: str,
                              relation_type: str = None,
                              max_depth: int = 2) -> List[Tuple[MemoryNode, List[str]]]:
        """
        Get memories connected to the given memory
        Returns list of (memory, path) tuples
        """
        if memory_id not in self.graph:
            return []

        results = []

        # BFS to find connected memories
        for target_id in nx.descendants(self.graph, memory_id):
            if target_id == memory_id:
                continue

            try:
                paths = list(nx.all_simple_paths(
                    self.graph, memory_id, target_id, cutoff=max_depth
                ))

                if paths:
                    # Filter by relation_type if specified
                    if relation_type:
                        valid_path = False
                        for path in paths:
                            for i in range(len(path)-1):
                                edge_data = self.graph[path[i]][path[i+1]]
                                if edge_data.get('relation_type') == relation_type:
                                    valid_path = True
                                    break
                        if not valid_path:
                            continue

                    results.append((self.memories[target_id], paths[0]))
            except nx.NetworkXNoPath:
                continue

        return results

    def consolidate_semantic_memory(self, episodic_ids: List[str],
                                   summary: str,
                                   embedding: Optional[np.ndarray] = None) -> str:
        """
        Create a semantic memory by consolidating multiple episodic memories
        Links the semantic memory to its source episodic memories
        """
        # Create semantic memory
        semantic_id = self.add_memory(
            memory_type="semantic",
            content=f"Consolidated from {len(episodic_ids)} episodic memories",
            summary=summary,
            embedding=embedding,
            importance=0.8,  # Semantic memories are typically more important
            metadata={"source_episodes": episodic_ids}
        )

        # Link to source episodic memories
        for ep_id in episodic_ids:
            if ep_id in self.memories:
                self.add_relation(
                    ep_id, "abstracted_to", semantic_id,
                    relation_type="abstraction",
                    strength=1.0
                )

        return semantic_id

    def forget_low_importance(self, threshold: float = 0.2, keep_recent: int = 100):
        """
        Forget memories below importance threshold (except recent ones)
        Implements a simple forgetting mechanism
        """
        # Always keep recent memories
        recent_ids = set(self.episodic_timeline[-keep_recent:])

        to_remove = []
        for memory_id, memory in self.memories.items():
            if memory_id not in recent_ids and memory.importance < threshold:
                to_remove.append(memory_id)

        for memory_id in to_remove:
            self._remove_memory(memory_id)

        return len(to_remove)

    def _remove_memory(self, memory_id: str):
        """Remove a memory from the graph"""
        if memory_id in self.memories:
            del self.memories[memory_id]

        if memory_id in self.graph:
            self.graph.remove_node(memory_id)

        if memory_id in self.episodic_timeline:
            self.episodic_timeline.remove(memory_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory graph statistics"""
        episodic_count = sum(1 for m in self.memories.values() if m.memory_type == "episodic")
        semantic_count = sum(1 for m in self.memories.values() if m.memory_type == "semantic")

        return {
            "total_memories": len(self.memories),
            "episodic_memories": episodic_count,
            "semantic_memories": semantic_count,
            "total_relations": self.graph.number_of_edges(),
            "avg_importance": np.mean([m.importance for m in self.memories.values()]) if self.memories else 0,
            "most_accessed": self._get_most_accessed(5)
        }

    def _get_most_accessed(self, n: int) -> List[Dict]:
        """Get the n most accessed memories"""
        sorted_memories = sorted(
            self.memories.values(),
            key=lambda m: m.access_count,
            reverse=True
        )[:n]

        return [
            {
                "memory_id": m.memory_id,
                "summary": m.summary,
                "access_count": m.access_count
            }
            for m in sorted_memories
        ]

    def save(self, filepath: str):
        """Save memory graph to JSON file"""
        data = {
            "memories": {mid: m.to_dict() for mid, m in self.memories.items()},
            "episodic_timeline": self.episodic_timeline,
            "edges": [
                {
                    "source": u,
                    "target": v,
                    **data
                }
                for u, v, data in self.graph.edges(data=True)
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'MemoryGraph':
        """Load memory graph from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        mg = cls()

        # Restore memories
        for mid, mdata in data['memories'].items():
            mg.memories[mid] = MemoryNode.from_dict(mdata)
            mg.graph.add_node(mid, **mdata)

        # Restore timeline
        mg.episodic_timeline = data['episodic_timeline']

        # Restore edges
        for edge in data['edges']:
            source = edge.pop('source')
            target = edge.pop('target')
            mg.graph.add_edge(source, target, **edge)

        return mg

    def __repr__(self):
        return f"MemoryGraph(episodic={sum(1 for m in self.memories.values() if m.memory_type == 'episodic')}, semantic={sum(1 for m in self.memories.values() if m.memory_type == 'semantic')})"
