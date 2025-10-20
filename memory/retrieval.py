"""
Memory Retrieval: Smart retrieval from MemoryGraph
Supports various retrieval strategies: recency, importance, similarity
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np


class MemoryRetriever:
    """
    Retrieves memories using multiple strategies
    """

    def __init__(self, memory_graph):
        self.memory_graph = memory_graph

    def retrieve_recent(self, n: int = 10, memory_type: str = None) -> List[Dict]:
        """
        Retrieve n most recent memories
        """
        memories = self.memory_graph.get_recent_memories(n, memory_type)
        return [self._memory_to_dict(m) for m in memories]

    def retrieve_by_importance(self, threshold: float = 0.7,
                              limit: int = 20) -> List[Dict]:
        """
        Retrieve important memories above threshold
        """
        important_memories = [
            m for m in self.memory_graph.memories.values()
            if m.importance >= threshold
        ]

        # Sort by importance
        important_memories.sort(key=lambda m: m.importance, reverse=True)

        return [self._memory_to_dict(m) for m in important_memories[:limit]]

    def retrieve_by_similarity(self, query_embedding: np.ndarray,
                              top_k: int = 10,
                              memory_type: str = None,
                              threshold: float = 0.5) -> List[Dict]:
        """
        Retrieve memories similar to query embedding
        """
        results = self.memory_graph.find_similar_memories(
            query_embedding, top_k, memory_type, threshold
        )

        return [
            {
                **self._memory_to_dict(memory),
                "similarity": similarity
            }
            for memory, similarity in results
        ]

    def retrieve_by_timerange(self, start_time: datetime,
                             end_time: datetime) -> List[Dict]:
        """
        Retrieve memories within time range
        """
        memories = self.memory_graph.get_episodic_range(start_time, end_time)
        return [self._memory_to_dict(m) for m in memories]

    def retrieve_connected(self, memory_id: str,
                          relation_type: str = None,
                          max_depth: int = 2) -> List[Dict]:
        """
        Retrieve memories connected to a given memory
        """
        connected = self.memory_graph.get_connected_memories(
            memory_id, relation_type, max_depth
        )

        return [
            {
                **self._memory_to_dict(memory),
                "path": path
            }
            for memory, path in connected
        ]

    def retrieve_hybrid(self, query_embedding: Optional[np.ndarray] = None,
                       importance_weight: float = 0.3,
                       recency_weight: float = 0.3,
                       similarity_weight: float = 0.4,
                       top_k: int = 10) -> List[Dict]:
        """
        Hybrid retrieval combining multiple factors
        """
        if query_embedding is None:
            # If no embedding, use importance + recency only
            importance_weight = 0.5
            recency_weight = 0.5
            similarity_weight = 0.0

        scored_memories = []

        for memory in self.memory_graph.memories.values():
            score = 0.0

            # Importance score
            score += memory.importance * importance_weight

            # Recency score (decay over time)
            time_diff = (datetime.now() - memory.timestamp).total_seconds()
            recency_score = np.exp(-time_diff / (7 * 24 * 3600))  # 7-day half-life
            score += recency_score * recency_weight

            # Similarity score
            if query_embedding is not None and memory.embedding is not None:
                similarity = np.dot(query_embedding, memory.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(memory.embedding)
                )
                score += similarity * similarity_weight

            scored_memories.append((memory, score))

        # Sort by score
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                **self._memory_to_dict(memory),
                "retrieval_score": float(score)
            }
            for memory, score in scored_memories[:top_k]
        ]

    def retrieve_context(self, query: str, query_embedding: Optional[np.ndarray] = None,
                        context_size: int = 5) -> Dict[str, Any]:
        """
        Retrieve comprehensive context for a query
        Combines multiple retrieval strategies
        """
        context = {
            "query": query,
            "timestamp": datetime.now().isoformat()
        }

        # Recent memories for temporal context
        context["recent"] = self.retrieve_recent(5)

        # Important memories for key facts
        context["important"] = self.retrieve_by_importance(threshold=0.7, limit=5)

        # Similar memories if embedding provided
        if query_embedding is not None:
            context["similar"] = self.retrieve_by_similarity(
                query_embedding, top_k=5, threshold=0.5
            )

        # Hybrid retrieval for best overall match
        context["relevant"] = self.retrieve_hybrid(
            query_embedding, top_k=context_size
        )

        return context

    def _memory_to_dict(self, memory) -> Dict:
        """Convert memory node to dictionary"""
        return {
            "memory_id": memory.memory_id,
            "memory_type": memory.memory_type,
            "timestamp": memory.timestamp.isoformat(),
            "content": memory.content,
            "summary": memory.summary,
            "importance": memory.importance,
            "metadata": memory.metadata,
            "access_count": memory.access_count
        }


class ContextBuilder:
    """
    Builds rich context by combining perception and memory
    """

    def __init__(self, perception_engine, memory_graph):
        self.perception_engine = perception_engine
        self.memory_graph = memory_graph
        self.retriever = MemoryRetriever(memory_graph)

    def build_situation_context(self, current_observation: str,
                               query_embedding: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Build comprehensive situation context
        Combines current perception with relevant memories
        """
        # Perceive current situation
        current_perception = self.perception_engine.perceive_text(
            current_observation, source="situation_context"
        )

        # Retrieve relevant memories
        memory_context = self.retriever.retrieve_context(
            current_observation, query_embedding, context_size=5
        )

        # Get world state
        world_state = self.perception_engine.query_world(
            current_perception['entities'][0]['name'] if current_perception['entities'] else "unknown"
        )

        return {
            "current_perception": current_perception,
            "memory_context": memory_context,
            "world_state": world_state,
            "timestamp": datetime.now().isoformat()
        }
