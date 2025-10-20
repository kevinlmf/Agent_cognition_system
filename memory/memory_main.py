"""
Memory Main: Orchestrates the memory system
Manages episodic, semantic memory and consolidation
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

from .memory_graph import MemoryGraph
from .episodic_memory import EpisodicMemory
from .semantic_memory import SemanticMemory
from .retrieval import MemoryRetriever, ContextBuilder


class MemoryEngine:
    """
    Main memory engine that coordinates episodic and semantic memory
    """

    def __init__(self):
        self.memory_graph = MemoryGraph()
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        self.retriever = MemoryRetriever(self.memory_graph)

        # Consolidation settings
        self.consolidation_threshold = 50  # Episodes before consolidation
        self.episodes_since_consolidation = 0

    def store_experience(self, world_snapshot: Dict,
                        perception_result: Dict,
                        action: Optional[Dict] = None,
                        reward: Optional[float] = None,
                        embedding: Optional[np.ndarray] = None) -> str:
        """
        Store a new experience
        Creates both episodic memory and memory graph entry
        """
        # Store in episodic memory
        episode_id = self.episodic_memory.add_episode(
            world_snapshot, perception_result, action, reward, embedding
        )

        # Store in memory graph
        summary = perception_result.get('summary', 'Experience recorded')
        content = f"Episode: {summary}"

        if action:
            content += f" | Action: {action.get('type', 'unknown')}"

        if reward is not None:
            content += f" | Reward: {reward:.2f}"

        memory_id = self.memory_graph.add_memory(
            memory_type="episodic",
            content=content,
            summary=summary,
            embedding=embedding,
            world_graph_snapshot=world_snapshot,
            importance=self.episodic_memory.episodes[-1]['importance'],
            metadata={
                "episode_id": episode_id,
                "action": action,
                "reward": reward
            }
        )

        self.episodes_since_consolidation += 1

        # Auto-consolidate if threshold reached
        if self.episodes_since_consolidation >= self.consolidation_threshold:
            self.consolidate_memories()

        return memory_id

    def consolidate_memories(self, lookback: int = 100) -> Dict[str, Any]:
        """
        Consolidate recent episodic memories into semantic knowledge
        """
        # Get recent episodes
        recent_episodes = self.episodic_memory.get_recent(lookback)

        # Extract semantic concepts
        new_concepts = self.semantic_memory.consolidate_from_episodes(
            recent_episodes, min_frequency=3
        )

        # Detect patterns
        new_patterns = self.semantic_memory.detect_patterns(recent_episodes)

        # Create semantic memory nodes in memory graph
        for concept_id in new_concepts:
            concept = self.semantic_memory.get_concept(concept_id)

            semantic_summary = f"Concept: {concept.name} (appeared {len(concept.examples)} times)"

            semantic_mem_id = self.memory_graph.add_memory(
                memory_type="semantic",
                content=f"Semantic concept learned from episodes: {concept.name}",
                summary=semantic_summary,
                importance=concept.strength,
                metadata={
                    "concept_id": concept_id,
                    "concept_type": concept.concept_type,
                    "examples": concept.examples[:10]  # Limit for storage
                }
            )

            # Link to source episodes in memory graph
            for ep_id in concept.examples[:5]:  # Link to top 5 examples
                # Find corresponding memory ID
                for mem_id, memory in self.memory_graph.memories.items():
                    if memory.metadata.get('episode_id') == ep_id:
                        self.memory_graph.add_relation(
                            mem_id, "abstracted_to", semantic_mem_id,
                            relation_type="abstraction"
                        )

        self.episodes_since_consolidation = 0

        return {
            "new_concepts": len(new_concepts),
            "new_patterns": len(new_patterns),
            "total_semantic_concepts": len(self.semantic_memory.concepts),
            "timestamp": datetime.now().isoformat()
        }

    def retrieve(self, query: str = None,
                query_embedding: Optional[np.ndarray] = None,
                retrieval_strategy: str = "hybrid",
                top_k: int = 10,
                **kwargs) -> List[Dict]:
        """
        Retrieve memories using specified strategy
        """
        if retrieval_strategy == "recent":
            return self.retriever.retrieve_recent(top_k, kwargs.get('memory_type'))

        elif retrieval_strategy == "important":
            return self.retriever.retrieve_by_importance(
                kwargs.get('threshold', 0.7), top_k
            )

        elif retrieval_strategy == "similar":
            if query_embedding is None:
                raise ValueError("query_embedding required for similarity retrieval")
            return self.retriever.retrieve_by_similarity(
                query_embedding, top_k, kwargs.get('memory_type'), kwargs.get('threshold', 0.5)
            )

        elif retrieval_strategy == "hybrid":
            return self.retriever.retrieve_hybrid(
                query_embedding,
                kwargs.get('importance_weight', 0.3),
                kwargs.get('recency_weight', 0.3),
                kwargs.get('similarity_weight', 0.4),
                top_k
            )

        else:
            raise ValueError(f"Unknown retrieval strategy: {retrieval_strategy}")

    def get_context(self, query: str, query_embedding: Optional[np.ndarray] = None) -> Dict:
        """
        Get comprehensive context for a query
        """
        return self.retriever.retrieve_context(query, query_embedding)

    def reflect(self, topic: str, query_embedding: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Reflect on memories related to a topic
        Generates insights from consolidated memories
        """
        # Retrieve relevant memories
        relevant_memories = self.retrieve(
            query=topic,
            query_embedding=query_embedding,
            retrieval_strategy="hybrid",
            top_k=20
        )

        # Get related semantic concepts
        if query_embedding is not None:
            related_concepts = self.semantic_memory.search_concepts(
                query_embedding, top_k=5
            )
        else:
            related_concepts = []

        # Get patterns
        patterns = list(self.semantic_memory.patterns.values())[:5]

        return {
            "topic": topic,
            "relevant_memories": relevant_memories[:10],
            "key_concepts": [
                {
                    "name": c['concept'].name,
                    "strength": c['concept'].strength,
                    "examples": len(c['concept'].examples)
                }
                for c in related_concepts
            ],
            "patterns": patterns,
            "timestamp": datetime.now().isoformat()
        }

    def prune_memories(self, importance_threshold: float = 0.2,
                      keep_recent: int = 100) -> Dict[str, int]:
        """
        Prune low-importance memories
        """
        # Prune memory graph
        forgotten = self.memory_graph.forget_low_importance(
            importance_threshold, keep_recent
        )

        # Prune semantic concepts
        pruned_concepts = self.semantic_memory.prune_weak_concepts(
            importance_threshold
        )

        return {
            "forgotten_memories": forgotten,
            "pruned_concepts": pruned_concepts,
            "remaining_memories": len(self.memory_graph.memories),
            "remaining_concepts": len(self.semantic_memory.concepts)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics
        """
        return {
            "memory_graph": self.memory_graph.get_statistics(),
            "episodic_memory": self.episodic_memory.get_statistics(),
            "semantic_memory": self.semantic_memory.get_statistics(),
            "episodes_since_consolidation": self.episodes_since_consolidation
        }

    def save(self, base_path: str):
        """Save all memory components"""
        self.memory_graph.save(f"{base_path}_memory_graph.json")
        # Note: Episodic and semantic memories would need save methods too

    def load(self, base_path: str):
        """Load all memory components"""
        self.memory_graph = MemoryGraph.load(f"{base_path}_memory_graph.json")
        # Note: Load other components similarly
