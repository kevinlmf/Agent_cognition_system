"""
Semantic Memory: Abstract knowledge and patterns
Stores consolidated, generalized knowledge extracted from episodes
"""
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import numpy as np
from collections import defaultdict


class Concept:
    """Represents a semantic concept"""
    def __init__(self, concept_id: str, name: str, concept_type: str):
        self.concept_id = concept_id
        self.name = name
        self.concept_type = concept_type  # 'entity', 'pattern', 'rule', 'fact'
        self.properties: Dict[str, Any] = {}
        self.examples: List[str] = []  # Episode IDs
        self.related_concepts: Set[str] = set()
        self.strength: float = 0.5  # How well-established this concept is
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 0

    def add_example(self, episode_id: str):
        """Link an episode as an example of this concept"""
        if episode_id not in self.examples:
            self.examples.append(episode_id)
            self.strength = min(1.0, self.strength + 0.05)

    def relate_to(self, concept_id: str):
        """Create relationship to another concept"""
        self.related_concepts.add(concept_id)

    def access(self):
        """Mark concept as accessed (for importance tracking)"""
        self.access_count += 1
        self.last_accessed = datetime.now()

    def to_dict(self) -> Dict:
        return {
            "concept_id": self.concept_id,
            "name": self.name,
            "concept_type": self.concept_type,
            "properties": self.properties,
            "num_examples": len(self.examples),
            "num_relations": len(self.related_concepts),
            "strength": self.strength,
            "access_count": self.access_count
        }


class SemanticMemory:
    """
    Manages semantic memories (abstract knowledge, patterns, rules)
    """

    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.patterns: Dict[str, Dict] = {}  # Learned patterns
        self.rules: List[Dict] = []  # Learned rules
        self.concept_embeddings: Dict[str, np.ndarray] = {}

    def add_concept(self, name: str, concept_type: str,
                   properties: Dict = None,
                   embedding: Optional[np.ndarray] = None) -> str:
        """
        Add a new semantic concept
        Returns concept_id
        """
        concept_id = f"concept_{len(self.concepts)}_{name.replace(' ', '_')}"

        concept = Concept(concept_id, name, concept_type)

        if properties:
            concept.properties = properties

        self.concepts[concept_id] = concept

        if embedding is not None:
            self.concept_embeddings[concept_id] = embedding

        return concept_id

    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Get a concept by ID"""
        if concept_id in self.concepts:
            concept = self.concepts[concept_id]
            concept.access()
            return concept
        return None

    def find_concept_by_name(self, name: str) -> Optional[Concept]:
        """Find concept by name"""
        for concept in self.concepts.values():
            if concept.name.lower() == name.lower():
                concept.access()
                return concept
        return None

    def link_concepts(self, concept_id1: str, concept_id2: str,
                     relation_type: str = "related"):
        """Create a relationship between concepts"""
        if concept_id1 in self.concepts and concept_id2 in self.concepts:
            self.concepts[concept_id1].relate_to(concept_id2)
            self.concepts[concept_id2].relate_to(concept_id1)

    def consolidate_from_episodes(self, episodes: List[Dict],
                                 min_frequency: int = 3) -> List[str]:
        """
        Extract semantic concepts from episodic memories
        Returns list of new concept IDs
        """
        new_concepts = []

        # Count entity frequencies across episodes
        entity_freq = defaultdict(int)
        entity_types = {}

        for episode in episodes:
            perception = episode.get('perception', {})
            entities = perception.get('entities', [])

            for entity in entities:
                entity_name = entity.get('name', entity.get('text', ''))
                entity_freq[entity_name] += 1
                entity_types[entity_name] = entity.get('type', 'general')

        # Create concepts for frequent entities
        for entity_name, freq in entity_freq.items():
            if freq >= min_frequency:
                # Check if concept already exists
                existing = self.find_concept_by_name(entity_name)

                if existing:
                    # Strengthen existing concept
                    for episode in episodes:
                        perception = episode.get('perception', {})
                        entities = perception.get('entities', [])
                        if any(e.get('name') == entity_name for e in entities):
                            existing.add_example(episode['episode_id'])
                else:
                    # Create new concept
                    concept_id = self.add_concept(
                        name=entity_name,
                        concept_type='entity',
                        properties={
                            'frequency': freq,
                            'entity_type': entity_types.get(entity_name, 'general')
                        }
                    )

                    # Link to source episodes
                    concept = self.concepts[concept_id]
                    for episode in episodes:
                        perception = episode.get('perception', {})
                        entities = perception.get('entities', [])
                        if any(e.get('name') == entity_name for e in entities):
                            concept.add_example(episode['episode_id'])

                    new_concepts.append(concept_id)

        return new_concepts

    def detect_patterns(self, episodes: List[Dict]) -> List[Dict]:
        """
        Detect recurring patterns in episodes
        """
        patterns = []

        # Simple pattern: repeated entity co-occurrences
        co_occurrences = defaultdict(int)

        for episode in episodes:
            perception = episode.get('perception', {})
            entities = [e.get('name', '') for e in perception.get('entities', [])]

            # Count pairs
            for i, e1 in enumerate(entities):
                for e2 in entities[i+1:]:
                    pair = tuple(sorted([e1, e2]))
                    co_occurrences[pair] += 1

        # Identify significant patterns
        for pair, count in co_occurrences.items():
            if count >= 3:  # Threshold for pattern
                pattern_id = f"pattern_{len(self.patterns)}"
                pattern = {
                    "pattern_id": pattern_id,
                    "type": "co_occurrence",
                    "entities": list(pair),
                    "frequency": count,
                    "strength": min(1.0, count * 0.1)
                }
                self.patterns[pattern_id] = pattern
                patterns.append(pattern)

        return patterns

    def extract_rule(self, condition: str, conclusion: str,
                    confidence: float, supporting_episodes: List[str]) -> str:
        """
        Add a learned rule
        Example: "IF Fed raises rates THEN market volatility increases"
        """
        rule_id = f"rule_{len(self.rules)}"
        rule = {
            "rule_id": rule_id,
            "condition": condition,
            "conclusion": conclusion,
            "confidence": confidence,
            "supporting_episodes": supporting_episodes,
            "created_at": datetime.now().isoformat()
        }

        self.rules.append(rule)
        return rule_id

    def search_concepts(self, query_embedding: np.ndarray,
                       top_k: int = 10) -> List[Dict]:
        """
        Search concepts by embedding similarity
        """
        if not self.concept_embeddings:
            return []

        results = []
        for concept_id, embedding in self.concept_embeddings.items():
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )

            results.append({
                "concept": self.concepts[concept_id],
                "similarity": float(similarity)
            })

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

    def get_related_concepts(self, concept_id: str, max_depth: int = 2) -> List[Concept]:
        """
        Get concepts related to a given concept (BFS)
        """
        if concept_id not in self.concepts:
            return []

        visited = set([concept_id])
        queue = [(concept_id, 0)]
        related = []

        while queue:
            current_id, depth = queue.pop(0)

            if depth >= max_depth:
                continue

            current = self.concepts[current_id]

            for related_id in current.related_concepts:
                if related_id not in visited:
                    visited.add(related_id)
                    queue.append((related_id, depth + 1))
                    related.append(self.concepts[related_id])

        return related

    def get_strongest_concepts(self, n: int = 20) -> List[Concept]:
        """Get concepts with highest strength (well-established)"""
        sorted_concepts = sorted(
            self.concepts.values(),
            key=lambda c: c.strength,
            reverse=True
        )
        return sorted_concepts[:n]

    def prune_weak_concepts(self, threshold: float = 0.2):
        """Remove weak, rarely accessed concepts"""
        to_remove = []

        for concept_id, concept in self.concepts.items():
            if concept.strength < threshold and concept.access_count < 2:
                to_remove.append(concept_id)

        for concept_id in to_remove:
            del self.concepts[concept_id]
            if concept_id in self.concept_embeddings:
                del self.concept_embeddings[concept_id]

        return len(to_remove)

    def get_statistics(self) -> Dict[str, Any]:
        """Get semantic memory statistics"""
        if not self.concepts:
            return {
                "total_concepts": 0,
                "total_patterns": 0,
                "total_rules": 0
            }

        concept_types = defaultdict(int)
        for concept in self.concepts.values():
            concept_types[concept.concept_type] += 1

        strengths = [c.strength for c in self.concepts.values()]

        return {
            "total_concepts": len(self.concepts),
            "concept_types": dict(concept_types),
            "total_patterns": len(self.patterns),
            "total_rules": len(self.rules),
            "avg_concept_strength": np.mean(strengths) if strengths else 0,
            "strongest_concepts": [c.to_dict() for c in self.get_strongest_concepts(5)]
        }
