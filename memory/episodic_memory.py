"""
Episodic Memory: Time-stamped event storage
Stores specific experiences in chronological order
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np


class EpisodicMemory:
    """
    Manages episodic memories (time-stamped events and experiences)
    """

    def __init__(self, max_memories: int = 10000):
        self.max_memories = max_memories
        self.episodes: List[Dict[str, Any]] = []

    def add_episode(self, world_snapshot: Dict, perception_result: Dict,
                   action: Optional[Dict] = None,
                   reward: Optional[float] = None,
                   embedding: Optional[np.ndarray] = None) -> str:
        """
        Store an episodic memory
        Returns episode_id
        """
        episode_id = f"ep_{len(self.episodes)}_{int(datetime.now().timestamp())}"

        episode = {
            "episode_id": episode_id,
            "timestamp": datetime.now().isoformat(),
            "world_snapshot": world_snapshot,
            "perception": perception_result,
            "action": action,
            "reward": reward,
            "embedding": embedding.tolist() if embedding is not None else None,
            "importance": self._calculate_importance(reward, perception_result)
        }

        self.episodes.append(episode)

        # Maintain max size
        if len(self.episodes) > self.max_memories:
            self._prune_memories()

        return episode_id

    def _calculate_importance(self, reward: Optional[float],
                             perception: Dict) -> float:
        """
        Calculate importance score for an episode
        Based on reward, novelty, and emotional salience
        """
        importance = 0.5  # Base importance

        # Reward-based importance
        if reward is not None:
            # High rewards or losses increase importance
            importance += abs(reward) * 0.3

        # Novelty-based importance (more entities = more novel)
        num_entities = len(perception.get('entities', []))
        importance += min(num_entities * 0.05, 0.3)

        return min(importance, 1.0)

    def get_episode(self, episode_id: str) -> Optional[Dict]:
        """Get a specific episode"""
        for episode in self.episodes:
            if episode['episode_id'] == episode_id:
                return episode
        return None

    def get_recent(self, n: int = 10) -> List[Dict]:
        """Get n most recent episodes"""
        return self.episodes[-n:]

    def get_by_timerange(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get episodes within a time range"""
        results = []
        for episode in self.episodes:
            ep_time = datetime.fromisoformat(episode['timestamp'])
            if start_time <= ep_time <= end_time:
                results.append(episode)
        return results

    def get_by_importance(self, threshold: float = 0.7, limit: int = 100) -> List[Dict]:
        """Get important episodes above threshold"""
        important = [
            ep for ep in self.episodes
            if ep.get('importance', 0) >= threshold
        ]
        return sorted(important, key=lambda x: x['importance'], reverse=True)[:limit]

    def search_by_embedding(self, query_embedding: np.ndarray,
                          top_k: int = 10,
                          threshold: float = 0.5) -> List[Dict]:
        """
        Search episodes by embedding similarity
        """
        if query_embedding is None:
            return []

        results = []
        for episode in self.episodes:
            if episode['embedding'] is None:
                continue

            ep_embedding = np.array(episode['embedding'])

            # Cosine similarity
            similarity = np.dot(query_embedding, ep_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(ep_embedding)
            )

            if similarity >= threshold:
                results.append({
                    "episode": episode,
                    "similarity": float(similarity)
                })

        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

    def get_context_window(self, episode_id: str, window_size: int = 5) -> List[Dict]:
        """
        Get episodes around a specific episode (temporal context)
        """
        # Find episode index
        idx = None
        for i, ep in enumerate(self.episodes):
            if ep['episode_id'] == episode_id:
                idx = i
                break

        if idx is None:
            return []

        # Get surrounding episodes
        start = max(0, idx - window_size)
        end = min(len(self.episodes), idx + window_size + 1)

        return self.episodes[start:end]

    def _prune_memories(self):
        """
        Prune old, low-importance memories
        Keep recent and important memories
        """
        # Always keep recent 1000 memories
        keep_recent = 1000
        recent_cutoff = len(self.episodes) - keep_recent

        # Sort older memories by importance
        older = self.episodes[:recent_cutoff]
        older.sort(key=lambda x: x.get('importance', 0), reverse=True)

        # Keep top 50% of older memories by importance
        keep_old = older[:len(older)//2]

        # Combine
        self.episodes = keep_old + self.episodes[recent_cutoff:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get episodic memory statistics"""
        if not self.episodes:
            return {
                "total_episodes": 0,
                "avg_importance": 0,
                "timespan": None
            }

        importances = [ep.get('importance', 0) for ep in self.episodes]
        first_time = datetime.fromisoformat(self.episodes[0]['timestamp'])
        last_time = datetime.fromisoformat(self.episodes[-1]['timestamp'])

        return {
            "total_episodes": len(self.episodes),
            "avg_importance": np.mean(importances),
            "max_importance": max(importances),
            "min_importance": min(importances),
            "timespan_hours": (last_time - first_time).total_seconds() / 3600,
            "oldest_episode": self.episodes[0]['timestamp'],
            "newest_episode": self.episodes[-1]['timestamp']
        }
