"""
Text Stream Collector: Processes streaming text data
Useful for real-time feeds, logs, chat streams, etc.
"""
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from collections import deque
import time


class TextStreamCollector:
    """
    Collects and buffers streaming text data
    """

    def __init__(self, buffer_size: int = 1000):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.total_messages = 0

    def add_message(self, text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Add a text message to the stream
        """
        message = {
            "id": self.total_messages,
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        self.buffer.append(message)
        self.total_messages += 1
        return message

    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the n most recent messages"""
        return list(self.buffer)[-n:]

    def get_range(self, start_id: int, end_id: int) -> List[Dict[str, Any]]:
        """Get messages within ID range"""
        return [msg for msg in self.buffer if start_id <= msg['id'] <= end_id]

    def filter_by_keyword(self, keyword: str, case_sensitive: bool = False) -> List[Dict[str, Any]]:
        """Filter messages containing a keyword"""
        if case_sensitive:
            return [msg for msg in self.buffer if keyword in msg['text']]
        else:
            keyword_lower = keyword.lower()
            return [msg for msg in self.buffer if keyword_lower in msg['text'].lower()]

    def batch_process(self, batch_size: int = 50,
                     processor: Callable[[List[Dict]], Any] = None) -> List[Any]:
        """
        Process messages in batches
        """
        results = []
        batch = []

        for message in self.buffer:
            batch.append(message)
            if len(batch) >= batch_size:
                if processor:
                    result = processor(batch)
                    results.append(result)
                batch = []

        # Process remaining messages
        if batch and processor:
            result = processor(batch)
            results.append(result)

        return results

    def clear_buffer(self):
        """Clear the message buffer"""
        self.buffer.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get collector statistics"""
        return {
            "total_messages": self.total_messages,
            "buffer_size": len(self.buffer),
            "max_buffer_size": self.buffer_size,
            "oldest_message_time": self.buffer[0]['timestamp'] if self.buffer else None,
            "newest_message_time": self.buffer[-1]['timestamp'] if self.buffer else None
        }


class SimulationEventCollector(TextStreamCollector):
    """
    Specialized collector for simulation events
    Useful for RL environments, trading simulations, etc.
    """

    def __init__(self, buffer_size: int = 1000):
        super().__init__(buffer_size)
        self.event_types = {}

    def add_event(self, event_type: str, description: str,
                 state: Dict[str, Any] = None,
                 reward: float = None) -> Dict[str, Any]:
        """
        Add a simulation event
        """
        metadata = {
            "event_type": event_type,
            "state": state,
            "reward": reward
        }

        message = self.add_message(description, metadata)

        # Track event types
        self.event_types[event_type] = self.event_types.get(event_type, 0) + 1

        return message

    def get_events_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Get all events of a specific type"""
        return [
            msg for msg in self.buffer
            if msg['metadata'].get('event_type') == event_type
        ]

    def get_event_statistics(self) -> Dict[str, Any]:
        """Get statistics about event types"""
        return {
            "event_type_counts": self.event_types.copy(),
            "total_event_types": len(self.event_types),
            **self.get_statistics()
        }
