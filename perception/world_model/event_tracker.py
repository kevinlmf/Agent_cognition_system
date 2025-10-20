"""
Event Tracker: Tracks events and updates WorldGraph
Maintains temporal relationships and causal chains
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from ..world_model.world_graph import WorldGraph


class Event:
    """Represents a single event"""
    def __init__(self, event_id: str, event_type: str, description: str,
                 entities: List[str], timestamp: datetime = None):
        self.event_id = event_id
        self.event_type = event_type
        self.description = description
        self.entities = entities
        self.timestamp = timestamp or datetime.now()
        self.metadata = {}

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "description": self.description,
            "entities": self.entities,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class EventTracker:
    """
    Tracks events and maintains causal relationships
    Updates WorldGraph based on events
    """

    def __init__(self):
        self.events: List[Event] = []
        self.event_by_id: Dict[str, Event] = {}
        self.causal_chains: List[List[str]] = []  # List of event_id chains

    def add_event(self, event_type: str, description: str,
                 entities: List[str], metadata: Dict = None) -> Event:
        """Add a new event"""
        event_id = f"event_{len(self.events)}"
        event = Event(event_id, event_type, description, entities)

        if metadata:
            event.metadata = metadata

        self.events.append(event)
        self.event_by_id[event_id] = event

        return event

    def link_causal_events(self, cause_event_id: str, effect_event_id: str):
        """Link two events in a causal relationship"""
        # Find or create causal chain
        found_chain = None
        for chain in self.causal_chains:
            if cause_event_id in chain:
                found_chain = chain
                break

        if found_chain:
            # Add to existing chain
            cause_idx = found_chain.index(cause_event_id)
            found_chain.insert(cause_idx + 1, effect_event_id)
        else:
            # Create new chain
            self.causal_chains.append([cause_event_id, effect_event_id])

    def update_world_graph(self, world_graph: WorldGraph, event: Event):
        """Update WorldGraph based on an event"""
        # Add event entities to graph
        for entity in event.entities:
            if entity not in world_graph.graph:
                world_graph.add_entity(entity, entity_type="event_entity")

        # Add event-specific relations
        if event.event_type == "announcement":
            # For announcements, first entity is the announcer
            if len(event.entities) >= 2:
                world_graph.add_relation(
                    event.entities[0], "announced", event.entities[1],
                    confidence=0.9,
                    attributes={"event_id": event.event_id}
                )

        elif event.event_type == "change":
            # For changes, track what changed
            if len(event.entities) >= 2:
                world_graph.add_relation(
                    event.entities[0], "changed", event.entities[1],
                    confidence=0.9,
                    attributes={"event_id": event.event_id}
                )

        elif event.event_type == "impact":
            # For impacts, track cause and effect
            if len(event.entities) >= 2:
                world_graph.add_relation(
                    event.entities[0], "affects", event.entities[1],
                    confidence=0.8,
                    attributes={"event_id": event.event_id}
                )

    def get_recent_events(self, n: int = 10) -> List[Event]:
        """Get n most recent events"""
        return self.events[-n:]

    def get_events_by_type(self, event_type: str) -> List[Event]:
        """Get all events of a specific type"""
        return [e for e in self.events if e.event_type == event_type]

    def get_events_with_entity(self, entity: str) -> List[Event]:
        """Get all events involving an entity"""
        return [e for e in self.events if entity in e.entities]

    def get_causal_chain(self, event_id: str) -> Optional[List[str]]:
        """Get the causal chain containing an event"""
        for chain in self.causal_chains:
            if event_id in chain:
                return chain
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get event tracking statistics"""
        event_types = {}
        for event in self.events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1

        return {
            "total_events": len(self.events),
            "event_types": event_types,
            "causal_chains": len(self.causal_chains),
            "oldest_event": self.events[0].timestamp.isoformat() if self.events else None,
            "newest_event": self.events[-1].timestamp.isoformat() if self.events else None
        }
