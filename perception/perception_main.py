"""
Perception Main: Orchestrates the perception pipeline
External World → Collectors → Preprocessors → WorldGraph
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

from .collectors.web_collector import WebCollector, NewsCollector, MarketDataCollector
from .collectors.text_stream_collector import TextStreamCollector, SimulationEventCollector
from .preprocessors.entity_extractor import EntityExtractor
from .preprocessors.summarizer import TextSummarizer, EventSummarizer
from .world_model.world_graph import WorldGraph
from .world_model.event_tracker import EventTracker


class PerceptionEngine:
    """
    Main perception engine that coordinates data collection and world model building
    """

    def __init__(self):
        # Collectors
        self.web_collector = WebCollector()
        self.news_collector = NewsCollector()
        self.market_collector = MarketDataCollector()
        self.stream_collector = TextStreamCollector()

        # Preprocessors
        self.entity_extractor = EntityExtractor()
        self.summarizer = TextSummarizer()
        self.event_summarizer = EventSummarizer()

        # World model
        self.world_graph = WorldGraph()
        self.event_tracker = EventTracker()

    def perceive_text(self, text: str, source: str = "text") -> Dict[str, Any]:
        """
        Process raw text and update world model
        Returns perception result
        """
        # Extract entities and relations
        graph_data = self.entity_extractor.build_entity_graph_data(text)

        # Create summary
        summary = self.summarizer.summarize(text, num_sentences=2)

        # Update WorldGraph
        for entity in graph_data['entities']:
            self.world_graph.add_entity(
                entity['name'],
                entity_type=entity['type'],
                attributes={"source": source}
            )

        for relation in graph_data['relations']:
            self.world_graph.add_relation(
                relation['source'],
                relation['relation'],
                relation['target'],
                confidence=0.8
            )

        # Track as event
        entity_names = [e['name'] for e in graph_data['entities']]
        event = self.event_tracker.add_event(
            event_type="text_perception",
            description=summary,
            entities=entity_names,
            metadata={"source": source, "keywords": graph_data['keywords']}
        )

        return {
            "summary": summary,
            "entities": graph_data['entities'],
            "relations": graph_data['relations'],
            "keywords": graph_data['keywords'],
            "event_id": event.event_id,
            "timestamp": datetime.now().isoformat()
        }

    def perceive_url(self, url: str) -> Dict[str, Any]:
        """
        Fetch and process content from a URL
        """
        # Fetch content
        result = self.web_collector.fetch_url(url)

        if result.get('status') == 'failed':
            return {
                "status": "failed",
                "error": result.get('error'),
                "url": url
            }

        # Process the content
        content = result.get('content', '')
        perception = self.perceive_text(content, source=url)
        perception['url'] = url

        return perception

    def perceive_news(self, topic: str, max_articles: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch and process news about a topic
        """
        articles = self.news_collector.fetch_news(topic, max_results=max_articles)
        perceptions = []

        for article in articles:
            content = f"{article['title']}. {article['content']}"
            perception = self.perceive_text(content, source=article['source'])
            perception['article'] = article
            perceptions.append(perception)

        return perceptions

    def perceive_market_update(self, symbol: str) -> Dict[str, Any]:
        """
        Process market data and update world model
        """
        market_data = self.market_collector.fetch_market_status(symbol)

        # Build narrative from market data
        narrative = f"{symbol} is trading at ${market_data['price']} with {market_data['change']}% change. Market sentiment is {market_data['market_sentiment']}."

        perception = self.perceive_text(narrative, source="market_data")
        perception['market_data'] = market_data

        return perception

    def perceive_stream_batch(self, n_messages: int = 50) -> Dict[str, Any]:
        """
        Process a batch of stream messages
        """
        messages = self.stream_collector.get_recent(n_messages)

        if not messages:
            return {"status": "no_messages"}

        # Combine messages for processing
        combined_text = ' '.join([m['text'] for m in messages])
        summary = self.event_summarizer.summarize_events(messages)

        # Extract entities from combined text
        graph_data = self.entity_extractor.build_entity_graph_data(combined_text)

        # Update world graph
        for entity in graph_data['entities']:
            self.world_graph.add_entity(
                entity['name'],
                entity_type=entity['type']
            )

        return {
            "num_messages": len(messages),
            "summary": summary,
            "entities": graph_data['entities'],
            "timestamp": datetime.now().isoformat()
        }

    def get_world_snapshot(self) -> Dict[str, Any]:
        """
        Get current state of the perceived world
        """
        return {
            "world_graph": self.world_graph.to_dict(),
            "recent_events": [e.to_dict() for e in self.event_tracker.get_recent_events(10)],
            "statistics": {
                "world_graph": self.world_graph.get_statistics(),
                "events": self.event_tracker.get_statistics()
            },
            "timestamp": datetime.now().isoformat()
        }

    def query_world(self, entity: str, depth: int = 1) -> Dict[str, Any]:
        """
        Query the world model about an entity
        """
        # Get entity information
        entity_info = self.world_graph.get_entity(entity)

        if not entity_info:
            return {"status": "not_found", "entity": entity}

        # Get relations
        relations = self.world_graph.get_relations(entity)

        # Get related events
        events = self.event_tracker.get_events_with_entity(entity)

        # Get subgraph
        subgraph = self.world_graph.get_subgraph(entity, depth=depth)

        return {
            "entity": entity,
            "entity_info": entity_info,
            "relations": relations,
            "related_events": [e.to_dict() for e in events[-5:]],
            "subgraph_size": {
                "nodes": subgraph.number_of_nodes(),
                "edges": subgraph.number_of_edges()
            }
        }

    def reset(self):
        """Reset the perception engine"""
        self.world_graph = WorldGraph()
        self.event_tracker = EventTracker()
        self.stream_collector.clear_buffer()
