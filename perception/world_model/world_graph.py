"""
WorldGraph: Represents the current state of the external world
Uses NetworkX to build entity-relation graphs dynamically
"""
import networkx as nx
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import json


class WorldGraph:
    """
    WorldGraph maintains a dynamic graph of entities and their relationships
    in the external world at a specific point in time.
    """

    def __init__(self, graph_id: str = None):
        self.graph = nx.MultiDiGraph()  # Allow multiple edges between nodes
        self.graph_id = graph_id or f"world_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.timestamp = datetime.now()
        self.metadata = {}

    def add_entity(self, entity: str, entity_type: str = "general",
                   attributes: Dict[str, Any] = None):
        """Add an entity (node) to the world graph"""
        if attributes is None:
            attributes = {}

        self.graph.add_node(
            entity,
            entity_type=entity_type,
            timestamp=datetime.now().isoformat(),
            **attributes
        )

    def add_relation(self, source: str, relation: str, target: str,
                    confidence: float = 1.0, attributes: Dict[str, Any] = None):
        """Add a relationship (edge) between entities"""
        if attributes is None:
            attributes = {}

        # Ensure both entities exist
        if source not in self.graph:
            self.add_entity(source)
        if target not in self.graph:
            self.add_entity(target)

        self.graph.add_edge(
            source, target,
            relation=relation,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            **attributes
        )

    def get_entity(self, entity: str) -> Optional[Dict]:
        """Get entity information"""
        if entity in self.graph:
            return dict(self.graph.nodes[entity])
        return None

    def get_relations(self, entity: str, direction: str = "both") -> List[Tuple]:
        """
        Get all relations of an entity
        direction: 'out', 'in', or 'both'
        """
        relations = []

        if entity not in self.graph:
            return relations

        if direction in ["out", "both"]:
            for target in self.graph.successors(entity):
                for key, edge_data in self.graph[entity][target].items():
                    relations.append((entity, edge_data['relation'], target, edge_data))

        if direction in ["in", "both"]:
            for source in self.graph.predecessors(entity):
                for key, edge_data in self.graph[source][entity].items():
                    relations.append((source, edge_data['relation'], entity, edge_data))

        return relations

    def query_path(self, source: str, target: str, max_depth: int = 3) -> List[List[str]]:
        """Find all paths between two entities"""
        try:
            paths = list(nx.all_simple_paths(
                self.graph, source, target, cutoff=max_depth
            ))
            return paths
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return []

    def get_subgraph(self, entity: str, depth: int = 1) -> nx.DiGraph:
        """Get a subgraph centered on an entity within specified depth"""
        if entity not in self.graph:
            return nx.DiGraph()

        # BFS to find all nodes within depth
        nodes = {entity}
        current_layer = {entity}

        for _ in range(depth):
            next_layer = set()
            for node in current_layer:
                next_layer.update(self.graph.successors(node))
                next_layer.update(self.graph.predecessors(node))
            nodes.update(next_layer)
            current_layer = next_layer

        return self.graph.subgraph(nodes).copy()

    def merge_graph(self, other_graph: 'WorldGraph', strategy: str = "update"):
        """
        Merge another WorldGraph into this one
        strategy: 'update' (overwrite), 'append' (keep both), 'weighted' (confidence-based)
        """
        for node, data in other_graph.graph.nodes(data=True):
            if strategy == "update" or node not in self.graph:
                self.graph.add_node(node, **data)
            elif strategy == "weighted":
                # Keep node with more recent timestamp
                existing = self.graph.nodes[node]
                if data.get('timestamp', '') > existing.get('timestamp', ''):
                    self.graph.add_node(node, **data)

        for source, target, key, data in other_graph.graph.edges(data=True, keys=True):
            if strategy == "update":
                self.graph.add_edge(source, target, **data)
            elif strategy == "append":
                self.graph.add_edge(source, target, **data)
            elif strategy == "weighted":
                # Keep edge with higher confidence
                existing_edges = self.graph.get_edge_data(source, target)
                if existing_edges:
                    max_conf = max(e.get('confidence', 0) for e in existing_edges.values())
                    if data.get('confidence', 0) > max_conf:
                        self.graph.add_edge(source, target, **data)
                else:
                    self.graph.add_edge(source, target, **data)

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        
        # Handle empty graph case
        if num_nodes == 0:
            return {
                "num_entities": 0,
                "num_relations": 0,
                "density": 0.0,
                "is_connected": False,
                "num_components": 0,
                "entity_types": {},
                "relation_types": {}
            }
        
        return {
            "num_entities": num_nodes,
            "num_relations": num_edges,
            "density": nx.density(self.graph),
            "is_connected": nx.is_weakly_connected(self.graph),
            "num_components": nx.number_weakly_connected_components(self.graph),
            "entity_types": self._count_entity_types(),
            "relation_types": self._count_relation_types()
        }

    def _count_entity_types(self) -> Dict[str, int]:
        """Count entities by type"""
        type_count = {}
        for node, data in self.graph.nodes(data=True):
            etype = data.get('entity_type', 'general')
            type_count[etype] = type_count.get(etype, 0) + 1
        return type_count

    def _count_relation_types(self) -> Dict[str, int]:
        """Count relations by type"""
        rel_count = {}
        for _, _, data in self.graph.edges(data=True):
            rel = data.get('relation', 'unknown')
            rel_count[rel] = rel_count.get(rel, 0) + 1
        return rel_count

    def to_dict(self) -> Dict:
        """Export graph to dictionary format"""
        return {
            "graph_id": self.graph_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "nodes": [
                {"id": node, **data}
                for node, data in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, "key": k, **data}
                for u, v, k, data in self.graph.edges(data=True, keys=True)
            ],
            "statistics": self.get_statistics()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'WorldGraph':
        """Load graph from dictionary"""
        wg = cls(graph_id=data['graph_id'])
        wg.timestamp = datetime.fromisoformat(data['timestamp'])
        wg.metadata = data.get('metadata', {})

        for node in data['nodes']:
            node_id = node.pop('id')
            wg.graph.add_node(node_id, **node)

        for edge in data['edges']:
            source = edge.pop('source')
            target = edge.pop('target')
            edge.pop('key', None)
            wg.graph.add_edge(source, target, **edge)

        return wg

    def save(self, filepath: str):
        """Save graph to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'WorldGraph':
        """Load graph from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self):
        return f"WorldGraph(id={self.graph_id}, entities={self.graph.number_of_nodes()}, relations={self.graph.number_of_edges()})"
