"""
Entity Extractor: Extracts entities and relations from text
Uses simple pattern matching and keyword extraction
Can be extended with NER models (spaCy, transformers, etc.)
"""
import re
from typing import List, Dict, Tuple, Any, Set
from collections import Counter


class EntityExtractor:
    """
    Extracts entities from text using pattern matching and heuristics
    """

    def __init__(self):
        # Common entity patterns (can be extended)
        self.patterns = {
            "organization": r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|Ltd|LLC|Company)\b',
            "money": r'\$\d+(?:,\d{3})*(?:\.\d{2})?(?:\s*(?:million|billion|trillion))?',
            "percentage": r'\d+(?:\.\d+)?%',
            "date": r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
        }

        # Financial keywords
        self.financial_entities = {
            "Federal Reserve", "Fed", "Treasury", "SEC", "FDIC",
            "Stock Market", "Nasdaq", "S&P 500", "Dow Jones",
            "Interest Rate", "Inflation", "GDP", "Unemployment"
        }

        # Relation keywords
        self.relation_keywords = {
            "raises": "increases",
            "lowers": "decreases",
            "increases": "increases",
            "decreases": "decreases",
            "affects": "affects",
            "influences": "influences",
            "causes": "causes",
            "announces": "announces",
            "reports": "reports"
        }

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text
        Returns list of entity dictionaries
        """
        entities = []

        # Extract using patterns
        for entity_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    "text": match.group(),
                    "type": entity_type,
                    "start": match.start(),
                    "end": match.end()
                })

        # Extract known financial entities
        text_lower = text.lower()
        for entity in self.financial_entities:
            if entity.lower() in text_lower:
                pos = text_lower.find(entity.lower())
                entities.append({
                    "text": entity,
                    "type": "financial_entity",
                    "start": pos,
                    "end": pos + len(entity)
                })

        # Extract capitalized phrases (potential entities)
        capitalized = re.finditer(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for match in capitalized:
            # Skip if already captured
            if not any(e['start'] == match.start() for e in entities):
                entities.append({
                    "text": match.group(),
                    "type": "named_entity",
                    "start": match.start(),
                    "end": match.end()
                })

        return entities

    def extract_relations(self, text: str, entities: List[Dict[str, Any]] = None) -> List[Tuple[str, str, str]]:
        """
        Extract relations between entities
        Returns list of (subject, relation, object) tuples
        """
        if entities is None:
            entities = self.extract_entities(text)

        relations = []

        # Simple heuristic: find relation keywords between entities
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Get text between entities
                start = min(entity1['end'], entity2['end'])
                end = max(entity1['start'], entity2['start'])

                if start < end:
                    between = text[start:end].lower()

                    # Check for relation keywords
                    for keyword, relation in self.relation_keywords.items():
                        if keyword in between:
                            # Determine direction
                            if entity1['start'] < entity2['start']:
                                relations.append((
                                    entity1['text'],
                                    relation,
                                    entity2['text']
                                ))
                            else:
                                relations.append((
                                    entity2['text'],
                                    relation,
                                    entity1['text']
                                ))
                            break

        return relations

    def extract_keywords(self, text: str, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Extract important keywords from text
        Returns list of (keyword, frequency) tuples
        """
        # Remove common words (simple stopwords)
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }

        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

        # Filter stopwords and count
        filtered_words = [w for w in words if w not in stopwords]
        word_counts = Counter(filtered_words)

        return word_counts.most_common(top_n)

    def build_entity_graph_data(self, text: str) -> Dict[str, Any]:
        """
        Extract entities and relations and format for WorldGraph
        Returns structured data ready for graph construction
        """
        entities = self.extract_entities(text)
        relations = self.extract_relations(text, entities)
        keywords = self.extract_keywords(text)

        return {
            "entities": [
                {
                    "name": e['text'],
                    "type": e['type']
                }
                for e in entities
            ],
            "relations": [
                {
                    "source": r[0],
                    "relation": r[1],
                    "target": r[2]
                }
                for r in relations
            ],
            "keywords": [
                {
                    "word": k[0],
                    "frequency": k[1]
                }
                for k in keywords
            ]
        }


class SimpleNERExtractor(EntityExtractor):
    """
    Enhanced entity extractor using basic NER techniques
    Can be extended to use spaCy or transformers
    """

    def __init__(self):
        super().__init__()
        # Add more sophisticated patterns
        self.patterns.update({
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "url": r'https?://[^\s]+',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        })

    def extract_entities_with_context(self, text: str, context_window: int = 50) -> List[Dict[str, Any]]:
        """
        Extract entities with surrounding context
        """
        entities = self.extract_entities(text)

        for entity in entities:
            start = max(0, entity['start'] - context_window)
            end = min(len(text), entity['end'] + context_window)
            entity['context'] = text[start:end]

        return entities
