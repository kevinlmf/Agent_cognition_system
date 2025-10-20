"""
Summarizer: Creates summaries of text content
Simple extractive summarization - can be extended with LLMs
"""
from typing import List, Dict, Any
import re
from collections import Counter


class TextSummarizer:
    """
    Simple extractive text summarizer
    Uses sentence scoring based on word frequency
    """

    def __init__(self):
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those'
        }

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def score_sentences(self, sentences: List[str]) -> Dict[str, float]:
        """
        Score sentences based on word frequency
        Higher score = more important sentence
        """
        # Calculate word frequencies
        all_words = []
        for sentence in sentences:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower())
            filtered = [w for w in words if w not in self.stopwords]
            all_words.extend(filtered)

        word_freq = Counter(all_words)

        # Score each sentence
        sentence_scores = {}
        for sentence in sentences:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower())
            filtered = [w for w in words if w not in self.stopwords]

            if filtered:
                score = sum(word_freq[w] for w in filtered) / len(filtered)
                sentence_scores[sentence] = score
            else:
                sentence_scores[sentence] = 0

        return sentence_scores

    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """
        Create extractive summary by selecting top sentences
        """
        if not text.strip():
            return ""

        sentences = self.split_sentences(text)

        if len(sentences) <= num_sentences:
            return text

        # Score and rank sentences
        scores = self.score_sentences(sentences)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Select top sentences and maintain original order
        top_sentences = [s[0] for s in ranked[:num_sentences]]
        ordered_summary = [s for s in sentences if s in top_sentences]

        return '. '.join(ordered_summary) + '.'

    def summarize_with_keywords(self, text: str, keywords: List[str],
                               num_sentences: int = 3) -> str:
        """
        Create summary emphasizing sentences containing keywords
        """
        sentences = self.split_sentences(text)
        scores = self.score_sentences(sentences)

        # Boost scores for sentences containing keywords
        for sentence in sentences:
            sentence_lower = sentence.lower()
            keyword_count = sum(1 for kw in keywords if kw.lower() in sentence_lower)
            if keyword_count > 0:
                scores[sentence] *= (1 + keyword_count * 0.5)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in ranked[:num_sentences]]
        ordered_summary = [s for s in sentences if s in top_sentences]

        return '. '.join(ordered_summary) + '.'

    def get_key_points(self, text: str, num_points: int = 5) -> List[str]:
        """
        Extract key points as bullet points
        """
        sentences = self.split_sentences(text)
        scores = self.score_sentences(sentences)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [s[0] for s in ranked[:num_points]]


class EventSummarizer(TextSummarizer):
    """
    Specialized summarizer for event streams
    """

    def summarize_events(self, events: List[Dict[str, Any]], max_length: int = 200) -> str:
        """
        Summarize a list of event messages
        """
        # Combine event descriptions
        text = ' '.join([e.get('text', str(e)) for e in events])

        summary = self.summarize(text)

        # Truncate if too long
        if len(summary) > max_length:
            summary = summary[:max_length-3] + '...'

        return summary

    def create_timeline_summary(self, events: List[Dict[str, Any]]) -> str:
        """
        Create a timeline-based summary of events
        """
        if not events:
            return "No events recorded."

        # Sort by timestamp if available
        sorted_events = sorted(
            events,
            key=lambda e: e.get('timestamp', ''),
            reverse=False
        )

        # Select important events
        if len(sorted_events) > 10:
            # Take first, last, and sample middle events
            selected = [
                sorted_events[0],
                *sorted_events[len(sorted_events)//4:len(sorted_events)//4+3],
                *sorted_events[3*len(sorted_events)//4:3*len(sorted_events)//4+3],
                sorted_events[-1]
            ]
        else:
            selected = sorted_events

        # Create timeline
        timeline = []
        for event in selected:
            text = event.get('text', str(event))
            # Truncate long event texts
            if len(text) > 100:
                text = text[:97] + '...'
            timeline.append(text)

        return ' → '.join(timeline)
