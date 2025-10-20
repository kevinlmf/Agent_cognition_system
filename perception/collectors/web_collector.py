"""
Web Collector: Collects data from web sources (news, APIs, etc.)
"""
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
import json


class WebCollector:
    """Collects data from web sources"""

    def __init__(self, user_agent: str = "PerceptionAgent/1.0"):
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': user_agent})

    def fetch_url(self, url: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
        """
        Fetch content from a URL
        Returns structured data with metadata
        """
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()

            return {
                "url": url,
                "content": response.text,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "timestamp": datetime.now().isoformat(),
                "content_type": response.headers.get('Content-Type', 'unknown')
            }
        except requests.RequestException as e:
            return {
                "url": url,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "failed"
            }

    def fetch_json_api(self, url: str, params: Dict = None,
                      headers: Dict = None) -> Optional[Dict[str, Any]]:
        """
        Fetch data from a JSON API
        """
        try:
            response = self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()

            return {
                "url": url,
                "data": response.json(),
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
        except (requests.RequestException, json.JSONDecodeError) as e:
            return {
                "url": url,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "failed"
            }

    def batch_fetch(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Fetch multiple URLs in batch"""
        results = []
        for url in urls:
            result = self.fetch_url(url)
            if result:
                results.append(result)
        return results


class NewsCollector(WebCollector):
    """
    Specialized collector for news sources
    Can be extended to integrate with NewsAPI, RSS feeds, etc.
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key

    def fetch_news(self, topic: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch news articles about a topic
        Placeholder implementation - integrate with actual news API
        """
        # Example structure for news data
        return [
            {
                "title": f"News about {topic}",
                "content": "Article content...",
                "source": "NewsSource",
                "published_at": datetime.now().isoformat(),
                "url": f"https://example.com/news/{topic}",
                "timestamp": datetime.now().isoformat()
            }
        ]


class MarketDataCollector(WebCollector):
    """
    Specialized collector for financial market data
    Example implementation for market perception
    """

    def fetch_market_status(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current market status for a symbol
        Placeholder - integrate with actual financial APIs
        """
        return {
            "symbol": symbol,
            "price": 0.0,
            "volume": 0,
            "change": 0.0,
            "timestamp": datetime.now().isoformat(),
            "market_sentiment": "neutral"
        }

    def fetch_economic_indicators(self) -> Dict[str, Any]:
        """
        Fetch major economic indicators
        Placeholder implementation
        """
        return {
            "interest_rate": 0.0,
            "inflation": 0.0,
            "gdp_growth": 0.0,
            "unemployment": 0.0,
            "timestamp": datetime.now().isoformat()
        }
