"""
Example: Trading Agent with Perception + Memory
Demonstrates a realistic trading scenario
"""
import sys
sys.path.append('..')

from main import CognitiveAgent
import numpy as np
from datetime import datetime, timedelta


class TradingAgent:
    """
    Trading agent that uses perception and memory for decision making
    """

    def __init__(self):
        self.agent = CognitiveAgent(mode="langgraph")
        self.position = 0  # Current position: -1 (short), 0 (neutral), 1 (long)
        self.cash = 10000.0
        self.shares = 0
        self.trade_history = []

    def perceive_market(self, market_data: dict, news: str = None):
        """
        Perceive market state
        """
        # Build observation text
        obs_text = f"Price: ${market_data['price']:.2f}, Volume: {market_data['volume']}"

        if news:
            obs_text += f" | News: {news}"

        # Perceive
        result = self.agent.perceive(obs_text, source="market")

        return result

    def decide_action(self, market_data: dict) -> str:
        """
        Decide trading action based on cognitive state
        """
        # Get cognitive state
        state = self.agent.get_state()

        # Simple decision logic based on memories
        important_memories = state.important_memories

        # Count positive vs negative sentiments in recent memories
        positive_count = 0
        negative_count = 0

        for mem in important_memories:
            summary = mem.get('summary', '').lower()
            if 'increase' in summary or 'surge' in summary or 'positive' in summary:
                positive_count += 1
            elif 'decrease' in summary or 'drop' in summary or 'negative' in summary:
                negative_count += 1

        # Decision
        if positive_count > negative_count and self.position <= 0:
            return "buy"
        elif negative_count > positive_count and self.position >= 0:
            return "sell"
        else:
            return "hold"

    def execute_trade(self, action: str, price: float) -> float:
        """
        Execute trade and return reward
        """
        reward = 0.0

        if action == "buy" and self.position != 1:
            # Buy
            shares_to_buy = self.cash // price
            cost = shares_to_buy * price
            self.cash -= cost
            self.shares += shares_to_buy
            self.position = 1
            reward = 0.1  # Small positive reward for taking action

        elif action == "sell" and self.position != -1:
            # Sell
            revenue = self.shares * price
            self.cash += revenue
            profit = revenue - (self.shares * self.trade_history[-1]['price'] if self.trade_history else revenue)
            self.shares = 0
            self.position = -1
            reward = profit / 1000.0  # Reward based on profit

        # Record trade
        self.trade_history.append({
            "action": action,
            "price": price,
            "position": self.position,
            "portfolio_value": self.cash + self.shares * price,
            "timestamp": datetime.now()
        })

        # Record action in memory
        self.agent.record_action(
            action_type="trade",
            parameters={"action": action, "price": price},
            result=f"Executed {action} at ${price:.2f}",
            reward=reward
        )

        return reward

    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value"""
        return self.cash + self.shares * current_price


def simulate_trading_day():
    """
    Simulate a trading day with market events
    """
    print("=" * 70)
    print("TRADING AGENT SIMULATION")
    print("=" * 70)

    agent = TradingAgent()

    # Simulate market data
    market_events = [
        # Morning: Market opens
        {
            "time": "09:30",
            "price": 100.0,
            "volume": 1000,
            "news": "Market opens steady. Investors await Fed announcement."
        },
        # Mid-morning: News breaks
        {
            "time": "10:30",
            "price": 102.5,
            "volume": 1500,
            "news": "Federal Reserve announces rate cut. Market sentiment improves."
        },
        # Noon: Rally continues
        {
            "time": "12:00",
            "price": 105.0,
            "volume": 2000,
            "news": "Stock prices surge on positive Fed news. Strong buying momentum."
        },
        # Afternoon: Profit taking
        {
            "time": "14:00",
            "price": 103.0,
            "volume": 1800,
            "news": "Profit taking sets in. Prices pull back slightly from highs."
        },
        # Close: Stabilization
        {
            "time": "16:00",
            "price": 104.0,
            "volume": 1200,
            "news": "Market closes near highs. Overall positive day for equities."
        }
    ]

    print("\n📊 Starting Portfolio:")
    print(f"   Cash: ${agent.cash:.2f}")
    print(f"   Shares: {agent.shares}")
    print(f"   Position: {agent.position}")

    # Process each market event
    for i, event in enumerate(market_events):
        print(f"\n{'='*70}")
        print(f"⏰ Time: {event['time']}")
        print(f"💰 Price: ${event['price']:.2f} | Volume: {event['volume']}")
        print(f"📰 News: {event['news']}")

        # Perceive market
        agent.perceive_market(
            {"price": event['price'], "volume": event['volume']},
            event['news']
        )

        # Decide action
        action = agent.decide_action({"price": event['price']})
        print(f"\n🤖 Agent Decision: {action.upper()}")

        # Execute trade
        reward = agent.execute_trade(action, event['price'])
        print(f"   Reward: {reward:+.2f}")

        # Show portfolio
        portfolio_value = agent.get_portfolio_value(event['price'])
        print(f"   Portfolio Value: ${portfolio_value:.2f}")
        print(f"   Position: {['SHORT', 'NEUTRAL', 'LONG'][agent.position + 1]}")

    # End of day summary
    print(f"\n{'='*70}")
    print("📈 END OF DAY SUMMARY")
    print(f"{'='*70}")

    final_value = agent.get_portfolio_value(market_events[-1]['price'])
    pnl = final_value - 10000.0

    print(f"\n💼 Final Portfolio:")
    print(f"   Cash: ${agent.cash:.2f}")
    print(f"   Shares: {agent.shares}")
    print(f"   Total Value: ${final_value:.2f}")
    print(f"   P&L: ${pnl:+.2f} ({pnl/10000*100:+.2f}%)")

    # Show trades
    print(f"\n📜 Trade History:")
    for trade in agent.trade_history:
        if trade['action'] != 'hold':
            print(f"   {trade['action'].upper():<6} @ ${trade['price']:.2f}")

    # Memory consolidation
    print(f"\n🧠 Consolidating memories...")
    result = agent.agent.consolidate()
    print(f"   New concepts learned: {result['new_concepts']}")
    print(f"   Patterns detected: {result['new_patterns']}")

    # System statistics
    print(f"\n📊 Cognitive System Stats:")
    stats = agent.agent.get_statistics()
    mem_stats = stats['memory']['memory_graph']
    print(f"   Total memories: {mem_stats['total_memories']}")
    print(f"   Episodic: {mem_stats['episodic_memories']}")
    print(f"   Semantic: {mem_stats['semantic_memories']}")
    print(f"   World entities tracked: {stats['perception']['num_entities']}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    simulate_trading_day()
