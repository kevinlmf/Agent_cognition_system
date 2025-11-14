"""
Bitcoin Trading Environment Example
Demonstrates memory system effectiveness in cryptocurrency trading
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import CognitiveAgent
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any


class BitcoinTradingAgent:
    """
    Bitcoin trading agent using perception and memory
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        self.agent = CognitiveAgent(mode="langgraph")
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.btc_holdings = 0.0
        self.trade_history = []
        self.price_history = []
        
    def perceive_market(self, btc_price: float, volume: float, 
                       news: str = None, indicators: Dict = None):
        """
        Perceive Bitcoin market state
        """
        obs_text = f"Bitcoin price: ${btc_price:.2f}, Volume: {volume:.2f} BTC"
        
        if indicators:
            obs_text += f", RSI: {indicators.get('rsi', 'N/A')}, "
            obs_text += f"MACD: {indicators.get('macd', 'N/A')}"
        
        if news:
            obs_text += f" | News: {news}"
        
        result = self.agent.perceive(obs_text, source="bitcoin_market")
        return result
    
    def decide_action(self, btc_price: float, indicators: Dict = None) -> str:
        """
        Decide trading action based on memory
        """
        state = self.agent.get_state()
        
        # Retrieve relevant memories
        relevant_memories = state.relevant_memories
        important_memories = state.important_memories
        
        # Analyze memory patterns
        bullish_signals = 0
        bearish_signals = 0
        
        for mem in relevant_memories + important_memories:
            summary = mem.get('summary', '').lower()
            content = mem.get('content', '').lower()
            
            # Bullish indicators
            if any(word in summary or word in content 
                   for word in ['surge', 'rally', 'breakout', 'bullish', 'buy', 'increase']):
                bullish_signals += mem.get('importance', 0.5)
            
            # Bearish indicators
            if any(word in summary or word in content 
                   for word in ['crash', 'drop', 'bearish', 'sell', 'decrease', 'correction']):
                bearish_signals += mem.get('importance', 0.5)
        
        # Technical analysis from memory
        if indicators:
            rsi = indicators.get('rsi', 50)
            if rsi < 30:  # Oversold
                bullish_signals += 0.3
            elif rsi > 70:  # Overbought
                bearish_signals += 0.3
        
        # Decision logic
        if bullish_signals > bearish_signals + 0.2 and self.btc_holdings == 0:
            return "buy"
        elif bearish_signals > bullish_signals + 0.2 and self.btc_holdings > 0:
            return "sell"
        else:
            return "hold"
    
    def execute_trade(self, action: str, btc_price: float) -> float:
        """
        Execute trade and return reward
        """
        reward = 0.0
        
        if action == "buy" and self.btc_holdings == 0:
            # Buy Bitcoin
            btc_amount = self.balance / btc_price
            self.btc_holdings = btc_amount
            self.balance = 0.0
            reward = 0.1  # Small positive reward for action
            
        elif action == "sell" and self.btc_holdings > 0:
            # Sell Bitcoin
            revenue = self.btc_holdings * btc_price
            profit = revenue - self.initial_balance
            self.balance = revenue
            self.btc_holdings = 0.0
            reward = profit / 1000.0  # Reward based on profit
        
        # Record trade
        self.trade_history.append({
            "action": action,
            "btc_price": btc_price,
            "timestamp": datetime.now(),
            "portfolio_value": self.get_portfolio_value(btc_price)
        })
        
        self.price_history.append(btc_price)
        
        # Record in memory
        self.agent.record_action(
            action_type="bitcoin_trade",
            parameters={"action": action, "price": btc_price},
            result=f"Executed {action} at ${btc_price:.2f}",
            reward=reward
        )
        
        return reward
    
    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value"""
        return self.balance + self.btc_holdings * current_price
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get trading statistics"""
        if not self.trade_history:
            return {}
        
        total_trades = len([t for t in self.trade_history if t['action'] != 'hold'])
        final_value = self.get_portfolio_value(self.price_history[-1] if self.price_history else 0)
        total_return = (final_value - self.initial_balance) / self.initial_balance
        
        return {
            "total_trades": total_trades,
            "initial_balance": self.initial_balance,
            "final_value": final_value,
            "total_return": total_return,
            "btc_holdings": self.btc_holdings,
            "cash_balance": self.balance
        }


def simulate_bitcoin_trading():
    """
    Simulate Bitcoin trading with market events
    """
    print("=" * 70)
    print("BITCOIN TRADING AGENT SIMULATION")
    print("=" * 70)
    
    agent = BitcoinTradingAgent(initial_balance=10000.0)
    
    # Simulate Bitcoin market events
    market_events = [
        {
            "time": "Day 1 09:00",
            "price": 45000.0,
            "volume": 1250.5,
            "rsi": 45,
            "macd": 0.5,
            "news": "Bitcoin consolidates around $45k. Institutional interest remains strong."
        },
        {
            "time": "Day 1 15:00",
            "price": 46500.0,
            "volume": 1800.2,
            "rsi": 55,
            "macd": 1.2,
            "news": "Bitcoin breaks above $46k resistance. Bullish momentum building."
        },
        {
            "time": "Day 2 09:00",
            "price": 48000.0,
            "volume": 2200.8,
            "rsi": 65,
            "macd": 2.1,
            "news": "Bitcoin surges to $48k. FOMO buying accelerates."
        },
        {
            "time": "Day 2 15:00",
            "price": 47500.0,
            "volume": 1900.3,
            "rsi": 70,
            "macd": 1.8,
            "news": "Profit taking sets in. Bitcoin pulls back slightly."
        },
        {
            "time": "Day 3 09:00",
            "price": 47000.0,
            "volume": 1500.0,
            "rsi": 68,
            "macd": 1.5,
            "news": "Bitcoin stabilizes. Market awaits next catalyst."
        }
    ]
    
    print(f"\n💰 Starting Balance: ${agent.initial_balance:.2f}")
    print(f"📊 BTC Holdings: {agent.btc_holdings:.6f} BTC")
    
    # Process market events
    for event in market_events:
        print(f"\n{'='*70}")
        print(f"⏰ {event['time']}")
        print(f"💰 BTC Price: ${event['price']:.2f}")
        print(f"📈 Volume: {event['volume']:.2f} BTC")
        print(f"📰 News: {event['news']}")
        
        # Perceive market
        indicators = {"rsi": event['rsi'], "macd": event['macd']}
        agent.perceive_market(
            event['price'], 
            event['volume'],
            event['news'],
            indicators
        )
        
        # Decide action
        action = agent.decide_action(event['price'], indicators)
        print(f"\n🤖 Agent Decision: {action.upper()}")
        
        # Execute trade
        reward = agent.execute_trade(action, event['price'])
        print(f"   Reward: {reward:+.4f}")
        
        # Show portfolio
        portfolio_value = agent.get_portfolio_value(event['price'])
        print(f"   Portfolio Value: ${portfolio_value:.2f}")
        print(f"   BTC Holdings: {agent.btc_holdings:.6f} BTC")
        print(f"   Cash: ${agent.balance:.2f}")
    
    # Summary
    print(f"\n{'='*70}")
    print("📈 TRADING SUMMARY")
    print(f"{'='*70}")
    
    stats = agent.get_statistics()
    print(f"\n💼 Final Statistics:")
    print(f"   Total Trades: {stats['total_trades']}")
    print(f"   Final Value: ${stats['final_value']:.2f}")
    print(f"   Total Return: {stats['total_return']:.2%}")
    print(f"   BTC Holdings: {stats['btc_holdings']:.6f} BTC")
    print(f"   Cash Balance: ${stats['cash_balance']:.2f}")
    
    # Memory statistics
    print(f"\n🧠 Memory System Stats:")
    mem_stats = agent.agent.get_statistics()
    print(f"   Total memories: {mem_stats['memory']['memory_graph']['total_memories']}")
    print(f"   Episodic: {mem_stats['memory']['memory_graph']['episodic_memories']}")
    print(f"   Semantic: {mem_stats['memory']['memory_graph']['semantic_memories']}")
    
    print(f"\n{'='*70}")
    
    return agent


if __name__ == "__main__":
    simulate_bitcoin_trading()

