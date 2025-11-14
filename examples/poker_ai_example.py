"""
Poker AI Example: Complete demonstration of Memory → Opponent Model → Range → Best Response → RL

This example shows:
1. How memory records opponent actions
2. How opponent models are built and updated
3. How ranges are estimated using Bayesian updates
4. How best response strategies are calculated
5. How RL adapts the strategy over time
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.poker_agent import PokerRLAgent, GameState
from memory.opponent_memory import HandAction
from datetime import datetime
import numpy as np


def simulate_poker_session():
    """
    Simulate a poker session demonstrating the full AI system
    """
    print("=" * 80)
    print("POKER AI SYSTEM DEMONSTRATION")
    print("Memory → Opponent Model → Range → Best Response → RL")
    print("=" * 80)
    
    # Initialize agent
    agent = PokerRLAgent(agent_id="poker_ai_1", gto_weight=0.3)
    
    # Opponents
    opponents = ["opponent_1", "opponent_2"]
    
    print("\n🎯 Initializing opponents...")
    for opp_id in opponents:
        agent.initialize_opponent(opp_id)
        print(f"   ✓ Initialized model for {opp_id}")
    
    # Simulate hands
    print("\n" + "=" * 80)
    print("SIMULATING HANDS")
    print("=" * 80)
    
    # Hand 1: Opponent 1 - Tight player
    print("\n📊 HAND 1: Opponent 1")
    print("-" * 80)
    
    hand_id_1 = "hand_001"
    
    # Pre-flop actions
    print("\n🃏 Pre-flop:")
    action1 = HandAction(
        action_type="raise",
        amount=20.0,
        street="preflop",
        position="late",
        pot_size=10.0
    )
    agent.observe_action("opponent_1", hand_id_1, action1)
    print(f"   Opponent 1: {action1.action_type.upper()} ${action1.amount}")
    
    # Get opponent model
    opp1_model = agent.get_opponent_model("opponent_1")
    print(f"\n   📈 Opponent Model Updated:")
    print(f"      Player Type: {opp1_model.player_type}")
    stats = opp1_model.get_tendency()
    print(f"      VPIP: {stats.vpip:.2%}")
    print(f"      PFR: {stats.pfr:.2%}")
    print(f"      Aggression Factor: {stats.aggression_factor:.2f}")
    
    # Flop actions
    print("\n🃏 Flop:")
    action2 = HandAction(
        action_type="bet",
        amount=30.0,
        street="flop",
        position="late",
        pot_size=50.0
    )
    agent.observe_action("opponent_1", hand_id_1, action2)
    print(f"   Opponent 1: {action2.action_type.upper()} ${action2.amount}")
    
    # Get range estimate
    range_est = agent.range_estimators["opponent_1"]
    current_range = range_est.get_range("flop")
    print(f"\n   🎯 Range Estimate (Flop):")
    for strength, prob in current_range.items():
        print(f"      {strength.capitalize()}: {prob:.2%}")
    print(f"      Confidence: {range_est.get_range_confidence():.2%}")
    
    # Our decision
    print("\n🤖 Our Decision:")
    game_state = GameState(
        street="flop",
        pot_size=50.0,
        bet_to_call=30.0,
        position="early",
        stack_size=1000.0,
        opponent_stack=1000.0
    )
    
    action_ev = agent.decide_action("opponent_1", game_state, "medium")
    print(f"   Action: {action_ev.action.upper()}")
    print(f"   Expected Value: ${action_ev.ev:.2f}")
    print(f"   Win Probability: {action_ev.win_probability:.2%}")
    print(f"   Confidence: {action_ev.confidence:.2%}")
    
    # Hand result
    agent.record_hand_result(hand_id_1, "opponent_1", "fold", -30.0)
    print(f"\n   💰 Result: Lost ${30.0}")
    
    # Hand 2: Opponent 1 - More actions
    print("\n" + "=" * 80)
    print("📊 HAND 2: Opponent 1 (More data)")
    print("-" * 80)
    
    hand_id_2 = "hand_002"
    
    # Multiple actions
    actions_hand2 = [
        HandAction("call", 5.0, "preflop", "late", 10.0),
        HandAction("check", 0.0, "flop", "late", 20.0),
        HandAction("bet", 25.0, "turn", "late", 20.0),
    ]
    
    for action in actions_hand2:
        agent.observe_action("opponent_1", hand_id_2, action)
        print(f"   {action.street.capitalize()}: {action.action_type.upper()} ${action.amount}")
    
    # Updated model
    print("\n   📈 Updated Opponent Model:")
    opp1_summary = agent.get_opponent_summary("opponent_1")
    model_summary = opp1_summary['model_summary']
    print(f"      Player Type: {model_summary['player_type']}")
    print(f"      Hands Observed: {model_summary['num_hands_observed']}")
    print(f"      Range Confidence: {model_summary['range_confidence']:.2%}")
    
    # Hand result with showdown
    agent.record_hand_result(
        hand_id_2, "opponent_1", "showdown", 50.0,
        showdown_cards=("Ah", "Kd")
    )
    print(f"\n   💰 Result: Won ${50.0} (Showdown: Ah Kd)")
    
    # Hand 3: Opponent 2 - Loose player
    print("\n" + "=" * 80)
    print("📊 HAND 3: Opponent 2 (Different player type)")
    print("-" * 80)
    
    hand_id_3 = "hand_003"
    
    # Loose player actions
    actions_hand3 = [
        HandAction("call", 5.0, "preflop", "early", 10.0),
        HandAction("call", 20.0, "flop", "early", 40.0),
        HandAction("call", 30.0, "turn", "early", 80.0),
    ]
    
    for action in actions_hand3:
        agent.observe_action("opponent_2", hand_id_3, action)
        print(f"   {action.street.capitalize()}: {action.action_type.upper()} ${action.amount}")
    
    # Opponent 2 model
    opp2_model = agent.get_opponent_model("opponent_2")
    print(f"\n   📈 Opponent 2 Model:")
    print(f"      Player Type: {opp2_model.player_type}")
    
    # Our decision against opponent 2
    game_state_2 = GameState(
        street="river",
        pot_size=140.0,
        bet_to_call=0.0,
        position="late",
        stack_size=1000.0,
        opponent_stack=1000.0
    )
    
    action_ev_2 = agent.decide_action("opponent_2", game_state_2, "strong")
    print(f"\n   🤖 Our Decision:")
    print(f"      Action: {action_ev_2.action.upper()}")
    print(f"      Expected Value: ${action_ev_2.ev:.2f}")
    
    # Exploitation opportunities
    opportunities = agent.get_exploitation_opportunities("opponent_2", game_state_2)
    if opportunities:
        print(f"\n   🎯 Exploitation Opportunities:")
        for opp in opportunities:
            print(f"      {opp['type']}: {opp['reason']}")
            print(f"         Recommended: {opp['recommended_action'].upper()}")
    
    agent.record_hand_result(hand_id_3, "opponent_2", "showdown", 70.0)
    print(f"\n   💰 Result: Won ${70.0}")
    
    # Performance summary
    print("\n" + "=" * 80)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 80)
    
    metrics = agent.get_performance_metrics()
    print(f"\n   Total Hands: {metrics['total_hands']}")
    print(f"   Total Profit: ${metrics['total_profit']:.2f}")
    print(f"   Avg Profit/Hand: ${metrics['avg_profit_per_hand']:.2f}")
    print(f"   Win Rate: {metrics['win_rate']:.2%}")
    print(f"   GTO Weight: {metrics['gto_weight']:.2f}")
    print(f"   Opponents Tracked: {metrics['num_opponents_tracked']}")
    
    # System statistics
    print("\n" + "=" * 80)
    print("🧠 SYSTEM STATISTICS")
    print("=" * 80)
    
    system_stats = agent.get_system_statistics()
    memory_stats = system_stats['memory']
    print(f"\n   Memory:")
    print(f"      Total Hands Tracked: {memory_stats['total_hands_tracked']}")
    print(f"      Total Opponents: {memory_stats['total_opponents']}")
    print(f"      Total Showdowns: {memory_stats['total_showdowns']}")
    
    # Opponent summaries
    print(f"\n   Opponent Models:")
    for opp_id in opponents:
        summary = agent.get_opponent_summary(opp_id)
        model = summary['model_summary']
        print(f"\n      {opp_id}:")
        print(f"         Type: {model['player_type']}")
        print(f"         VPIP: {model['tendency']['vpip']:.2%}")
        print(f"         PFR: {model['tendency']['pfr']:.2%}")
        print(f"         Aggression: {model['tendency']['aggression_factor']:.2f}")
        print(f"         Range Confidence: {model['range_confidence']:.2%}")
    
    print("\n" + "=" * 80)
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 80)
    
    print("\n📝 Key Takeaways:")
    print("   1. Memory records every opponent action")
    print("   2. Opponent models are built dynamically from memory")
    print("   3. Range estimates update using Bayesian inference")
    print("   4. Best response calculates optimal exploitative strategy")
    print("   5. RL adapts strategy based on outcomes")
    print("\n🎯 This is the core architecture of world-class Poker AI!")


if __name__ == "__main__":
    simulate_poker_session()

