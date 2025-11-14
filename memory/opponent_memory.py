"""
Opponent Memory: Records every hand action for each opponent
Core component for Poker AI - builds the foundation for opponent modeling

Mathematical Foundation:
P(opponent strategy | history of actions) = b_t(opponent)

This module stores:
- Action history: bet/call/raise/fold sequences
- Betting patterns: frequencies, sizes, timing
- Showdown results: actual cards when revealed
- Context: position, pot size, street (pre-flop/flop/turn/river)
"""
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import numpy as np
from dataclasses import dataclass, field


@dataclass
class HandAction:
    """Single action in a poker hand"""
    action_type: str  # 'fold', 'call', 'check', 'bet', 'raise', 'all_in'
    amount: float  # Bet/raise amount (0 for check/call/fold)
    street: str  # 'preflop', 'flop', 'turn', 'river'
    position: str  # 'early', 'middle', 'late', 'blinds'
    pot_size: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HandHistory:
    """Complete history of one hand"""
    hand_id: str
    opponent_id: str
    actions: List[HandAction] = field(default_factory=list)
    showdown_cards: Optional[Tuple[str, str]] = None  # (card1, card2) if revealed
    final_action: Optional[str] = None  # 'fold', 'showdown', 'win', 'lose'
    pot_outcome: float = 0.0  # Net chips won/lost
    timestamp: datetime = field(default_factory=datetime.now)
    
    def add_action(self, action: HandAction):
        """Add an action to this hand"""
        self.actions.append(action)
    
    def get_action_sequence(self) -> List[str]:
        """Get sequence of action types"""
        return [a.action_type for a in self.actions]
    
    def get_betting_sequence(self) -> List[float]:
        """Get sequence of bet amounts"""
        return [a.amount for a in self.actions]


class OpponentMemory:
    """
    Memory system for tracking opponent behavior
    
    Stores:
    1. Action history: Every action from every hand
    2. Betting patterns: Aggregated statistics
    3. Showdown data: Actual cards when revealed (for range estimation)
    4. Tendency profiles: Pre-computed player type indicators
    """
    
    def __init__(self):
        # Core storage: hand_id -> HandHistory
        self.hand_histories: Dict[str, HandHistory] = {}
        
        # Indexed by opponent_id
        self.opponent_hands: Dict[str, List[str]] = defaultdict(list)
        
        # Aggregated statistics per opponent
        self.opponent_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Showdown data: (opponent_id, hand_id) -> cards
        self.showdown_data: Dict[Tuple[str, str], Tuple[str, str]] = {}
    
    def record_hand_start(self, hand_id: str, opponent_id: str) -> HandHistory:
        """Start tracking a new hand"""
        history = HandHistory(hand_id=hand_id, opponent_id=opponent_id)
        self.hand_histories[hand_id] = history
        self.opponent_hands[opponent_id].append(hand_id)
        return history
    
    def record_action(self, hand_id: str, action: HandAction):
        """Record an action in a hand"""
        if hand_id not in self.hand_histories:
            raise ValueError(f"Hand {hand_id} not found. Call record_hand_start first.")
        
        self.hand_histories[hand_id].add_action(action)
    
    def record_showdown(self, hand_id: str, opponent_id: str, cards: Tuple[str, str]):
        """Record showdown cards (when opponent reveals their hand)"""
        if hand_id not in self.hand_histories:
            return
        
        self.hand_histories[hand_id].showdown_cards = cards
        self.hand_histories[hand_id].final_action = 'showdown'
        self.showdown_data[(opponent_id, hand_id)] = cards
    
    def record_hand_end(self, hand_id: str, final_action: str, pot_outcome: float = 0.0):
        """Mark hand as complete"""
        if hand_id not in self.hand_histories:
            return
        
        self.hand_histories[hand_id].final_action = final_action
        self.hand_histories[hand_id].pot_outcome = pot_outcome
        
        # Update opponent statistics
        opponent_id = self.hand_histories[hand_id].opponent_id
        self._update_opponent_stats(opponent_id)
    
    def _update_opponent_stats(self, opponent_id: str):
        """Update aggregated statistics for an opponent"""
        hand_ids = self.opponent_hands[opponent_id]
        if not hand_ids:
            return
        
        stats = {
            'total_hands': len(hand_ids),
            'vpip': 0.0,  # Voluntarily Put money In Pot
            'pfr': 0.0,   # Pre-Flop Raise
            'af': 0.0,    # Aggression Factor (bets+raises) / calls
            'cbet': 0.0,  # Continuation bet frequency
            'fold_to_3bet': 0.0,
            'bluff_frequency': 0.0,
            'showdown_wins': 0,
            'showdown_losses': 0,
        }
        
        # Count actions
        total_bets = 0
        total_calls = 0
        total_raises = 0
        total_folds = 0
        preflop_raises = 0
        vpip_count = 0
        cbet_count = 0
        cbet_opportunities = 0
        fold_to_3bet_count = 0
        fold_to_3bet_opportunities = 0
        
        for hand_id in hand_ids:
            history = self.hand_histories[hand_id]
            actions = history.actions
            
            if not actions:
                continue
            
            # VPIP: Did they put money in pre-flop?
            preflop_actions = [a for a in actions if a.street == 'preflop']
            if preflop_actions:
                if any(a.action_type in ['call', 'bet', 'raise'] for a in preflop_actions):
                    vpip_count += 1
                
                # PFR: Pre-flop raise
                if any(a.action_type == 'raise' for a in preflop_actions):
                    preflop_raises += 1
            
            # Count action types
            for action in actions:
                if action.action_type == 'bet':
                    total_bets += 1
                elif action.action_type == 'call':
                    total_calls += 1
                elif action.action_type == 'raise':
                    total_raises += 1
                elif action.action_type == 'fold':
                    total_folds += 1
            
            # C-bet: Bet after raising pre-flop
            if preflop_actions and any(a.action_type == 'raise' for a in preflop_actions):
                flop_actions = [a for a in actions if a.street == 'flop']
                if flop_actions:
                    cbet_opportunities += 1
                    if flop_actions[0].action_type == 'bet':
                        cbet_count += 1
            
            # Fold to 3-bet
            if len(preflop_actions) >= 2:
                if preflop_actions[0].action_type == 'raise' and preflop_actions[1].action_type == 'raise':
                    fold_to_3bet_opportunities += 1
                    if len(preflop_actions) > 2 and preflop_actions[2].action_type == 'fold':
                        fold_to_3bet_count += 1
            
            # Showdown results
            if history.final_action == 'showdown':
                if history.pot_outcome > 0:
                    stats['showdown_wins'] += 1
                else:
                    stats['showdown_losses'] += 1
        
        # Calculate percentages
        if stats['total_hands'] > 0:
            stats['vpip'] = vpip_count / stats['total_hands']
            stats['pfr'] = preflop_raises / stats['total_hands']
        
        if total_calls > 0:
            stats['af'] = (total_bets + total_raises) / total_calls
        
        if cbet_opportunities > 0:
            stats['cbet'] = cbet_count / cbet_opportunities
        
        if fold_to_3bet_opportunities > 0:
            stats['fold_to_3bet'] = fold_to_3bet_count / fold_to_3bet_opportunities
        
        # Estimate bluff frequency (simplified: bets that lost at showdown)
        showdown_hands = [h for h in hand_ids 
                         if self.hand_histories[h].final_action == 'showdown']
        bluff_count = 0
        for hand_id in showdown_hands:
            history = self.hand_histories[hand_id]
            if history.pot_outcome < 0:  # Lost at showdown
                # Check if they bet/raised
                if any(a.action_type in ['bet', 'raise'] for a in history.actions):
                    bluff_count += 1
        
        if showdown_hands:
            stats['bluff_frequency'] = bluff_count / len(showdown_hands)
        
        self.opponent_stats[opponent_id] = stats
    
    def get_opponent_history(self, opponent_id: str, 
                            limit: Optional[int] = None) -> List[HandHistory]:
        """Get all hand histories for an opponent"""
        hand_ids = self.opponent_hands[opponent_id]
        if limit:
            hand_ids = hand_ids[-limit:]
        
        return [self.hand_histories[hid] for hid in hand_ids if hid in self.hand_histories]
    
    def get_opponent_stats(self, opponent_id: str) -> Dict[str, Any]:
        """Get aggregated statistics for an opponent"""
        return self.opponent_stats.get(opponent_id, {})
    
    def get_action_frequency(self, opponent_id: str, 
                           action_type: str,
                           street: Optional[str] = None) -> float:
        """Get frequency of a specific action type"""
        histories = self.get_opponent_history(opponent_id)
        if not histories:
            return 0.0
        
        total_actions = 0
        matching_actions = 0
        
        for history in histories:
            for action in history.actions:
                if street is None or action.street == street:
                    total_actions += 1
                    if action.action_type == action_type:
                        matching_actions += 1
        
        return matching_actions / total_actions if total_actions > 0 else 0.0
    
    def get_recent_actions(self, opponent_id: str, n: int = 10) -> List[HandAction]:
        """Get n most recent actions from an opponent"""
        histories = self.get_opponent_history(opponent_id, limit=n*2)
        actions = []
        
        for history in reversed(histories):
            actions.extend(reversed(history.actions))
            if len(actions) >= n:
                break
        
        return actions[:n]
    
    def get_showdown_cards(self, opponent_id: str) -> List[Tuple[str, str]]:
        """Get all revealed cards for an opponent"""
        cards = []
        for (oid, hand_id), card_pair in self.showdown_data.items():
            if oid == opponent_id:
                cards.append(card_pair)
        return cards
    
    def classify_player_type(self, opponent_id: str) -> str:
        """
        Classify opponent into player type based on statistics
        Returns: 'TAG', 'LAG', 'LP', 'Tight-Passive', 'Balanced', 'Unknown'
        """
        stats = self.get_opponent_stats(opponent_id)
        
        if not stats or stats.get('total_hands', 0) < 10:
            return 'Unknown'
        
        vpip = stats.get('vpip', 0.5)
        pfr = stats.get('pfr', 0.0)
        af = stats.get('af', 1.0)
        
        # Tight-Aggressive (TAG)
        if vpip < 0.25 and pfr > 0.15 and af > 1.5:
            return 'TAG'
        
        # Loose-Aggressive (LAG)
        if vpip > 0.35 and pfr > 0.25 and af > 2.0:
            return 'LAG'
        
        # Loose-Passive (LP)
        if vpip > 0.35 and pfr < 0.15 and af < 1.0:
            return 'LP'
        
        # Tight-Passive
        if vpip < 0.25 and pfr < 0.10 and af < 1.0:
            return 'Tight-Passive'
        
        # Balanced (GTO-like)
        if 0.20 < vpip < 0.30 and 0.15 < pfr < 0.25 and 1.0 < af < 2.0:
            return 'Balanced'
        
        return 'Unknown'
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall memory statistics"""
        total_hands = len(self.hand_histories)
        total_opponents = len(self.opponent_hands)
        total_showdowns = len(self.showdown_data)
        
        return {
            'total_hands_tracked': total_hands,
            'total_opponents': total_opponents,
            'total_showdowns': total_showdowns,
            'avg_hands_per_opponent': total_hands / total_opponents if total_opponents > 0 else 0,
            'opponent_ids': list(self.opponent_hands.keys())
        }

