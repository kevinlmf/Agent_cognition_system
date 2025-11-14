"""
Best Response Calculator: Exploitative strategy against opponent model
a_t = argmax_a E_{s ~ b_t(opp)} [R(a, s)]

This module implements:
1. Best response calculation: Optimal action against opponent's range
2. Exploitative play: Exploit opponent's weaknesses
3. GTO baseline: Balance between exploitation and GTO
4. Expected value calculation: EV(a | opponent_range, game_state)
"""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from .opponent_model import OpponentModel
from .range_estimator import RangeEstimator


@dataclass
class GameState:
    """Current poker game state"""
    street: str  # 'preflop', 'flop', 'turn', 'river'
    pot_size: float
    bet_to_call: float
    position: str
    stack_size: float
    opponent_stack: float
    board_cards: List[str] = None  # Community cards
    metadata: Dict[str, Any] = None


@dataclass
class ActionEV:
    """Expected value for an action"""
    action: str
    ev: float
    win_probability: float
    confidence: float
    metadata: Dict[str, Any] = None


class BestResponseCalculator:
    """
    Calculates best response strategy against opponent model
    
    Mathematical Foundation:
    a* = argmax_a E_{s ~ b_t(opp)} [R(a, s)]
    
    Where:
    - b_t(opp) is opponent's range estimate
    - R(a, s) is reward for action a against opponent state s
    - E is expectation over opponent's range
    """
    
    def __init__(self, opponent_model: OpponentModel, 
                 range_estimator: RangeEstimator,
                 gto_weight: float = 0.3):
        """
        Args:
            opponent_model: Opponent's world model
            range_estimator: Current range estimate
            gto_weight: Weight for GTO baseline (0 = pure exploitation, 1 = pure GTO)
        """
        self.opponent_model = opponent_model
        self.range_estimator = range_estimator
        self.gto_weight = gto_weight
    
    def calculate_best_response(self, game_state: GameState,
                                our_hand_strength: str) -> ActionEV:
        """
        Calculate best response action
        
        Args:
            game_state: Current game state
            our_hand_strength: Our hand strength ('premium', 'strong', 'medium', 'weak')
        
        Returns:
            ActionEV with optimal action and EV
        """
        # Get opponent's range
        opponent_range = self.range_estimator.get_range(game_state.street)
        
        # Calculate EV for each action
        action_evs = {}
        
        actions = ['fold', 'call', 'bet', 'raise']
        if game_state.bet_to_call == 0:
            actions.remove('fold')  # Can't fold to no bet
        
        for action in actions:
            ev = self._calculate_action_ev(
                action, game_state, our_hand_strength, opponent_range
            )
            action_evs[action] = ev
        
        # Find best action
        best_action = max(action_evs.keys(), key=lambda a: action_evs[a].ev)
        best_ev = action_evs[best_action]
        
        return best_ev
    
    def _calculate_action_ev(self, action: str, game_state: GameState,
                            our_hand_strength: str,
                            opponent_range: Dict[str, float]) -> ActionEV:
        """
        Calculate expected value for an action
        
        EV(a) = Σ_{s in opponent_range} P(s) * R(a, s)
        """
        # Base EV from opponent range
        ev_exploitative = self._calculate_exploitative_ev(
            action, game_state, our_hand_strength, opponent_range
        )
        
        # GTO baseline EV
        ev_gto = self._calculate_gto_ev(
            action, game_state, our_hand_strength
        )
        
        # Combine: weighted average
        ev_combined = (
            (1.0 - self.gto_weight) * ev_exploitative +
            self.gto_weight * ev_gto
        )
        
        # Calculate win probability
        win_prob = self._estimate_win_probability(
            our_hand_strength, opponent_range, game_state
        )
        
        # Calculate confidence
        range_confidence = self.range_estimator.get_range_confidence()
        
        return ActionEV(
            action=action,
            ev=ev_combined,
            win_probability=win_prob,
            confidence=range_confidence,
            metadata={
                'exploitative_ev': ev_exploitative,
                'gto_ev': ev_gto,
                'opponent_range': opponent_range
            }
        )
    
    def _calculate_exploitative_ev(self, action: str, game_state: GameState,
                                  our_hand_strength: str,
                                  opponent_range: Dict[str, float]) -> float:
        """
        Calculate exploitative EV: exploit opponent's weaknesses
        """
        # Get opponent's predicted action probabilities
        opponent_action_probs = self.opponent_model.predict_action(
            game_state.street, game_state.pot_size, game_state.bet_to_call
        )
        
        # Calculate EV against each opponent hand strength
        ev_by_strength = {}
        
        for opp_strength, opp_prob in opponent_range.items():
            # What would opponent do with this hand?
            opp_action = self._predict_opponent_action(
                opp_strength, game_state, opponent_action_probs
            )
            
            # Calculate outcome
            outcome = self._simulate_action_outcome(
                action, opp_action, our_hand_strength, opp_strength,
                game_state
            )
            
            ev_by_strength[opp_strength] = outcome * opp_prob
        
        # Weighted sum
        total_ev = sum(ev_by_strength.values())
        
        return total_ev
    
    def _predict_opponent_action(self, opp_strength: str, game_state: GameState,
                                action_probs: Dict[str, float]) -> str:
        """
        Predict what opponent would do with given hand strength
        """
        # Stronger hands more likely to bet/raise
        # Weaker hands more likely to fold/call
        
        if opp_strength in ['premium', 'strong']:
            # More aggressive
            if np.random.random() < 0.6:
                return 'bet' if game_state.bet_to_call == 0 else 'raise'
            else:
                return 'call'
        
        elif opp_strength == 'medium':
            # Mixed strategy
            rand = np.random.random()
            if rand < action_probs.get('bet', 0.3):
                return 'bet'
            elif rand < action_probs.get('bet', 0.3) + action_probs.get('call', 0.4):
                return 'call'
            else:
                return 'fold'
        
        else:  # weak or bluff
            # More likely to fold
            if np.random.random() < action_probs.get('fold', 0.5):
                return 'fold'
            else:
                return 'call'
    
    def _simulate_action_outcome(self, our_action: str, opp_action: str,
                               our_strength: str, opp_strength: str,
                               game_state: GameState) -> float:
        """
        Simulate outcome of our action against opponent's action
        Returns: Expected chips won/lost
        """
        # Hand strength comparison
        strength_order = {'premium': 4, 'strong': 3, 'medium': 2, 'weak': 1, 'bluff': 0}
        our_strength_val = strength_order.get(our_strength, 2)
        opp_strength_val = strength_order.get(opp_strength, 2)
        
        # Outcome scenarios
        if our_action == 'fold':
            return -game_state.bet_to_call
        
        elif our_action == 'call':
            if opp_action == 'fold':
                return game_state.pot_size  # Win pot
            elif opp_action in ['bet', 'raise']:
                # Showdown
                if our_strength_val > opp_strength_val:
                    return game_state.pot_size + game_state.bet_to_call
                elif our_strength_val < opp_strength_val:
                    return -game_state.bet_to_call
                else:
                    return 0  # Split pot (simplified)
            else:  # opp calls
                # Showdown
                if our_strength_val > opp_strength_val:
                    return game_state.pot_size / 2
                elif our_strength_val < opp_strength_val:
                    return -game_state.pot_size / 2
                else:
                    return 0
        
        elif our_action in ['bet', 'raise']:
            bet_size = self._calculate_bet_size(game_state, our_strength)
            
            if opp_action == 'fold':
                return game_state.pot_size  # Win pot
            elif opp_action == 'call':
                # Showdown
                if our_strength_val > opp_strength_val:
                    return game_state.pot_size + bet_size
                elif our_strength_val < opp_strength_val:
                    return -bet_size
                else:
                    return game_state.pot_size / 2 - bet_size / 2
            else:  # opp raises
                # Re-raise scenario (simplified)
                return -bet_size * 0.5  # Conservative estimate
        
        return 0.0
    
    def _calculate_bet_size(self, game_state: GameState,
                           hand_strength: str) -> float:
        """
        Calculate optimal bet size based on hand strength
        """
        # Bet sizing strategy
        if hand_strength == 'premium':
            # Value bet: large size
            return game_state.pot_size * 0.75
        elif hand_strength == 'strong':
            return game_state.pot_size * 0.6
        elif hand_strength == 'medium':
            return game_state.pot_size * 0.4
        elif hand_strength == 'bluff':
            # Bluff: large size to maximize fold equity
            return game_state.pot_size * 0.7
        else:  # weak
            return game_state.pot_size * 0.3
    
    def _calculate_gto_ev(self, action: str, game_state: GameState,
                         our_hand_strength: str) -> float:
        """
        Calculate GTO baseline EV (simplified)
        """
        # GTO strategy: balanced, unexploitable
        # Simplified: use hand strength and pot odds
        
        strength_ev = {
            'premium': 0.8,
            'strong': 0.5,
            'medium': 0.2,
            'weak': -0.2,
            'bluff': -0.1
        }
        
        base_ev = strength_ev.get(our_hand_strength, 0.0)
        
        # Adjust for action
        if action == 'fold':
            return -game_state.bet_to_call
        elif action == 'call':
            return base_ev * game_state.pot_size * 0.5
        elif action in ['bet', 'raise']:
            return base_ev * game_state.pot_size
        
        return 0.0
    
    def _estimate_win_probability(self, our_strength: str,
                                 opponent_range: Dict[str, float],
                                 game_state: GameState) -> float:
        """
        Estimate win probability against opponent's range
        """
        strength_order = {'premium': 4, 'strong': 3, 'medium': 2, 'weak': 1, 'bluff': 0}
        our_val = strength_order.get(our_strength, 2)
        
        win_prob = 0.0
        
        for opp_strength, opp_prob in opponent_range.items():
            opp_val = strength_order.get(opp_strength, 2)
            
            if our_val > opp_val:
                win_prob += opp_prob * 0.8  # Win most of the time
            elif our_val == opp_val:
                win_prob += opp_prob * 0.5  # 50/50
            else:
                win_prob += opp_prob * 0.2  # Lose most of the time
        
        return win_prob
    
    def get_exploitation_opportunities(self, game_state: GameState) -> List[Dict[str, Any]]:
        """
        Identify exploitation opportunities based on opponent model
        """
        opportunities = []
        
        tendency = self.opponent_model.get_tendency()
        
        # Exploit tight players: bluff more
        if tendency.vpip < 0.25:
            opportunities.append({
                'type': 'bluff_opportunity',
                'reason': 'Opponent is tight, likely to fold to aggression',
                'recommended_action': 'bet',
                'confidence': 0.7
            })
        
        # Exploit loose players: value bet more
        if tendency.vpip > 0.35:
            opportunities.append({
                'type': 'value_opportunity',
                'reason': 'Opponent is loose, will call with weak hands',
                'recommended_action': 'bet',
                'confidence': 0.8
            })
        
        # Exploit passive players: bet more
        if tendency.aggression_factor < 1.0:
            opportunities.append({
                'type': 'aggression_opportunity',
                'reason': 'Opponent is passive, bet for value',
                'recommended_action': 'bet',
                'confidence': 0.75
            })
        
        # Exploit high fold-to-3bet: 3-bet more
        if tendency.fold_to_3bet > 0.6:
            opportunities.append({
                'type': '3bet_opportunity',
                'reason': 'Opponent folds to 3-bets frequently',
                'recommended_action': 'raise',
                'confidence': 0.7
            })
        
        return opportunities

