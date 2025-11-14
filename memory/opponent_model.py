"""
Opponent Model: World Model for each opponent
Builds probabilistic models of opponent behavior from memory

Mathematical Foundation:
b_t(opponent) = P(σ | a_1:t)

This module implements:
1. Tendency estimation: Statistical profiles (bluff frequency, aggression, etc.)
2. Range estimation: P(cards | actions, player_type)
3. Policy class estimation: What strategy class does this opponent use?
4. Dynamic belief updates: Bayesian updating as new actions observed
"""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import defaultdict
from dataclasses import dataclass

from .opponent_memory import OpponentMemory, HandAction


@dataclass
class OpponentTendency:
    """Statistical profile of opponent tendencies"""
    vpip: float  # Voluntarily Put money In Pot
    pfr: float   # Pre-Flop Raise
    aggression_factor: float
    cbet_frequency: float
    fold_to_3bet: float
    bluff_frequency: float
    continuation_bet_size: float  # Average bet size as % of pot
    three_bet_frequency: float
    metadata: Dict[str, Any] = None


@dataclass
class OpponentRange:
    """Estimated hand range for opponent"""
    # Range represented as probability distribution over hand combinations
    # For simplicity, we use hand strength categories
    hand_strength_dist: Dict[str, float]  # 'premium', 'strong', 'medium', 'weak', 'bluff'
    confidence: float  # How confident we are in this range (0-1)
    last_updated: str = None


class OpponentModel:
    """
    World Model for a single opponent
    
    Implements:
    - Bayesian belief formation: b_t = P(strategy | history)
    - Range estimation: P(cards | actions, type)
    - Policy class inference: What type of player is this?
    - Dynamic updates: Update beliefs as new data arrives
    """
    
    def __init__(self, opponent_id: str, memory: OpponentMemory):
        self.opponent_id = opponent_id
        self.memory = memory
        
        # Core model components
        self.tendency: Optional[OpponentTendency] = None
        self.current_range: Optional[OpponentRange] = None
        self.player_type: str = "Unknown"
        
        # Bayesian priors (can be updated)
        self.prior_beliefs: Dict[str, float] = {
            'tight': 0.33,
            'loose': 0.33,
            'balanced': 0.34
        }
        
        # Action likelihood models: P(action | hand_strength, street)
        self.action_likelihoods: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Initialize from memory
        self._initialize_from_memory()
    
    def _initialize_from_memory(self):
        """Initialize model from existing memory"""
        stats = self.memory.get_opponent_stats(self.opponent_id)
        
        if stats:
            self.tendency = OpponentTendency(
                vpip=stats.get('vpip', 0.5),
                pfr=stats.get('pfr', 0.0),
                aggression_factor=stats.get('af', 1.0),
                cbet_frequency=stats.get('cbet', 0.5),
                fold_to_3bet=stats.get('fold_to_3bet', 0.5),
                bluff_frequency=stats.get('bluff_frequency', 0.0),
                continuation_bet_size=0.6,  # Default 60% pot
                three_bet_frequency=0.05,  # Default 5%
                metadata=stats
            )
            
            self.player_type = self.memory.classify_player_type(self.opponent_id)
            self._update_range()
        else:
            # Initialize with default values if no stats available
            self.tendency = OpponentTendency(
                vpip=0.5,
                pfr=0.0,
                aggression_factor=1.0,
                cbet_frequency=0.5,
                fold_to_3bet=0.5,
                bluff_frequency=0.0,
                continuation_bet_size=0.6,
                three_bet_frequency=0.05,
                metadata={}
            )
            self.player_type = "Unknown"
            self._update_range()
    
    def update_belief(self, new_action: HandAction):
        """
        Bayesian belief update: b_{t+1} = P(σ | a_1:t, a_{t+1})
        
        Updates:
        1. Tendency statistics
        2. Range estimate
        3. Player type classification
        """
        # Record action in memory
        # (Assuming memory is updated externally)
        
        # Update tendency
        self._update_tendency()
        
        # Update range estimate using Bayesian update
        self._update_range()
        
        # Re-classify player type
        self.player_type = self.memory.classify_player_type(self.opponent_id)
    
    def _update_tendency(self):
        """Update tendency profile from memory"""
        stats = self.memory.get_opponent_stats(self.opponent_id)
        
        if stats:
            self.tendency = OpponentTendency(
                vpip=stats.get('vpip', 0.5),
                pfr=stats.get('pfr', 0.0),
                aggression_factor=stats.get('af', 1.0),
                cbet_frequency=stats.get('cbet', 0.5),
                fold_to_3bet=stats.get('fold_to_3bet', 0.5),
                bluff_frequency=stats.get('bluff_frequency', 0.0),
                continuation_bet_size=self.tendency.continuation_bet_size if self.tendency else 0.6,
                three_bet_frequency=self.tendency.three_bet_frequency if self.tendency else 0.05,
                metadata=stats
            )
    
    def _update_range(self):
        """
        Update hand range estimate: Range_t = f(Range_{t-1}, a_t)
        
        Uses Bayesian updating:
        P(cards | a_1:t) ∝ P(a_t | cards) * P(cards | a_1:t-1)
        """
        # Get recent actions
        recent_actions = self.memory.get_recent_actions(self.opponent_id, n=20)
        
        # Get showdown data for calibration
        showdown_cards = self.memory.get_showdown_cards(self.opponent_id)
        
        # Initialize range based on player type
        range_dist = self._get_prior_range()
        
        # Update based on actions
        for action in recent_actions:
            range_dist = self._bayesian_range_update(range_dist, action)
        
        # Calibrate using showdown data
        if showdown_cards:
            range_dist = self._calibrate_with_showdowns(range_dist, showdown_cards)
        
        # Calculate confidence (more data = higher confidence)
        num_hands = len(self.memory.get_opponent_history(self.opponent_id))
        confidence = min(1.0, num_hands / 50.0)  # Max confidence at 50 hands
        
        self.current_range = OpponentRange(
            hand_strength_dist=range_dist,
            confidence=confidence
        )
    
    def _get_prior_range(self) -> Dict[str, float]:
        """Get prior range distribution based on player type"""
        if self.player_type == 'TAG':
            # Tight-Aggressive: Premium hands more likely
            return {
                'premium': 0.15,
                'strong': 0.25,
                'medium': 0.30,
                'weak': 0.20,
                'bluff': 0.10
            }
        elif self.player_type == 'LAG':
            # Loose-Aggressive: Wider range, more bluffs
            return {
                'premium': 0.10,
                'strong': 0.20,
                'medium': 0.30,
                'weak': 0.25,
                'bluff': 0.15
            }
        elif self.player_type == 'LP':
            # Loose-Passive: Weak hands, few bluffs
            return {
                'premium': 0.10,
                'strong': 0.15,
                'medium': 0.35,
                'weak': 0.35,
                'bluff': 0.05
            }
        else:
            # Balanced/Unknown: Uniform prior
            return {
                'premium': 0.12,
                'strong': 0.20,
                'medium': 0.30,
                'weak': 0.28,
                'bluff': 0.10
            }
    
    def _bayesian_range_update(self, current_range: Dict[str, float], 
                               action: HandAction) -> Dict[str, float]:
        """
        Bayesian update: P(cards | a_1:t) ∝ P(a_t | cards) * P(cards | a_1:t-1)
        """
        # Likelihood: P(action | hand_strength)
        likelihoods = self._get_action_likelihood(action)
        
        # Update: posterior ∝ likelihood * prior
        updated_range = {}
        total = 0.0
        
        for strength in ['premium', 'strong', 'medium', 'weak', 'bluff']:
            likelihood = likelihoods.get(strength, 0.1)  # Default if unknown
            prior = current_range.get(strength, 0.2)
            posterior = likelihood * prior
            updated_range[strength] = posterior
            total += posterior
        
        # Normalize
        for strength in updated_range:
            updated_range[strength] /= total
        
        return updated_range
    
    def _get_action_likelihood(self, action: HandAction) -> Dict[str, float]:
        """
        P(action | hand_strength)
        Returns likelihood for each hand strength category
        """
        action_type = action.action_type
        street = action.street
        
        # Define likelihoods based on action type and street
        if action_type == 'fold':
            # More likely with weak hands
            return {
                'premium': 0.01,
                'strong': 0.05,
                'medium': 0.20,
                'weak': 0.50,
                'bluff': 0.24
            }
        
        elif action_type == 'call':
            # Medium hands more likely
            return {
                'premium': 0.10,
                'strong': 0.25,
                'medium': 0.40,
                'weak': 0.20,
                'bluff': 0.05
            }
        
        elif action_type == 'bet' or action_type == 'raise':
            # Bet size matters
            bet_size_pct = action.amount / action.pot_size if action.pot_size > 0 else 0.5
            
            if bet_size_pct > 0.8:  # Large bet
                # Premium hands or bluffs
                return {
                    'premium': 0.35,
                    'strong': 0.20,
                    'medium': 0.10,
                    'weak': 0.05,
                    'bluff': 0.30
                }
            elif bet_size_pct > 0.5:  # Medium bet
                return {
                    'premium': 0.25,
                    'strong': 0.30,
                    'medium': 0.25,
                    'weak': 0.10,
                    'bluff': 0.10
                }
            else:  # Small bet
                return {
                    'premium': 0.15,
                    'strong': 0.25,
                    'medium': 0.35,
                    'weak': 0.15,
                    'bluff': 0.10
                }
        
        elif action_type == 'check':
            # Weak to medium hands
            return {
                'premium': 0.05,
                'strong': 0.15,
                'medium': 0.40,
                'weak': 0.30,
                'bluff': 0.10
            }
        
        else:
            # Default uniform
            return {s: 0.2 for s in ['premium', 'strong', 'medium', 'weak', 'bluff']}
    
    def _calibrate_with_showdowns(self, current_range: Dict[str, float],
                                  showdown_cards: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Calibrate range using actual showdown data
        This is crucial for accurate range estimation
        """
        # Count hand strengths from showdowns
        strength_counts = defaultdict(int)
        
        for cards in showdown_cards:
            # Estimate hand strength from cards (simplified)
            strength = self._estimate_hand_strength(cards)
            strength_counts[strength] += 1
        
        # Blend with current range
        total_showdowns = len(showdown_cards)
        if total_showdowns == 0:
            return current_range
        
        # Weight: more showdowns = more weight on empirical data
        empirical_weight = min(0.5, total_showdowns / 20.0)
        prior_weight = 1.0 - empirical_weight
        
        calibrated_range = {}
        for strength in ['premium', 'strong', 'medium', 'weak', 'bluff']:
            empirical_prob = strength_counts[strength] / total_showdowns
            prior_prob = current_range.get(strength, 0.2)
            calibrated_range[strength] = (
                empirical_weight * empirical_prob + 
                prior_weight * prior_prob
            )
        
        # Normalize
        total = sum(calibrated_range.values())
        for strength in calibrated_range:
            calibrated_range[strength] /= total
        
        return calibrated_range
    
    def _estimate_hand_strength(self, cards: Tuple[str, str]) -> str:
        """
        Estimate hand strength category from cards
        Simplified version - real implementation would use proper hand evaluation
        """
        # This is a placeholder - real implementation would:
        # 1. Parse card ranks and suits
        # 2. Evaluate hand strength (pair, two pair, etc.)
        # 3. Map to strength category
        
        # For now, return medium as default
        return 'medium'
    
    def get_range_estimate(self) -> OpponentRange:
        """Get current range estimate"""
        if self.current_range is None:
            self._update_range()
        return self.current_range
    
    def get_tendency(self) -> OpponentTendency:
        """Get current tendency profile"""
        if self.tendency is None:
            self._update_tendency()
        return self.tendency
    
    def predict_action(self, street: str, pot_size: float, 
                     bet_to_call: float = 0.0) -> Dict[str, float]:
        """
        Predict opponent's action probabilities
        Returns: P(action | current state, opponent model)
        """
        if self.tendency is None:
            return {'fold': 0.33, 'call': 0.33, 'bet': 0.34}
        
        # Base probabilities from tendency
        probs = {}
        
        # Fold probability
        if bet_to_call > 0:
            # More likely to fold if facing large bet
            fold_base = 0.3 if self.player_type == 'LP' else 0.2
            fold_prob = fold_base * (1.0 + bet_to_call / pot_size)
            probs['fold'] = min(0.7, fold_prob)
        else:
            probs['fold'] = 0.0  # Can't fold to no bet
        
        # Call probability
        call_base = 0.4 if self.player_type == 'LP' else 0.3
        probs['call'] = call_base
        
        # Bet/raise probability
        if self.tendency.aggression_factor > 1.5:
            bet_prob = 0.4
        else:
            bet_prob = 0.2
        
        probs['bet'] = bet_prob
        probs['raise'] = bet_prob * 0.3  # Raises less frequent than bets
        
        # Normalize
        total = sum(probs.values())
        for action in probs:
            probs[action] /= total
        
        return probs
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        range_est = self.get_range_estimate()
        tendency = self.get_tendency()
        
        return {
            'opponent_id': self.opponent_id,
            'player_type': self.player_type,
            'tendency': {
                'vpip': tendency.vpip,
                'pfr': tendency.pfr,
                'aggression_factor': tendency.aggression_factor,
                'bluff_frequency': tendency.bluff_frequency,
                'cbet_frequency': tendency.cbet_frequency
            },
            'range_estimate': range_est.hand_strength_dist,
            'range_confidence': range_est.confidence,
            'num_hands_observed': len(self.memory.get_opponent_history(self.opponent_id))
        }

