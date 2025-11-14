"""
Range Estimator: Bayesian range estimation and dynamic updates
P(cards | a_1:t, player_type)

This module implements:
1. Bayesian range updates: Range_t = f(Range_{t-1}, a_t)
2. Street-by-street range narrowing
3. Position-aware range estimation
4. Integration with opponent model
"""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import defaultdict

from .opponent_model import OpponentModel
from .opponent_memory import HandAction


class RangeEstimator:
    """
    Estimates opponent's hand range using Bayesian updating
    
    Mathematical Foundation:
    P(cards | a_1:t, player_type) = 
        P(a_t | cards) * P(cards | a_1:t-1, player_type) / P(a_t | a_1:t-1)
    
    Implements:
    - Dynamic range updates as actions observed
    - Street-by-street narrowing
    - Position and context awareness
    """
    
    def __init__(self, opponent_model: OpponentModel):
        self.opponent_model = opponent_model
        
        # Current range estimate (probability distribution)
        self.current_range: Dict[str, float] = {}
        
        # Range history (for debugging/analysis)
        self.range_history: List[Dict[str, float]] = []
        
        # Street-specific ranges
        self.street_ranges: Dict[str, Dict[str, float]] = {}
    
    def initialize_range(self, street: str = 'preflop', 
                        position: Optional[str] = None):
        """
        Initialize range estimate based on opponent model
        
        Uses prior from opponent model's player type
        """
        # Get prior from opponent model
        self.current_range = self.opponent_model._get_prior_range()
        
        # Adjust based on position if provided
        if position:
            self.current_range = self._adjust_for_position(
                self.current_range, position
            )
        
        # Store street-specific range
        self.street_ranges[street] = self.current_range.copy()
        
        # Record in history
        self.range_history.append({
            'street': street,
            'range': self.current_range.copy(),
            'action': 'initialization'
        })
    
    def update_range(self, action: HandAction):
        """
        Bayesian range update: Range_t = f(Range_{t-1}, a_t)
        
        P(cards | a_1:t) ∝ P(a_t | cards) * P(cards | a_1:t-1)
        """
        if not self.current_range:
            self.initialize_range(action.street)
        
        # Get likelihood: P(action | hand_strength)
        likelihoods = self.opponent_model._get_action_likelihood(action)
        
        # Bayesian update
        updated_range = {}
        total = 0.0
        
        for strength in ['premium', 'strong', 'medium', 'weak', 'bluff']:
            likelihood = likelihoods.get(strength, 0.1)
            prior = self.current_range.get(strength, 0.2)
            posterior = likelihood * prior
            updated_range[strength] = posterior
            total += posterior
        
        # Normalize
        for strength in updated_range:
            updated_range[strength] /= total
        
        # Update current range
        self.current_range = updated_range
        
        # Update street-specific range
        self.street_ranges[action.street] = self.current_range.copy()
        
        # Record in history
        self.range_history.append({
            'street': action.street,
            'range': self.current_range.copy(),
            'action': action.action_type,
            'action_amount': action.amount
        })
    
    def narrow_range_for_street(self, new_street: str):
        """
        Narrow range when transitioning to new street
        (e.g., preflop -> flop)
        
        Logic: Some hand strengths become less likely on certain boards
        """
        if not self.current_range:
            return
        
        # Get narrowing factors based on street
        narrowing_factors = self._get_street_narrowing_factors(new_street)
        
        # Apply narrowing
        narrowed_range = {}
        total = 0.0
        
        for strength in ['premium', 'strong', 'medium', 'weak', 'bluff']:
            factor = narrowing_factors.get(strength, 1.0)
            narrowed_range[strength] = (
                self.current_range.get(strength, 0.2) * factor
            )
            total += narrowed_range[strength]
        
        # Normalize
        for strength in narrowed_range:
            narrowed_range[strength] /= total
        
        self.current_range = narrowed_range
        self.street_ranges[new_street] = self.current_range.copy()
    
    def _get_street_narrowing_factors(self, street: str) -> Dict[str, float]:
        """
        Factors to apply when narrowing range for new street
        """
        if street == 'flop':
            # Premium hands stay strong, weak hands less likely
            return {
                'premium': 1.2,  # More likely to continue
                'strong': 1.1,
                'medium': 1.0,
                'weak': 0.7,     # Less likely to continue
                'bluff': 0.8
            }
        elif street == 'turn':
            # Further narrowing
            return {
                'premium': 1.1,
                'strong': 1.0,
                'medium': 0.9,
                'weak': 0.5,
                'bluff': 0.6
            }
        elif street == 'river':
            # Final narrowing
            return {
                'premium': 1.0,
                'strong': 0.95,
                'medium': 0.8,
                'weak': 0.3,
                'bluff': 0.4
            }
        else:
            return {s: 1.0 for s in ['premium', 'strong', 'medium', 'weak', 'bluff']}
    
    def _adjust_for_position(self, range_dist: Dict[str, float], 
                            position: str) -> Dict[str, float]:
        """
        Adjust range based on position
        Early position: tighter range
        Late position: wider range
        """
        if position == 'early':
            # Tighter: premium hands more likely
            factors = {
                'premium': 1.2,
                'strong': 1.1,
                'medium': 0.9,
                'weak': 0.7,
                'bluff': 0.6
            }
        elif position == 'late':
            # Wider: more bluffs and medium hands
            factors = {
                'premium': 0.9,
                'strong': 0.95,
                'medium': 1.1,
                'weak': 1.0,
                'bluff': 1.2
            }
        else:
            # Middle/blinds: no adjustment
            return range_dist
        
        adjusted = {}
        total = 0.0
        
        for strength in range_dist:
            adjusted[strength] = range_dist[strength] * factors.get(strength, 1.0)
            total += adjusted[strength]
        
        # Normalize
        for strength in adjusted:
            adjusted[strength] /= total
        
        return adjusted
    
    def get_range(self, street: Optional[str] = None) -> Dict[str, float]:
        """
        Get current range estimate
        If street specified, return street-specific range
        """
        if street and street in self.street_ranges:
            return self.street_ranges[street]
        return self.current_range.copy()
    
    def get_range_confidence(self) -> float:
        """
        Calculate confidence in current range estimate
        Based on number of actions observed and consistency
        """
        if not self.range_history:
            return 0.0
        
        # More actions = higher confidence (up to a point)
        num_actions = len(self.range_history)
        action_confidence = min(1.0, num_actions / 10.0)
        
        # Consistency: how much has range changed?
        if len(self.range_history) > 1:
            # Compare recent ranges
            recent_ranges = self.range_history[-5:]
            consistency = self._calculate_consistency(recent_ranges)
        else:
            consistency = 0.5
        
        # Combined confidence
        confidence = 0.6 * action_confidence + 0.4 * consistency
        
        return confidence
    
    def _calculate_consistency(self, ranges: List[Dict[str, float]]) -> float:
        """Calculate consistency across range estimates"""
        if len(ranges) < 2:
            return 0.5
        
        # Calculate variance in each strength category
        variances = []
        for strength in ['premium', 'strong', 'medium', 'weak', 'bluff']:
            values = [r.get(strength, 0.2) for r in ranges]
            variance = np.var(values)
            variances.append(variance)
        
        # Lower variance = higher consistency
        avg_variance = np.mean(variances)
        consistency = 1.0 / (1.0 + avg_variance * 10)
        
        return consistency
    
    def get_hand_strength_probability(self, strength: str) -> float:
        """Get probability of a specific hand strength"""
        return self.current_range.get(strength, 0.0)
    
    def get_range_summary(self) -> Dict[str, Any]:
        """Get comprehensive range summary"""
        return {
            'current_range': self.current_range,
            'confidence': self.get_range_confidence(),
            'street_ranges': self.street_ranges,
            'num_updates': len(self.range_history),
            'opponent_type': self.opponent_model.player_type
        }

