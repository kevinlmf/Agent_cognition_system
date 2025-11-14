"""
Poker RL Agent: Complete Poker AI system
Integrates Memory → Opponent Model → Range → Best Response → RL Adaptation

Architecture:
Memory (OpponentMemory) 
  ↓
Opponent Model (World Model per opponent)
  ↓
Range Estimator (Bayesian range updates)
  ↓
Best Response Calculator (Exploitative strategy)
  ↓
RL Adaptation (Strategy refinement)
"""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime
from collections import defaultdict

from .opponent_memory import OpponentMemory, HandAction, HandHistory
from .opponent_model import OpponentModel
from .range_estimator import RangeEstimator
from .best_response import BestResponseCalculator, GameState, ActionEV


class PokerRLAgent:
    """
    Complete Poker AI Agent with Memory-based Opponent Modeling
    
    Implements the full pipeline:
    1. Memory: Record every opponent action
    2. Opponent Modeling: Build world model for each opponent
    3. Range Estimation: Bayesian range updates
    4. Best Response: Calculate optimal strategy
    5. RL Adaptation: Learn and improve over time
    """
    
    def __init__(self, agent_id: str = "poker_ai",
                 gto_weight: float = 0.3,
                 learning_rate: float = 0.01):
        """
        Args:
            agent_id: Unique identifier for this agent
            gto_weight: Balance between exploitation (0) and GTO (1)
            learning_rate: Learning rate for RL updates
        """
        self.agent_id = agent_id
        
        # Core components
        self.opponent_memory = OpponentMemory()
        
        # Per-opponent models
        self.opponent_models: Dict[str, OpponentModel] = {}
        self.range_estimators: Dict[str, RangeEstimator] = {}
        self.best_response_calculators: Dict[str, BestResponseCalculator] = {}
        
        # RL components
        self.gto_weight = gto_weight
        self.learning_rate = learning_rate
        
        # Strategy tracking
        self.strategy_history: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.total_hands: int = 0
        self.total_profit: float = 0.0
        self.hand_results: List[Dict[str, Any]] = []
    
    def observe_action(self, opponent_id: str, hand_id: str, action: HandAction):
        """
        Observe and record an opponent action
        
        This is the core memory update:
        Memory → Opponent Model → Range → Best Response
        """
        # 1. Record in memory
        if hand_id not in self.opponent_memory.hand_histories:
            self.opponent_memory.record_hand_start(hand_id, opponent_id)
        
        self.opponent_memory.record_action(hand_id, action)
        
        # 2. Update opponent model (if exists)
        if opponent_id in self.opponent_models:
            self.opponent_models[opponent_id].update_belief(action)
        
        # 3. Update range estimate
        if opponent_id in self.range_estimators:
            self.range_estimators[opponent_id].update_range(action)
    
    def initialize_opponent(self, opponent_id: str):
        """Initialize models for a new opponent"""
        # Create opponent model
        opponent_model = OpponentModel(opponent_id, self.opponent_memory)
        self.opponent_models[opponent_id] = opponent_model
        
        # Create range estimator
        range_estimator = RangeEstimator(opponent_model)
        self.range_estimators[opponent_id] = range_estimator
        
        # Create best response calculator
        best_response = BestResponseCalculator(
            opponent_model, range_estimator, gto_weight=self.gto_weight
        )
        self.best_response_calculators[opponent_id] = best_response
    
    def get_opponent_model(self, opponent_id: str) -> OpponentModel:
        """Get or create opponent model"""
        if opponent_id not in self.opponent_models:
            self.initialize_opponent(opponent_id)
        return self.opponent_models[opponent_id]
    
    def decide_action(self, opponent_id: str, game_state: GameState,
                     our_hand_strength: str) -> ActionEV:
        """
        Decide optimal action using full pipeline
        
        Pipeline:
        1. Get opponent model (world model)
        2. Get range estimate
        3. Calculate best response
        4. Return action with EV
        """
        # Ensure opponent models exist
        if opponent_id not in self.opponent_models:
            self.initialize_opponent(opponent_id)
        
        # Get best response calculator
        calculator = self.best_response_calculators[opponent_id]
        
        # Calculate best response
        action_ev = calculator.calculate_best_response(
            game_state, our_hand_strength
        )
        
        # Record decision
        self.strategy_history.append({
            'opponent_id': opponent_id,
            'game_state': {
                'street': game_state.street,
                'pot_size': game_state.pot_size,
                'bet_to_call': game_state.bet_to_call
            },
            'our_hand_strength': our_hand_strength,
            'action': action_ev.action,
            'ev': action_ev.ev,
            'win_probability': action_ev.win_probability,
            'timestamp': datetime.now().isoformat()
        })
        
        return action_ev
    
    def record_hand_result(self, hand_id: str, opponent_id: str,
                          final_action: str, pot_outcome: float,
                          showdown_cards: Optional[Tuple[str, str]] = None):
        """
        Record hand result and update models
        
        This triggers:
        1. Memory update
        2. Model recalibration
        3. RL learning
        """
        # Record in memory
        self.opponent_memory.record_hand_end(hand_id, final_action, pot_outcome)
        
        if showdown_cards:
            self.opponent_memory.record_showdown(
                hand_id, opponent_id, showdown_cards
            )
        
        # Update statistics
        self.total_hands += 1
        self.total_profit += pot_outcome
        
        self.hand_results.append({
            'hand_id': hand_id,
            'opponent_id': opponent_id,
            'final_action': final_action,
            'pot_outcome': pot_outcome,
            'timestamp': datetime.now().isoformat()
        })
        
        # Trigger model updates
        if opponent_id in self.opponent_models:
            # Recalibrate range with new showdown data
            self.opponent_models[opponent_id]._update_range()
        
        # RL learning: Update strategy based on outcome
        self._rl_update(opponent_id, pot_outcome)
    
    def _rl_update(self, opponent_id: str, reward: float):
        """
        RL update: Learn from outcomes
        
        Updates:
        1. GTO weight (balance exploitation vs GTO)
        2. Bet sizing strategy
        3. Action selection probabilities
        """
        # Simple RL: Adjust GTO weight based on performance
        # If exploiting works well, reduce GTO weight
        # If exploiting fails, increase GTO weight
        
        recent_results = [r for r in self.hand_results[-10:] 
                        if r['opponent_id'] == opponent_id]
        
        if len(recent_results) >= 5:
            avg_reward = np.mean([r['pot_outcome'] for r in recent_results])
            
            if avg_reward > 0:
                # Exploitation working: reduce GTO weight slightly
                self.gto_weight = max(0.1, self.gto_weight - self.learning_rate * 0.1)
            else:
                # Exploitation failing: increase GTO weight
                self.gto_weight = min(0.7, self.gto_weight + self.learning_rate * 0.1)
            
            # Update calculator with new GTO weight
            if opponent_id in self.best_response_calculators:
                self.best_response_calculators[opponent_id].gto_weight = self.gto_weight
    
    def get_exploitation_opportunities(self, opponent_id: str,
                                      game_state: GameState) -> List[Dict[str, Any]]:
        """Get exploitation opportunities for an opponent"""
        if opponent_id not in self.best_response_calculators:
            return []
        
        calculator = self.best_response_calculators[opponent_id]
        return calculator.get_exploitation_opportunities(game_state)
    
    def get_opponent_summary(self, opponent_id: str) -> Dict[str, Any]:
        """Get comprehensive summary for an opponent"""
        if opponent_id not in self.opponent_models:
            return {'error': 'Opponent not found'}
        
        model = self.opponent_models[opponent_id]
        range_est = self.range_estimators.get(opponent_id)
        
        summary = {
            'opponent_id': opponent_id,
            'model_summary': model.get_model_summary(),
            'range_estimate': range_est.get_range_summary() if range_est else None,
            'num_hands_observed': len(self.opponent_memory.get_opponent_history(opponent_id)),
            'memory_stats': self.opponent_memory.get_opponent_stats(opponent_id)
        }
        
        return summary
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get overall performance metrics"""
        if not self.hand_results:
            return {
                'total_hands': 0,
                'total_profit': 0.0,
                'avg_profit_per_hand': 0.0,
                'win_rate': 0.0
            }
        
        wins = sum(1 for r in self.hand_results if r['pot_outcome'] > 0)
        losses = sum(1 for r in self.hand_results if r['pot_outcome'] < 0)
        
        return {
            'total_hands': self.total_hands,
            'total_profit': self.total_profit,
            'avg_profit_per_hand': self.total_profit / self.total_hands,
            'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0.0,
            'wins': wins,
            'losses': losses,
            'gto_weight': self.gto_weight,
            'num_opponents_tracked': len(self.opponent_models)
        }
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'memory': self.opponent_memory.get_statistics(),
            'performance': self.get_performance_metrics(),
            'opponents': {
                opp_id: self.get_opponent_summary(opp_id)
                for opp_id in self.opponent_models.keys()
            }
        }
    
    def save_state(self, filepath: str):
        """Save agent state (simplified - would need full serialization)"""
        # In production, would serialize all models and memory
        state = {
            'agent_id': self.agent_id,
            'gto_weight': self.gto_weight,
            'total_hands': self.total_hands,
            'total_profit': self.total_profit,
            'opponent_ids': list(self.opponent_models.keys())
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load agent state"""
        import json
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.agent_id = state.get('agent_id', self.agent_id)
        self.gto_weight = state.get('gto_weight', self.gto_weight)
        self.total_hands = state.get('total_hands', 0)
        self.total_profit = state.get('total_profit', 0.0)
        
        # Reinitialize opponents
        for opp_id in state.get('opponent_ids', []):
            self.initialize_opponent(opp_id)

