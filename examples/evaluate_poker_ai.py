"""
Poker AI Evaluation Framework
评估Poker AI系统的完整框架

评估维度：
1. Opponent Modeling准确度
2. Range Estimation准确度
3. Best Response策略质量
4. 整体性能（profit, win rate）
5. Memory系统效果
6. 与baseline对比
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
from collections import defaultdict

from memory.poker_agent import PokerRLAgent, GameState
from memory.opponent_memory import HandAction, OpponentMemory
from memory.opponent_model import OpponentModel


class PokerAIEvaluator:
    """Poker AI系统评估器"""
    
    def __init__(self):
        self.metrics = {}
        self.results = {}
    
    def evaluate_opponent_modeling(self, agent: PokerRLAgent,
                                  ground_truth: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        评估Opponent Modeling准确度
        
        Args:
            agent: PokerRLAgent实例
            ground_truth: 真实对手统计信息
                {
                    "opponent_1": {
                        "vpip": 0.25,
                        "pfr": 0.15,
                        "af": 1.8,
                        "player_type": "TAG"
                    },
                    ...
                }
        """
        print("\n" + "="*70)
        print("评估维度 1: Opponent Modeling准确度")
        print("="*70)
        
        modeling_errors = {}
        modeling_accuracies = {}
        
        for opp_id, true_stats in ground_truth.items():
            if opp_id not in agent.opponent_models:
                continue
            
            model = agent.opponent_models[opp_id]
            predicted_stats = model.get_tendency()
            
            errors = {}
            accuracies = {}
            
            # VPIP误差
            if 'vpip' in true_stats:
                vpip_error = abs(predicted_stats.vpip - true_stats['vpip'])
                vpip_accuracy = 1.0 - min(1.0, vpip_error / 0.5)  # 归一化
                errors['vpip'] = vpip_error
                accuracies['vpip'] = vpip_accuracy
            
            # PFR误差
            if 'pfr' in true_stats:
                pfr_error = abs(predicted_stats.pfr - true_stats['pfr'])
                pfr_accuracy = 1.0 - min(1.0, pfr_error / 0.5)
                errors['pfr'] = pfr_error
                accuracies['pfr'] = pfr_accuracy
            
            # Aggression Factor误差
            if 'af' in true_stats:
                af_error = abs(predicted_stats.aggression_factor - true_stats['af'])
                af_accuracy = 1.0 - min(1.0, af_error / 3.0)  # AF通常在0-3范围
                errors['af'] = af_error
                accuracies['af'] = af_accuracy
            
            # Player Type准确度
            if 'player_type' in true_stats:
                type_match = 1.0 if model.player_type == true_stats['player_type'] else 0.0
                accuracies['player_type'] = type_match
            
            modeling_errors[opp_id] = errors
            modeling_accuracies[opp_id] = accuracies
            
            print(f"\n对手 {opp_id}:")
            print(f"  真实: VPIP={true_stats.get('vpip', 'N/A'):.2%}, "
                  f"PFR={true_stats.get('pfr', 'N/A'):.2%}, "
                  f"AF={true_stats.get('af', 'N/A'):.2f}, "
                  f"Type={true_stats.get('player_type', 'N/A')}")
            print(f"  预测: VPIP={predicted_stats.vpip:.2%}, "
                  f"PFR={predicted_stats.pfr:.2%}, "
                  f"AF={predicted_stats.aggression_factor:.2f}, "
                  f"Type={model.player_type}")
            print(f"  准确度: {np.mean(list(accuracies.values())):.2%}")
        
        # 计算总体指标
        all_accuracies = []
        for accs in modeling_accuracies.values():
            all_accuracies.extend(accs.values())
        
        avg_accuracy = np.mean(all_accuracies) if all_accuracies else 0.0
        
        result = {
            'opponent_modeling': {
                'average_accuracy': float(avg_accuracy),
                'per_opponent_errors': modeling_errors,
                'per_opponent_accuracies': modeling_accuracies,
                'num_opponents': len(modeling_accuracies)
            }
        }
        
        print(f"\n总体Opponent Modeling准确度: {avg_accuracy:.2%}")
        
        return result
    
    def evaluate_range_estimation(self, agent: PokerRLAgent,
                                   test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        评估Range Estimation准确度
        
        Args:
            agent: PokerRLAgent实例
            test_cases: 测试用例列表
                [
                    {
                        "opponent_id": "opponent_1",
                        "actions": [HandAction, ...],
                        "true_range": {"premium": 0.2, "strong": 0.3, ...},
                        "street": "flop"
                    },
                    ...
                ]
        """
        print("\n" + "="*70)
        print("评估维度 2: Range Estimation准确度")
        print("="*70)
        
        range_errors = []
        range_accuracies = []
        
        for i, test_case in enumerate(test_cases):
            opp_id = test_case['opponent_id']
            true_range = test_case['true_range']
            street = test_case.get('street', 'flop')
            
            if opp_id not in agent.range_estimators:
                continue
            
            range_est = agent.range_estimators[opp_id]
            predicted_range = range_est.get_range(street)
            
            # 计算每个strength的误差
            errors = {}
            for strength in ['premium', 'strong', 'medium', 'weak', 'bluff']:
                if strength in true_range and strength in predicted_range:
                    error = abs(predicted_range[strength] - true_range[strength])
                    errors[strength] = error
            
            # 计算总体误差（L1距离）
            total_error = sum(errors.values())
            accuracy = 1.0 - min(1.0, total_error / 2.0)  # 归一化
            
            range_errors.append(total_error)
            range_accuracies.append(accuracy)
            
            print(f"\n测试用例 {i+1}:")
            print(f"  真实Range: {true_range}")
            print(f"  预测Range: {predicted_range}")
            print(f"  误差: {total_error:.3f}, 准确度: {accuracy:.2%}")
        
        avg_error = np.mean(range_errors) if range_errors else 0.0
        avg_accuracy = np.mean(range_accuracies) if range_accuracies else 0.0
        
        result = {
            'range_estimation': {
                'average_error': float(avg_error),
                'average_accuracy': float(avg_accuracy),
                'num_test_cases': len(test_cases),
                'per_case_errors': range_errors
            }
        }
        
        print(f"\n总体Range Estimation准确度: {avg_accuracy:.2%}")
        print(f"平均误差: {avg_error:.3f}")
        
        return result
    
    def evaluate_best_response(self, agent: PokerRLAgent,
                              game_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        评估Best Response策略质量
        
        Args:
            agent: PokerRLAgent实例
            game_scenarios: 游戏场景列表
                [
                    {
                        "opponent_id": "opponent_1",
                        "game_state": GameState,
                        "our_hand_strength": "medium",
                        "optimal_action": "call",  # 或None（如果未知）
                        "expected_ev": 10.0  # 或None
                    },
                    ...
                ]
        """
        print("\n" + "="*70)
        print("评估维度 3: Best Response策略质量")
        print("="*70)
        
        action_evs = []
        ev_values = []
        
        for i, scenario in enumerate(game_scenarios):
            opp_id = scenario['opponent_id']
            game_state = scenario['game_state']
            our_strength = scenario['our_hand_strength']
            
            if opp_id not in agent.best_response_calculators:
                continue
            
            # 计算best response
            action_ev = agent.decide_action(opp_id, game_state, our_strength)
            
            action_evs.append({
                'action': action_ev.action,
                'ev': action_ev.ev,
                'win_probability': action_ev.win_probability,
                'confidence': action_ev.confidence
            })
            
            ev_values.append(action_ev.ev)
            
            optimal_action = scenario.get('optimal_action')
            if optimal_action:
                action_match = 1.0 if action_ev.action == optimal_action else 0.0
            else:
                action_match = None
            
            print(f"\n场景 {i+1}:")
            print(f"  决策: {action_ev.action.upper()}")
            print(f"  期望价值: ${action_ev.ev:.2f}")
            print(f"  胜率: {action_ev.win_probability:.2%}")
            if optimal_action:
                print(f"  最优动作: {optimal_action.upper()}, 匹配: {'✓' if action_match else '✗'}")
        
        avg_ev = np.mean(ev_values) if ev_values else 0.0
        
        result = {
            'best_response': {
                'average_ev': float(avg_ev),
                'num_scenarios': len(game_scenarios),
                'action_evs': action_evs
            }
        }
        
        print(f"\n平均期望价值: ${avg_ev:.2f}")
        
        return result
    
    def evaluate_performance(self, agent: PokerRLAgent,
                           num_hands: int = 100) -> Dict[str, Any]:
        """
        评估整体性能
        
        Args:
            agent: PokerRLAgent实例
            num_hands: 评估的手数
        """
        print("\n" + "="*70)
        print("评估维度 4: 整体性能")
        print("="*70)
        
        metrics = agent.get_performance_metrics()
        
        result = {
            'performance': {
                'total_hands': metrics['total_hands'],
                'total_profit': float(metrics['total_profit']),
                'avg_profit_per_hand': float(metrics['avg_profit_per_hand']),
                'win_rate': float(metrics['win_rate']),
                'wins': metrics['wins'],
                'losses': metrics['losses'],
                'gto_weight': float(metrics['gto_weight']),
                'num_opponents_tracked': metrics['num_opponents_tracked']
            }
        }
        
        print(f"\n总手数: {metrics['total_hands']}")
        print(f"总利润: ${metrics['total_profit']:.2f}")
        print(f"平均利润/手: ${metrics['avg_profit_per_hand']:.2f}")
        print(f"胜率: {metrics['win_rate']:.2%}")
        print(f"GTO权重: {metrics['gto_weight']:.2f}")
        
        return result
    
    def evaluate_memory_effectiveness(self, agent: PokerRLAgent) -> Dict[str, Any]:
        """
        评估Memory系统效果
        """
        print("\n" + "="*70)
        print("评估维度 5: Memory系统效果")
        print("="*70)
        
        memory_stats = agent.opponent_memory.get_statistics()
        
        # 计算每个对手的记忆覆盖率
        opponent_coverage = {}
        for opp_id in agent.opponent_models.keys():
            history = agent.opponent_memory.get_opponent_history(opp_id)
            stats = agent.opponent_memory.get_opponent_stats(opp_id)
            
            opponent_coverage[opp_id] = {
                'num_hands': len(history),
                'has_stats': len(stats) > 0,
                'num_showdowns': len(agent.opponent_memory.get_showdown_cards(opp_id))
            }
        
        result = {
            'memory_effectiveness': {
                'total_hands_tracked': memory_stats['total_hands_tracked'],
                'total_opponents': memory_stats['total_opponents'],
                'total_showdowns': memory_stats['total_showdowns'],
                'avg_hands_per_opponent': memory_stats['avg_hands_per_opponent'],
                'opponent_coverage': opponent_coverage
            }
        }
        
        print(f"\n总手数跟踪: {memory_stats['total_hands_tracked']}")
        print(f"总对手数: {memory_stats['total_opponents']}")
        print(f"总摊牌数: {memory_stats['total_showdowns']}")
        print(f"平均手数/对手: {memory_stats['avg_hands_per_opponent']:.1f}")
        
        return result
    
    def compare_with_baseline(self, agent: PokerRLAgent,
                            baseline_agent: Any = None) -> Dict[str, Any]:
        """
        与baseline对比
        
        Baseline可以是：
        - Random策略
        - GTO策略
        - 简单统计策略
        """
        print("\n" + "="*70)
        print("评估维度 6: 与Baseline对比")
        print("="*70)
        
        # 如果没有提供baseline，使用随机策略作为baseline
        if baseline_agent is None:
            # 简单的随机baseline
            baseline_profit = np.random.normal(0, 50, 100).sum()  # 模拟随机策略
            baseline_win_rate = 0.5  # 随机策略假设50%胜率
        else:
            baseline_metrics = baseline_agent.get_performance_metrics()
            baseline_profit = baseline_metrics['total_profit']
            baseline_win_rate = baseline_metrics['win_rate']
        
        agent_metrics = agent.get_performance_metrics()
        agent_profit = agent_metrics['total_profit']
        agent_win_rate = agent_metrics['win_rate']
        
        profit_improvement = agent_profit - baseline_profit
        win_rate_improvement = agent_win_rate - baseline_win_rate
        
        result = {
            'baseline_comparison': {
                'baseline_profit': float(baseline_profit),
                'agent_profit': float(agent_profit),
                'profit_improvement': float(profit_improvement),
                'baseline_win_rate': float(baseline_win_rate),
                'agent_win_rate': float(agent_win_rate),
                'win_rate_improvement': float(win_rate_improvement)
            }
        }
        
        print(f"\nBaseline利润: ${baseline_profit:.2f}")
        print(f"Agent利润: ${agent_profit:.2f}")
        print(f"利润提升: ${profit_improvement:.2f} ({profit_improvement/baseline_profit*100:.1f}%)" if baseline_profit != 0 else "N/A")
        print(f"\nBaseline胜率: {baseline_win_rate:.2%}")
        print(f"Agent胜率: {agent_win_rate:.2%}")
        print(f"胜率提升: {win_rate_improvement:.2%}")
        
        return result
    
    def comprehensive_evaluation(self, agent: PokerRLAgent,
                                ground_truth: Optional[Dict] = None,
                                test_cases: Optional[List] = None,
                                game_scenarios: Optional[List] = None) -> Dict[str, Any]:
        """
        综合评估
        
        Returns:
            完整的评估结果字典
        """
        print("\n" + "="*80)
        print("POKER AI 综合评估")
        print("="*80)
        
        results = {}
        
        # 1. Opponent Modeling
        if ground_truth:
            results.update(self.evaluate_opponent_modeling(agent, ground_truth))
        
        # 2. Range Estimation
        if test_cases:
            results.update(self.evaluate_range_estimation(agent, test_cases))
        
        # 3. Best Response
        if game_scenarios:
            results.update(self.evaluate_best_response(agent, game_scenarios))
        
        # 4. Performance
        results.update(self.evaluate_performance(agent))
        
        # 5. Memory Effectiveness
        results.update(self.evaluate_memory_effectiveness(agent))
        
        # 6. Baseline Comparison
        results.update(self.compare_with_baseline(agent))
        
        # 计算综合得分
        scores = []
        if 'opponent_modeling' in results:
            scores.append(results['opponent_modeling']['average_accuracy'])
        if 'range_estimation' in results:
            scores.append(results['range_estimation']['average_accuracy'])
        if 'best_response' in results:
            # 将EV转换为0-1得分（假设最大EV为100）
            ev_score = min(1.0, (results['best_response']['average_ev'] + 50) / 100)
            scores.append(ev_score)
        if 'performance' in results:
            # 胜率作为得分
            scores.append(results['performance']['win_rate'])
        
        overall_score = np.mean(scores) if scores else 0.0
        
        results['overall_score'] = float(overall_score)
        results['evaluation_timestamp'] = datetime.now().isoformat()
        
        print("\n" + "="*80)
        print(f"综合得分: {overall_score:.2%}")
        print("="*80)
        
        return results


def create_test_scenarios() -> Tuple[Dict, List, List]:
    """
    创建测试场景
    """
    # Ground truth对手统计
    ground_truth = {
        "opponent_1": {
            "vpip": 0.25,
            "pfr": 0.15,
            "af": 1.8,
            "player_type": "TAG"
        },
        "opponent_2": {
            "vpip": 0.40,
            "pfr": 0.20,
            "af": 0.8,
            "player_type": "LP"
        }
    }
    
    # Range estimation测试用例
    test_cases = [
        {
            "opponent_id": "opponent_1",
            "actions": [
                HandAction("raise", 20.0, "preflop", "late", 10.0),
                HandAction("bet", 30.0, "flop", "late", 50.0)
            ],
            "true_range": {
                "premium": 0.3,
                "strong": 0.35,
                "medium": 0.25,
                "weak": 0.05,
                "bluff": 0.05
            },
            "street": "flop"
        }
    ]
    
    # Best response游戏场景
    game_scenarios = [
        {
            "opponent_id": "opponent_1",
            "game_state": GameState(
                street="flop",
                pot_size=50.0,
                bet_to_call=30.0,
                position="early",
                stack_size=1000.0,
                opponent_stack=1000.0
            ),
            "our_hand_strength": "medium",
            "optimal_action": "call"
        },
        {
            "opponent_id": "opponent_2",
            "game_state": GameState(
                street="river",
                pot_size=140.0,
                bet_to_call=0.0,
                position="late",
                stack_size=1000.0,
                opponent_stack=1000.0
            ),
            "our_hand_strength": "strong",
            "optimal_action": "bet"
        }
    ]
    
    return ground_truth, test_cases, game_scenarios


def main():
    """主函数：运行评估"""
    print("="*80)
    print("Poker AI 评估框架")
    print("="*80)
    
    # 创建agent并运行一些手牌（模拟）
    print("\n1. 初始化Poker AI Agent...")
    agent = PokerRLAgent(agent_id="eval_agent", gto_weight=0.3)
    
    # 初始化对手
    print("2. 初始化对手...")
    agent.initialize_opponent("opponent_1")
    agent.initialize_opponent("opponent_2")
    
    # 模拟一些手牌来填充memory
    print("3. 模拟手牌以填充Memory...")
    for i in range(20):
        hand_id = f"hand_{i:03d}"
        opp_id = "opponent_1" if i % 2 == 0 else "opponent_2"
        
        # 模拟动作
        actions = [
            HandAction("raise", 20.0, "preflop", "late", 10.0),
            HandAction("bet", 30.0, "flop", "late", 50.0)
        ]
        
        agent.observe_action(opp_id, hand_id, actions[0])
        agent.observe_action(opp_id, hand_id, actions[1])
        
        # 记录结果
        outcome = np.random.choice([-30, 50, 70])
        agent.record_hand_result(hand_id, opp_id, "showdown" if outcome > 0 else "fold", outcome)
    
    # 创建测试场景
    print("4. 创建测试场景...")
    ground_truth, test_cases, game_scenarios = create_test_scenarios()
    
    # 运行评估
    print("5. 运行综合评估...")
    evaluator = PokerAIEvaluator()
    results = evaluator.comprehensive_evaluation(
        agent,
        ground_truth=ground_truth,
        test_cases=test_cases,
        game_scenarios=game_scenarios
    )
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_results/poker_ai_evaluation_{timestamp}.json"
    
    os.makedirs("evaluation_results", exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n评估结果已保存到: {output_file}")
    
    return results


if __name__ == "__main__":
    main()

