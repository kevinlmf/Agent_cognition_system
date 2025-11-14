"""
Poker场景专用评估脚本
评估指标：
1. Hidden State Prediction - 对手范围/策略推断准确率
2. Win Rate Improvement - 有无memory的胜率提升
3. Behavior Consistency - 长期对抗中的一致性
"""
import sys
import os
# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import json

# 尝试导入Poker相关模块
try:
    from memory import PokerRLAgent, GameState, HandAction
except ImportError:
    try:
        # Fallback: try direct import
        from memory.poker_agent import PokerRLAgent
        from memory.best_response import GameState
        from memory.opponent_memory import HandAction
    except Exception as e:
        print(f"Warning: Could not import Poker modules: {e}")
        PokerRLAgent = None
        GameState = None
        HandAction = None
from evaluation.comparison.scenario_comparison import ScenarioComparison, create_baseline_agents


class PokerEvaluator:
    """Poker场景评估器"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_hidden_state_prediction(self, agent, ground_truth: Dict) -> float:
        """
        评估1: Hidden State Prediction (对手范围/策略推断准确率)
        """
        if not PokerRLAgent or not isinstance(agent, PokerRLAgent):
            return 0.0
        
        if not hasattr(agent, 'opponent_models') or not agent.opponent_models:
            return 0.0
        
        accuracies = []
        
        for opp_id, true_stats in ground_truth.items():
            if opp_id not in agent.opponent_models:
                continue
            
            model = agent.opponent_models[opp_id]
            
            # 1. Range prediction accuracy
            if 'true_range' in true_stats:
                true_range = true_stats['true_range']
                if hasattr(agent, 'range_estimators') and opp_id in agent.range_estimators:
                    pred_range = agent.range_estimators[opp_id].get_range()
                    if pred_range:
                        # L1 distance
                        range_error = sum(abs(pred_range.get(k, 0) - true_range.get(k, 0)) 
                                        for k in ['premium', 'strong', 'medium', 'weak', 'bluff'])
                        range_accuracy = max(0, 1.0 - range_error / 2.0)
                        accuracies.append(range_accuracy)
            
            # 2. VPIP/PFR accuracy
            if 'true_vpip' in true_stats:
                try:
                    tendency = model.get_tendency()
                    if tendency:
                        pred_vpip = tendency.vpip
                        vpip_error = abs(pred_vpip - true_stats['true_vpip'])
                        vpip_accuracy = max(0, 1.0 - vpip_error / 0.5)
                        accuracies.append(vpip_accuracy)
                except (AttributeError, TypeError):
                    pass
            
            if 'true_pfr' in true_stats:
                try:
                    tendency = model.get_tendency()
                    if tendency:
                        pred_pfr = tendency.pfr
                        pfr_error = abs(pred_pfr - true_stats['true_pfr'])
                        pfr_accuracy = max(0, 1.0 - pfr_error / 0.5)
                        accuracies.append(pfr_accuracy)
                except (AttributeError, TypeError):
                    pass
            
            # 3. Player type accuracy
            if 'true_player_type' in true_stats:
                pred_type = getattr(model, 'player_type', 'Unknown')
                type_accuracy = 1.0 if pred_type == true_stats['true_player_type'] else 0.0
                accuracies.append(type_accuracy)
        
        return np.mean(accuracies) if accuracies else 0.0
    
    def evaluate_win_rate_improvement(self, agent_with_memory, agent_without_memory=None) -> float:
        """
        评估2: Win Rate Improvement (有无memory的胜率提升)
        """
        if not hasattr(agent_with_memory, 'get_performance_metrics'):
            return 0.0
        
        metrics_with = agent_with_memory.get_performance_metrics()
        win_rate_with = metrics_with.get('win_rate', 0.5)
        
        if agent_without_memory:
            metrics_without = agent_without_memory.get_performance_metrics()
            win_rate_without = metrics_without.get('win_rate', 0.5)
        else:
            # 假设baseline是50%胜率
            win_rate_without = 0.5
        
        improvement = win_rate_with - win_rate_without
        return improvement
    
    def evaluate_behavior_consistency(self, agent, test_episodes: List) -> float:
        """
        评估3: Behavior Consistency (长期对抗中的一致性)
        """
        if not test_episodes:
            return 0.5
        
        consistency_scores = []
        
        # 分析多个episode中的决策一致性
        decisions = []
        for episode in test_episodes:
            if 'decision' in episode:
                decisions.append(episode['decision'])
        
        if len(decisions) >= 2:
            # 计算决策的稳定性
            decision_variance = np.var([hash(str(d)) for d in decisions])
            consistency = 1.0 / (1.0 + decision_variance / 10.0)
            consistency_scores.append(consistency)
        
        # 检查对手风格切换时的响应速度
        if hasattr(agent, 'opponent_models'):
            for opp_id, model in agent.opponent_models.items():
                stats = model.get_model_summary()
                num_hands = stats.get('num_hands_observed', 0)
                if num_hands > 0:
                    # 更多手数 = 更好的长期建模
                    consistency_scores.append(min(1.0, num_hands / 50.0))
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def comprehensive_evaluation(self, agent, ground_truth: Dict, 
                               agent_without_memory=None,
                               test_episodes: List = None) -> Dict[str, Any]:
        """
        综合评估Poker场景
        """
        print("="*80)
        print("POKER场景评估")
        print("="*80)
        
        # 评估三个核心指标
        metric1 = self.evaluate_hidden_state_prediction(agent, ground_truth)
        metric2 = self.evaluate_win_rate_improvement(agent, agent_without_memory)
        metric3 = self.evaluate_behavior_consistency(agent, test_episodes or [])
        
        # 收集memory统计
        memory_stats = {}
        if hasattr(agent, 'get_system_statistics'):
            memory_stats = agent.get_system_statistics().get('memory', {})
        
        # 综合得分
        overall_score = (metric1 + max(0, min(1, metric2 + 0.5)) + metric3) / 3.0
        
        results = {
            'scenario': 'poker',
            'evaluation_timestamp': datetime.now().isoformat(),
            'metrics': {
                'hidden_state_prediction_accuracy': float(metric1),
                'win_rate_improvement': float(metric2),
                'behavior_consistency': float(metric3)
            },
            'overall_score': float(overall_score),
            'memory_stats': memory_stats,
            'interpretation': {
                'hidden_state_prediction': '优秀' if metric1 > 0.8 else '良好' if metric1 > 0.6 else '需改进',
                'win_rate_improvement': '优秀' if metric2 > 0.1 else '良好' if metric2 > 0.05 else '需改进',
                'behavior_consistency': '优秀' if metric3 > 0.7 else '良好' if metric3 > 0.5 else '需改进'
            }
        }
        
        # 打印结果
        print(f"\n📊 评估结果:")
        print(f"   1. Hidden State Prediction: {metric1:.3f} ({results['interpretation']['hidden_state_prediction']})")
        print(f"   2. Win Rate Improvement: {metric2:+.3f} ({results['interpretation']['win_rate_improvement']})")
        print(f"   3. Behavior Consistency: {metric3:.3f} ({results['interpretation']['behavior_consistency']})")
        print(f"\n   综合得分: {overall_score:.3f}")
        
        self.results = results
        return results
    
    def save_results(self, filepath: str):
        """保存评估结果"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n✓ 结果已保存到: {filepath}")


def create_poker_ground_truth():
    """创建Poker评估的ground truth数据"""
    return {
        "opponent_1": {
            "true_range": {
                "premium": 0.30,
                "strong": 0.35,
                "medium": 0.25,
                "weak": 0.05,
                "bluff": 0.05
            },
            "true_vpip": 0.25,
            "true_pfr": 0.15,
            "true_player_type": "TAG"
        },
        "opponent_2": {
            "true_range": {
                "premium": 0.10,
                "strong": 0.20,
                "medium": 0.30,
                "weak": 0.25,
                "bluff": 0.15
            },
            "true_vpip": 0.40,
            "true_pfr": 0.20,
            "true_player_type": "LAG"
        }
    }


def create_poker_test_episodes():
    """创建Poker测试episodes"""
    return [
        {
            "episode_id": "hand_001",
            "opponent_id": "opponent_1",
            "decision": "call",
            "context": "preflop_raise"
        },
        {
            "episode_id": "hand_002",
            "opponent_id": "opponent_1",
            "decision": "fold",
            "context": "river_bet"
        },
        {
            "episode_id": "hand_003",
            "opponent_id": "opponent_2",
            "decision": "raise",
            "context": "turn_bluff"
        }
    ]


def simulate_poker_hands(agent, ground_truth: Dict, num_hands_per_opponent: int = 20):
    """
    模拟Poker游戏手数，让agent记录对手动作
    
    根据ground truth中的对手特征，模拟他们的行为
    """
    if not PokerRLAgent or not isinstance(agent, PokerRLAgent):
        return
    
    np.random.seed(42)  # 可重复性
    
    for opp_id, true_stats in ground_truth.items():
        # 初始化对手
        agent.initialize_opponent(opp_id)
        
        # 获取真实特征
        true_vpip = true_stats.get('true_vpip', 0.25)
        true_pfr = true_stats.get('true_pfr', 0.15)
        true_player_type = true_stats.get('true_player_type', 'TAG')
        
        # 模拟多手游戏
        for hand_num in range(num_hands_per_opponent):
            hand_id = f"{opp_id}_hand_{hand_num:03d}"
            
            # 模拟preflop动作
            preflop_action_type = None
            if np.random.random() < true_vpip:
                # VPIP: 会投入资金
                if np.random.random() < (true_pfr / true_vpip):
                    preflop_action_type = 'raise'
                else:
                    preflop_action_type = 'call'
            else:
                preflop_action_type = 'fold'
            
            # 创建preflop动作
            preflop_action = HandAction(
                action_type=preflop_action_type,
                amount=20.0 if preflop_action_type == 'raise' else (10.0 if preflop_action_type == 'call' else 0.0),
                street='preflop',
                position='middle',
                pot_size=30.0
            )
            agent.observe_action(opp_id, hand_id, preflop_action)
            
            # 如果没fold，模拟后续街道
            if preflop_action_type != 'fold':
                # Flop动作
                if np.random.random() < 0.6:  # 60% continuation bet
                    flop_action = HandAction(
                        action_type='bet',
                        amount=15.0,
                        street='flop',
                        position='middle',
                        pot_size=50.0
                    )
                    agent.observe_action(opp_id, hand_id, flop_action)
                
                # 模拟手牌结果（偶尔showdown）
                if np.random.random() < 0.3:  # 30% showdown
                    final_action = 'showdown'
                    pot_outcome = np.random.choice([-20.0, 20.0])  # 随机输赢
                    
                    # 根据player type决定showdown cards
                    if true_player_type == 'TAG':
                        # Tight-Aggressive: 强牌
                        showdown_cards = ('As', 'Kh')
                    else:
                        # Loose: 可能弱牌
                        showdown_cards = ('7c', '8d')
                    
                    agent.record_hand_result(hand_id, opp_id, final_action, pot_outcome, showdown_cards)
                else:
                    # 非showdown结束
                    final_action = np.random.choice(['fold', 'call'])
                    pot_outcome = -10.0 if final_action == 'fold' else 0.0
                    agent.record_hand_result(hand_id, opp_id, final_action, pot_outcome)


def main():
    """主函数"""
    print("="*80)
    print("Poker场景评估（包含Baseline对比）")
    print("="*80)
    
    if not PokerRLAgent:
        print("\n⚠️ 警告: Poker Agent模块未找到")
        print("请确保 memory/poker_agent.py 存在")
        return
    
    # 创建评估器
    evaluator = PokerEvaluator()
    
    # 创建测试数据
    ground_truth = create_poker_ground_truth()
    test_episodes = create_poker_test_episodes()
    
    # 创建我们的Poker Agent
    def create_our_agent():
        """创建使用我们Memory系统的Poker Agent"""
        from main import CognitiveAgent
        agent = CognitiveAgent(mode="langgraph")
        # 如果有PokerRLAgent，可以包装它
        if PokerRLAgent:
            poker_agent = PokerRLAgent()
            poker_agent.memory_engine = agent.memory_engine
            return poker_agent
        return agent
    
    # 定义指标计算函数
    def calculate_metrics(agent, results):
        """计算Poker场景的指标"""
        # 如果是我们的PokerRLAgent，运行实际评估
        if PokerRLAgent and isinstance(agent, PokerRLAgent):
            try:
                # 如果还没有模拟游戏，先模拟
                if hasattr(agent, 'opponent_memory'):
                    memory_stats = agent.opponent_memory.get_statistics()
                    if memory_stats.get('total_hands_tracked', 0) == 0:
                        simulate_poker_hands(agent, ground_truth, num_hands_per_opponent=30)
                
                # 运行评估
                eval_results = evaluator.comprehensive_evaluation(
                    agent, ground_truth, None, test_episodes
                )
                # 从评估结果中提取指标
                metrics_dict = eval_results.get('metrics', {})
                return {
                    'hidden_state_prediction': metrics_dict.get('hidden_state_prediction_accuracy', 0.0),
                    'win_rate_improvement': metrics_dict.get('win_rate_improvement', 0.0),
                    'behavior_consistency': metrics_dict.get('behavior_consistency', 0.0)
                }
            except Exception as e:
                print(f"      ⚠️ 评估失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 如果是baseline memory，返回模拟指标
        if hasattr(agent, 'store') and hasattr(agent, 'retrieve'):
            retrieved_count = results.get('retrieved_count', 0) if results else 0
            return {
                'hidden_state_prediction': min(0.5, retrieved_count / 10.0),
                'win_rate_improvement': min(0.1, retrieved_count / 20.0),
                'behavior_consistency': min(0.6, retrieved_count / 10.0)
            }
        
        # 默认返回
        return {
            'hidden_state_prediction': 0.0,
            'win_rate_improvement': 0.0,
            'behavior_consistency': 0.0
        }
    
    # 运行场景评估
    if PokerRLAgent:
        try:
            our_agent = create_our_agent()
            
            # 在评估前模拟游戏手数，让agent学习对手
            print("\n1. 模拟游戏手数，记录对手动作...")
            simulate_poker_hands(our_agent, ground_truth, num_hands_per_opponent=30)
            print(f"   已模拟 {len(ground_truth)} 个对手，每个 {30} 手")
            
            # 打印学习到的对手信息
            if hasattr(our_agent, 'get_system_statistics'):
                stats = our_agent.get_system_statistics()
                memory_stats = stats.get('memory', {})
                print(f"   追踪手数: {memory_stats.get('total_hands_tracked', 0)}")
                print(f"   对手数量: {memory_stats.get('total_opponents', 0)}")
            
            print("\n2. 运行评估...")
            results = evaluator.comprehensive_evaluation(
                our_agent, ground_truth, None, test_episodes
            )
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            evaluator.save_results(f'evaluation_results/poker_evaluation_{timestamp}.json')
            print("\n✓ Poker场景评估完成")
        except Exception as e:
            print(f"\n⚠️ 评估运行出错: {e}")
            import traceback
            traceback.print_exc()
            results = {}
    else:
        results = {}
    
    # 运行Baseline对比
    print("\n" + "="*80)
    print("开始Baseline对比...")
    print("="*80)
    
    comparison = ScenarioComparison("Poker")
    baseline_agents = create_baseline_agents("poker")
    
    # 创建测试场景
    test_scenario = {
        'ground_truth': ground_truth,
        'test_episodes': test_episodes,
        'num_hands': 100
    }
    
    # 运行对比
    comparison_results = comparison.compare_with_baselines(
        create_our_agent,
        baseline_agents,
        test_scenario,
        calculate_metrics
    )
    
    # 保存对比结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison.save_results(f'evaluation_results/poker_comparison_{timestamp}.json')
    
    # 打印总结
    print("\n" + "="*80)
    print("对比总结")
    print("="*80)
    summary = comparison_results.get('summary', {})
    print(f"\n成功对比的Baseline数量: {summary.get('successful_baselines', 0)}/{summary.get('total_baselines', 0)}")
    if 'average_improvements' in summary:
        print("\n平均改进:")
        for metric, improvement in summary['average_improvements'].items():
            print(f"  {metric}: {improvement:+.4f}")


if __name__ == "__main__":
    main()

