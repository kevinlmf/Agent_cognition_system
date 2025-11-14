"""
Scenario-specific Comparison
为每个场景提供baseline对比功能
"""
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
evaluation_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(evaluation_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import json

from evaluation.comparison.baseline_memory import (
    LSTMMemory, TransformerMemory, MemoryNetworkBaseline, EpisodicMemoryBaseline
)


class ScenarioComparison:
    """
    场景特定的对比评估器
    为每个场景（Poker、Industrial、Health等）提供baseline对比
    """
    
    def __init__(self, scenario_name: str):
        self.scenario_name = scenario_name
        self.results = {}
    
    def compare_with_baselines(self, 
                              our_agent_func: Callable,
                              baseline_agents: Dict[str, Callable],
                              test_scenario: Dict[str, Any],
                              metrics_func: Callable) -> Dict[str, Any]:
        """
        对比我们的系统与baselines在特定场景下的表现
        
        Args:
            our_agent_func: 创建我们的agent的函数
            baseline_agents: {baseline_name: create_agent_func} 字典
            test_scenario: 测试场景配置
            metrics_func: 计算指标的函数 (agent, results) -> metrics_dict
        
        Returns:
            对比结果字典
        """
        print(f"\n{'='*80}")
        print(f"{self.scenario_name} 场景 - Baseline对比")
        print(f"{'='*80}")
        
        comparison_results = {
            'scenario': self.scenario_name,
            'comparison_timestamp': datetime.now().isoformat(),
            'baselines': {}
        }
        
        # 运行我们的系统
        print(f"\n1. 运行我们的Memory系统...")
        our_agent = our_agent_func()
        our_results = self._run_scenario(our_agent, test_scenario)
        
        # 如果有预计算的results，使用它们
        if 'our_results' in test_scenario:
            our_results = test_scenario['our_results']
        
        our_metrics = metrics_func(our_agent, our_results)
        comparison_results['our_system'] = {
            'results': our_results,
            'metrics': our_metrics
        }
        
        print(f"\n   我们的系统指标:")
        for key, value in our_metrics.items():
            print(f"     {key}: {value:.4f}" if isinstance(value, (int, float)) else f"     {key}: {value}")
        
        # 运行每个baseline
        for baseline_name, baseline_func in baseline_agents.items():
            print(f"\n2. 运行{baseline_name} Baseline...")
            try:
                baseline_agent = baseline_func()
                baseline_results = self._run_scenario(baseline_agent, test_scenario)
                baseline_metrics = metrics_func(baseline_agent, baseline_results)
                
                comparison_results['baselines'][baseline_name] = {
                    'results': baseline_results,
                    'metrics': baseline_metrics
                }
                
                # 计算改进
                improvements = {}
                for key in our_metrics.keys():
                    if key in baseline_metrics and isinstance(our_metrics[key], (int, float)) and isinstance(baseline_metrics[key], (int, float)):
                        improvements[key] = our_metrics[key] - baseline_metrics[key]
                
                comparison_results['baselines'][baseline_name]['improvements'] = improvements
                
                print(f"   {baseline_name}指标:")
                for key, value in baseline_metrics.items():
                    print(f"     {key}: {value:.4f}" if isinstance(value, (int, float)) else f"     {key}: {value}")
                
                print(f"   改进:")
                for key, improvement in improvements.items():
                    print(f"     {key}: {improvement:+.4f}")
                    
            except Exception as e:
                print(f"   ⚠️ {baseline_name}运行失败: {e}")
                comparison_results['baselines'][baseline_name] = {
                    'error': str(e)
                }
        
        # 计算综合对比
        comparison_results['summary'] = self._calculate_summary(
            our_metrics, 
            comparison_results['baselines']
        )
        
        self.results = comparison_results
        return comparison_results
    
    def _run_scenario(self, agent, test_scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行测试场景，返回结果
        根据场景类型和agent类型智能运行
        """
        # 如果agent有run方法
        if hasattr(agent, 'run'):
            try:
                return agent.run(test_scenario)
            except Exception as e:
                print(f"      ⚠️ agent.run()失败: {e}")
        
        # 如果agent有evaluate方法
        if hasattr(agent, 'evaluate'):
            try:
                return agent.evaluate(test_scenario)
            except Exception as e:
                print(f"      ⚠️ agent.evaluate()失败: {e}")
        
        # 如果是我们的CognitiveAgent，使用evaluator
        if hasattr(agent, 'memory_engine'):
            # 这是一个CognitiveAgent，需要运行评估
            # 返回一个占位结果，实际评估会在metrics_func中完成
            return {'agent_type': 'cognitive_agent', 'has_memory': True}
        
        # 如果是baseline memory，模拟运行
        if hasattr(agent, 'store') and hasattr(agent, 'retrieve'):
            # 这是一个baseline memory，模拟存储和检索
            np.random.seed(42)
            results = {'agent_type': 'baseline_memory'}
            
            # 模拟存储一些数据
            if 'test_episodes' in test_scenario:
                for i, episode in enumerate(test_scenario['test_episodes'][:10]):
                    obs = np.random.randn(128)
                    agent.store(obs, content=f"Episode {i}")
            
            # 模拟检索
            if 'ground_truth' in test_scenario:
                query = np.random.randn(128)
                retrieved = agent.retrieve(query, top_k=5)
                results['retrieved_count'] = len(retrieved)
            
            return results
        
        # 默认返回空结果
        return {'agent_type': 'unknown'}
    
    def _calculate_summary(self, our_metrics: Dict, baselines: Dict) -> Dict[str, Any]:
        """计算对比总结"""
        summary = {
            'total_baselines': len(baselines),
            'successful_baselines': len([b for b in baselines.values() if 'metrics' in b]),
            'average_improvements': {}
        }
        
        # 计算平均改进
        all_improvements = {}
        for baseline_name, baseline_data in baselines.items():
            if 'improvements' in baseline_data:
                for metric, improvement in baseline_data['improvements'].items():
                    if metric not in all_improvements:
                        all_improvements[metric] = []
                    all_improvements[metric].append(improvement)
        
        for metric, improvements in all_improvements.items():
            summary['average_improvements'][metric] = np.mean(improvements)
        
        return summary
    
    def save_results(self, filepath: str):
        """保存对比结果"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n✓ 对比结果已保存到: {filepath}")


def create_baseline_agents(scenario_type: str = "generic") -> Dict[str, Callable]:
    """
    创建baseline agents的工厂函数
    
    Args:
        scenario_type: 场景类型 ("poker", "industrial", "health", "trading")
    
    Returns:
        {baseline_name: create_agent_func} 字典
    """
    baselines = {}
    
    if scenario_type == "poker":
        # Poker场景的baselines
        def create_lstm_poker():
            # 使用LSTM memory的Poker agent
            agent = LSTMMemory(hidden_size=128)
            return agent
        
        def create_transformer_poker():
            # 使用Transformer memory的Poker agent
            agent = TransformerMemory(d_model=128)
            return agent
        
        baselines = {
            'LSTM': create_lstm_poker,
            'Transformer': create_transformer_poker,
            'Memory Networks': lambda: MemoryNetworkBaseline(memory_size=1000)
        }
    
    elif scenario_type == "industrial":
        # Industrial场景的baselines
        baselines = {
            'LSTM': lambda: LSTMMemory(hidden_size=128),
            'Transformer': lambda: TransformerMemory(d_model=128),
            'Memory Networks': lambda: MemoryNetworkBaseline(memory_size=1000)
        }
    
    elif scenario_type == "health":
        # Health场景的baselines
        baselines = {
            'LSTM': lambda: LSTMMemory(hidden_size=128),
            'Transformer': lambda: TransformerMemory(d_model=128),
            'Episodic Memory': lambda: EpisodicMemoryBaseline(max_memories=10000)
        }
    
    else:  # generic/trading
        # 通用场景的baselines
        baselines = {
            'LSTM': lambda: LSTMMemory(hidden_size=128),
            'Transformer': lambda: TransformerMemory(d_model=128),
            'Memory Networks': lambda: MemoryNetworkBaseline(memory_size=1000),
            'Episodic Memory': lambda: EpisodicMemoryBaseline(max_memories=10000)
        }
    
    return baselines

