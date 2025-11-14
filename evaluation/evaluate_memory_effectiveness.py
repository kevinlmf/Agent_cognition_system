"""
Memory Effectiveness Evaluation Framework
评估Memory系统在不同环境中的有效性

支持的环境：
- Trading (Stock Market)
- Bitcoin Trading
- FlexSim Simulation
- Poker AI (如果存在)
"""
import sys
import os
# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import json
from dataclasses import dataclass, asdict


@dataclass
class EnvironmentMetrics:
    """环境特定的评估指标"""
    environment_name: str
    total_episodes: int
    memory_usage: Dict[str, Any]
    decision_quality: Dict[str, float]
    performance_metrics: Dict[str, float]
    memory_impact: Dict[str, float]  # 有memory vs 无memory的对比


class MemoryEffectivenessEvaluator:
    """
    评估Memory系统在不同环境中的有效性
    """
    
    def __init__(self):
        self.results = {}
        self.environment_results = {}
    
    def evaluate_environment(self, 
                           environment_name: str,
                           run_simulation: Callable,
                           baseline_run: Optional[Callable] = None) -> EnvironmentMetrics:
        """
        评估特定环境中的memory有效性
        
        Args:
            environment_name: 环境名称
            run_simulation: 运行模拟的函数，返回agent和统计信息
            baseline_run: 可选，运行无memory baseline的函数
        """
        print(f"\n{'='*80}")
        print(f"评估环境: {environment_name}")
        print(f"{'='*80}")
        
        # 运行带memory的模拟
        print("\n1. 运行带Memory的模拟...")
        agent_with_memory, stats_with_memory = run_simulation()
        
        # 收集memory指标
        memory_stats = self._collect_memory_metrics(agent_with_memory)
        
        # 收集决策质量指标
        decision_quality = self._evaluate_decision_quality(agent_with_memory, stats_with_memory)
        
        # 收集性能指标
        performance_metrics = self._extract_performance_metrics(stats_with_memory)
        
        # 如果有baseline，进行对比
        memory_impact = {}
        if baseline_run:
            print("\n2. 运行Baseline（无Memory）模拟...")
            agent_baseline, stats_baseline = baseline_run()
            memory_impact = self._compare_with_baseline(
                stats_with_memory, stats_baseline
            )
        else:
            print("\n2. 跳过Baseline对比（未提供baseline函数）")
        
        # 创建指标对象
        metrics = EnvironmentMetrics(
            environment_name=environment_name,
            total_episodes=memory_stats.get('total_episodes', 0),
            memory_usage=memory_stats,
            decision_quality=decision_quality,
            performance_metrics=performance_metrics,
            memory_impact=memory_impact
        )
        
        self.environment_results[environment_name] = metrics
        
        # 打印结果
        self._print_environment_results(metrics)
        
        return metrics
    
    def _collect_memory_metrics(self, agent) -> Dict[str, Any]:
        """收集memory使用情况指标"""
        try:
            stats = agent.agent.get_statistics()
            mem_stats = stats.get('memory', {})
            mem_graph = mem_stats.get('memory_graph', {})
            
            # 计算memory利用率
            total_memories = mem_graph.get('total_memories', 0)
            episodic_memories = mem_graph.get('episodic_memories', 0)
            semantic_memories = mem_graph.get('semantic_memories', 0)
            
            # Memory检索统计
            retrieval_stats = {
                'total_memories': total_memories,
                'episodic_memories': episodic_memories,
                'semantic_memories': semantic_memories,
                'memory_diversity': self._calculate_memory_diversity(agent),
                'memory_consolidation_rate': semantic_memories / max(1, episodic_memories)
            }
            
            return retrieval_stats
        except Exception as e:
            print(f"Warning: Could not collect memory metrics: {e}")
            return {'total_episodes': 0, 'error': str(e)}
    
    def _calculate_memory_diversity(self, agent) -> float:
        """计算memory多样性（基于embedding相似度）"""
        try:
            memories = agent.agent.memory_engine.memory_graph.memories
            if len(memories) < 2:
                return 0.0
            
            embeddings = []
            for mem in memories.values():
                if mem.embedding is not None:
                    embeddings.append(mem.embedding)
            
            if len(embeddings) < 2:
                return 0.0
            
            # 计算平均相似度
            embeddings_array = np.array(embeddings)
            similarities = []
            for i in range(len(embeddings_array)):
                for j in range(i+1, len(embeddings_array)):
                    sim = np.dot(embeddings_array[i], embeddings_array[j]) / (
                        np.linalg.norm(embeddings_array[i]) * 
                        np.linalg.norm(embeddings_array[j])
                    )
                    similarities.append(sim)
            
            avg_similarity = np.mean(similarities) if similarities else 0.0
            diversity = 1.0 - avg_similarity  # 多样性 = 1 - 平均相似度
            
            return float(diversity)
        except Exception as e:
            return 0.0
    
    def _evaluate_decision_quality(self, agent, stats: Dict) -> Dict[str, float]:
        """评估决策质量"""
        quality_metrics = {}
        
        # 基于reward的决策质量
        if 'total_reward' in stats or 'total_profit' in stats:
            reward = stats.get('total_reward', stats.get('total_profit', 0))
            quality_metrics['reward_based_quality'] = max(0, min(1, reward / 1000.0))
        
        # 基于胜率的决策质量（如果适用）
        if 'win_rate' in stats:
            quality_metrics['win_rate'] = stats['win_rate']
        
        # 基于return的决策质量（如果适用）
        if 'total_return' in stats:
            quality_metrics['return_based_quality'] = max(0, min(1, (stats['total_return'] + 1) / 2))
        
        # Memory检索相关性（如果有）
        try:
            state = agent.agent.get_state()
            relevant_memories = len(state.relevant_memories)
            important_memories = len(state.important_memories)
            quality_metrics['memory_relevance'] = min(1.0, (relevant_memories + important_memories) / 20.0)
        except:
            quality_metrics['memory_relevance'] = 0.0
        
        return quality_metrics
    
    def _extract_performance_metrics(self, stats: Dict) -> Dict[str, float]:
        """提取性能指标"""
        performance = {}
        
        # 通用性能指标
        if 'total_profit' in stats:
            performance['total_profit'] = stats['total_profit']
        if 'total_return' in stats:
            performance['total_return'] = stats['total_return']
        if 'win_rate' in stats:
            performance['win_rate'] = stats['win_rate']
        if 'avg_throughput' in stats:
            performance['avg_throughput'] = stats['avg_throughput']
        if 'total_reward' in stats:
            performance['total_reward'] = stats['total_reward']
        
        return performance
    
    def _compare_with_baseline(self, 
                              stats_with_memory: Dict,
                              stats_baseline: Dict) -> Dict[str, float]:
        """对比有memory和无memory的性能"""
        impact = {}
        
        # 对比各种指标
        for key in ['total_profit', 'total_return', 'win_rate', 'total_reward', 'avg_throughput']:
            if key in stats_with_memory and key in stats_baseline:
                with_mem = stats_with_memory[key]
                baseline = stats_baseline[key]
                
                if baseline != 0:
                    improvement = (with_mem - baseline) / abs(baseline)
                    impact[f'{key}_improvement'] = float(improvement)
                else:
                    impact[f'{key}_improvement'] = float('inf') if with_mem > 0 else 0.0
        
        return impact
    
    def _print_environment_results(self, metrics: EnvironmentMetrics):
        """打印环境评估结果"""
        print(f"\n{'='*80}")
        print(f"环境评估结果: {metrics.environment_name}")
        print(f"{'='*80}")
        
        print(f"\n📊 Memory使用情况:")
        mem_usage = metrics.memory_usage
        print(f"   总记忆数: {mem_usage.get('total_memories', 0)}")
        print(f"   情景记忆: {mem_usage.get('episodic_memories', 0)}")
        print(f"   语义记忆: {mem_usage.get('semantic_memories', 0)}")
        print(f"   Memory多样性: {mem_usage.get('memory_diversity', 0):.3f}")
        print(f"   记忆整合率: {mem_usage.get('memory_consolidation_rate', 0):.3f}")
        
        print(f"\n🎯 决策质量:")
        for key, value in metrics.decision_quality.items():
            print(f"   {key}: {value:.3f}")
        
        print(f"\n📈 性能指标:")
        for key, value in metrics.performance_metrics.items():
            print(f"   {key}: {value:.3f}")
        
        if metrics.memory_impact:
            print(f"\n💡 Memory影响（vs Baseline）:")
            for key, value in metrics.memory_impact.items():
                if isinstance(value, float) and not np.isinf(value):
                    print(f"   {key}: {value:+.2%}")
                else:
                    print(f"   {key}: {value}")
    
    def comprehensive_evaluation(self, environments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        综合评估多个环境
        
        Args:
            environments: 环境配置列表
                [
                    {
                        "name": "Trading",
                        "run": run_trading_simulation,
                        "baseline": run_trading_baseline  # 可选
                    },
                    ...
                ]
        """
        print("\n" + "="*80)
        print("MEMORY系统有效性综合评估")
        print("="*80)
        
        all_metrics = []
        
        for env_config in environments:
            name = env_config['name']
            run_func = env_config['run']
            baseline_func = env_config.get('baseline', None)
            
            metrics = self.evaluate_environment(name, run_func, baseline_func)
            all_metrics.append(metrics)
        
        # 计算综合得分
        overall_score = self._calculate_overall_score(all_metrics)
        
        # 生成报告
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'environments_evaluated': [m.environment_name for m in all_metrics],
            'environment_results': {m.environment_name: asdict(m) for m in all_metrics},
            'overall_score': overall_score,
            'summary': self._generate_summary(all_metrics, overall_score)
        }
        
        self.results = report
        
        # 打印综合报告
        self._print_comprehensive_report(report)
        
        return report
    
    def _calculate_overall_score(self, metrics_list: List[EnvironmentMetrics]) -> Dict[str, float]:
        """计算综合得分"""
        scores = {
            'memory_utilization': [],
            'decision_quality': [],
            'performance': [],
            'memory_impact': []
        }
        
        for metrics in metrics_list:
            # Memory利用率得分
            mem_usage = metrics.memory_usage
            utilization_score = (
                min(1.0, mem_usage.get('total_memories', 0) / 100.0) * 0.3 +
                mem_usage.get('memory_diversity', 0) * 0.4 +
                min(1.0, mem_usage.get('memory_consolidation_rate', 0)) * 0.3
            )
            scores['memory_utilization'].append(utilization_score)
            
            # 决策质量得分
            decision_scores = list(metrics.decision_quality.values())
            if decision_scores:
                scores['decision_quality'].append(np.mean(decision_scores))
            
            # 性能得分（归一化）
            perf_scores = []
            for key, value in metrics.performance_metrics.items():
                if 'return' in key or 'profit' in key:
                    perf_scores.append(max(0, min(1, (value + 1) / 2)))
                elif 'rate' in key or 'win' in key:
                    perf_scores.append(value)
                else:
                    perf_scores.append(min(1.0, value / 100.0))
            if perf_scores:
                scores['performance'].append(np.mean(perf_scores))
            
            # Memory影响得分
            if metrics.memory_impact:
                impact_scores = [v for v in metrics.memory_impact.values() 
                               if isinstance(v, float) and not np.isinf(v)]
                if impact_scores:
                    # 转换为0-1得分
                    normalized_impacts = [max(0, min(1, (imp + 1) / 2)) for imp in impact_scores]
                    scores['memory_impact'].append(np.mean(normalized_impacts))
        
        # 计算平均得分
        overall = {
            'memory_utilization': np.mean(scores['memory_utilization']) if scores['memory_utilization'] else 0.0,
            'decision_quality': np.mean(scores['decision_quality']) if scores['decision_quality'] else 0.0,
            'performance': np.mean(scores['performance']) if scores['performance'] else 0.0,
            'memory_impact': np.mean(scores['memory_impact']) if scores['memory_impact'] else 0.0
        }
        
        # 综合得分
        overall['total'] = np.mean(list(overall.values()))
        
        return overall
    
    def _generate_summary(self, metrics_list: List[EnvironmentMetrics], 
                         overall_score: Dict[str, float]) -> Dict[str, Any]:
        """生成摘要"""
        return {
            'total_environments': len(metrics_list),
            'overall_score': overall_score['total'],
            'best_environment': max(metrics_list, key=lambda m: 
                m.performance_metrics.get('total_profit', 
                m.performance_metrics.get('total_reward', 0))).environment_name,
            'memory_effectiveness': 'High' if overall_score['total'] > 0.7 else 
                                   'Medium' if overall_score['total'] > 0.5 else 'Low'
        }
    
    def _print_comprehensive_report(self, report: Dict[str, Any]):
        """打印综合报告"""
        print("\n" + "="*80)
        print("综合评估报告")
        print("="*80)
        
        summary = report['summary']
        overall_score = report['overall_score']
        
        print(f"\n📊 评估摘要:")
        print(f"   评估环境数: {summary['total_environments']}")
        print(f"   综合得分: {overall_score['total']:.2%}")
        print(f"   Memory有效性: {summary['memory_effectiveness']}")
        print(f"   最佳环境: {summary['best_environment']}")
        
        print(f"\n📈 各维度得分:")
        print(f"   Memory利用率: {overall_score['memory_utilization']:.2%}")
        print(f"   决策质量: {overall_score['decision_quality']:.2%}")
        print(f"   性能表现: {overall_score['performance']:.2%}")
        if overall_score['memory_impact'] > 0:
            print(f"   Memory影响: {overall_score['memory_impact']:.2%}")
        
        print("\n" + "="*80)
    
    def save_results(self, filepath: str):
        """保存评估结果"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n✓ 评估结果已保存到: {filepath}")


def run_trading_simulation():
    """运行Trading环境模拟"""
    import sys
    import os
    import importlib.util
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    # 使用绝对导入
    spec = importlib.util.spec_from_file_location(
        "trading_agent_example",
        os.path.join(project_root, "examples", "trading_agent_example.py")
    )
    trading_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trading_module)
    TradingAgent = trading_module.TradingAgent
    
    agent = TradingAgent()
    
    # 简化的模拟
    market_events = [
        {"price": 100.0, "volume": 1000, "news": "Market opens steady."},
        {"price": 102.5, "volume": 1500, "news": "Fed announces rate cut."},
        {"price": 105.0, "volume": 2000, "news": "Stock prices surge."},
    ]
    
    for event in market_events:
        agent.perceive_market({"price": event['price'], "volume": event['volume']}, event['news'])
        action = agent.decide_action({"price": event['price']})
        agent.execute_trade(action, event['price'])
    
    stats = agent.agent.get_statistics()
    portfolio_value = agent.get_portfolio_value(market_events[-1]['price'])
    pnl = portfolio_value - agent.initial_balance
    
    return agent, {
        'total_profit': pnl,
        'total_return': pnl / agent.initial_balance,
        'total_reward': sum([t.get('reward', 0) for t in agent.trade_history])
    }


def run_bitcoin_simulation():
    """运行Bitcoin环境模拟"""
    import sys
    import os
    import importlib.util
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    spec = importlib.util.spec_from_file_location(
        "bitcoin_trading_example",
        os.path.join(project_root, "examples", "bitcoin_trading_example.py")
    )
    bitcoin_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bitcoin_module)
    BitcoinTradingAgent = bitcoin_module.BitcoinTradingAgent
    
    agent = BitcoinTradingAgent()
    
    market_events = [
        {"price": 45000.0, "volume": 1250.5, "rsi": 45, "news": "Bitcoin consolidates."},
        {"price": 46500.0, "volume": 1800.2, "rsi": 55, "news": "Bitcoin breaks resistance."},
        {"price": 48000.0, "volume": 2200.8, "rsi": 65, "news": "Bitcoin surges."},
    ]
    
    for event in market_events:
        indicators = {"rsi": event['rsi']}
        agent.perceive_market(event['price'], event['volume'], event['news'], indicators)
        action = agent.decide_action(event['price'], indicators)
        agent.execute_trade(action, event['price'])
    
    stats_dict = agent.get_statistics()
    
    return agent, stats_dict


def run_flexsim_simulation():
    """运行FlexSim环境模拟"""
    import sys
    import os
    import importlib.util
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    spec = importlib.util.spec_from_file_location(
        "flexsim_example",
        os.path.join(project_root, "examples", "flexsim_example.py")
    )
    flexsim_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(flexsim_module)
    FlexSimAgent = flexsim_module.FlexSimAgent
    
    agent = FlexSimAgent()
    
    current_state = {
        "production_rate": 100.0,
        "queue_length": 30,
        "resource_utilization": 0.6,
        "throughput": 60.0,
        "bottlenecks": []
    }
    
    for step in range(3):
        current_state['queue_length'] += 10
        agent.perceive_system_state(current_state, [f"Step {step+1}"])
        action = agent.decide_optimization(current_state)
        current_state, _ = agent.apply_optimization(action, current_state)
    
    stats_dict = agent.get_statistics()
    
    return agent, stats_dict


def main():
    """主函数：运行综合评估"""
    print("="*80)
    print("Memory系统有效性评估")
    print("="*80)
    
    evaluator = MemoryEffectivenessEvaluator()
    
    # 定义要评估的环境
    environments = [
        {
            "name": "Stock Trading",
            "run": run_trading_simulation
        },
        {
            "name": "Bitcoin Trading",
            "run": run_bitcoin_simulation
        },
        {
            "name": "FlexSim Simulation",
            "run": run_flexsim_simulation
        }
    ]
    
    # 运行综合评估
    report = evaluator.comprehensive_evaluation(environments)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_results/memory_effectiveness_{timestamp}.json"
    evaluator.save_results(output_file)
    
    return report


if __name__ == "__main__":
    main()

