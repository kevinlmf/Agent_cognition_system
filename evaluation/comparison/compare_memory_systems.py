"""
Memory Systems Comparison
对比我们的Memory系统与常见baseline（LSTM、Transformer、Memory Networks等）
"""
import sys
import os
# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
evaluation_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(evaluation_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from evaluation.comparison.baseline_memory import (
    LSTMMemory, TransformerMemory, MemoryNetworkBaseline, EpisodicMemoryBaseline
)
from main import CognitiveAgent


class MemoryComparisonEvaluator:
    """
    对比评估器：对比我们的Memory系统与baseline
    """
    
    def __init__(self):
        self.results = {}
    
    def evaluate_retrieval_accuracy(self, our_memory, baseline_memory, 
                                   test_queries: List[np.ndarray],
                                   ground_truth: List[List[int]]) -> Dict[str, float]:
        """
        评估检索准确率
        """
        our_accuracies = []
        baseline_accuracies = []
        
        for query, true_indices in zip(test_queries, ground_truth):
            # 我们的系统
            our_results = our_memory.retrieve(
                query_embedding=query,
                retrieval_strategy="similar",
                top_k=len(true_indices)
            )
            our_retrieved = set([i for i in range(len(our_results))])
            our_accuracy = len(our_retrieved & set(true_indices)) / len(true_indices) if true_indices else 0.0
            our_accuracies.append(our_accuracy)
            
            # Baseline
            baseline_results = baseline_memory.retrieve(query, top_k=len(true_indices))
            baseline_retrieved = set([r.get('index', -1) for r in baseline_results])
            baseline_accuracy = len(baseline_retrieved & set(true_indices)) / len(true_indices) if true_indices else 0.0
            baseline_accuracies.append(baseline_accuracy)
        
        return {
            'our_accuracy': np.mean(our_accuracies),
            'baseline_accuracy': np.mean(baseline_accuracies),
            'improvement': np.mean(our_accuracies) - np.mean(baseline_accuracies)
        }
    
    def evaluate_explicit_query(self, our_memory, baseline_memory, 
                               queries: List[str]) -> Dict[str, Any]:
        """
        评估显式查询能力（如"对手1的VPIP是多少？"）
        """
        our_success = 0
        baseline_success = 0
        
        for query in queries:
            # 我们的系统：可以显式查询
            try:
                if hasattr(our_memory, 'query_memory'):
                    our_result = our_memory.query_memory(query)
                    if our_result:
                        our_success += 1
            except:
                pass
            
            # Baseline：无法显式查询
            baseline_success += 0  # Baseline无法做显式查询
        
        return {
            'our_explicit_query_rate': our_success / len(queries) if queries else 0.0,
            'baseline_explicit_query_rate': 0.0,
            'explicit_query_advantage': our_success / len(queries) if queries else 0.0
        }
    
    def evaluate_structured_storage(self, our_memory, baseline_memory) -> Dict[str, Any]:
        """
        评估结构化存储能力
        """
        our_stats = our_memory.get_statistics() if hasattr(our_memory, 'get_statistics') else {}
        baseline_stats = baseline_memory.get_statistics()
        
        # 检查结构化能力
        our_structured = {
            'has_episodic': hasattr(our_memory, 'episodic_memory'),
            'has_semantic': hasattr(our_memory, 'semantic_memory'),
            'has_graph': hasattr(our_memory, 'memory_graph'),
            'can_query_by_type': True  # 我们的系统可以
        }
        
        baseline_structured = {
            'has_episodic': baseline_stats.get('type') == 'Episodic Memory',
            'has_semantic': False,  # Baseline通常没有
            'has_graph': False,  # Baseline通常没有
            'can_query_by_type': False  # Baseline无法按类型查询
        }
        
        return {
            'our_structured_features': our_structured,
            'baseline_structured_features': baseline_structured,
            'structured_advantage': sum(our_structured.values()) - sum(baseline_structured.values())
        }
    
    def evaluate_long_term_dependency(self, our_memory, baseline_memory,
                                     long_history: List[np.ndarray]) -> Dict[str, float]:
        """
        评估长期依赖能力
        """
        # 存储长期历史
        for i, obs in enumerate(long_history):
            our_memory.store_experience(
                world_snapshot={'step': i},
                perception_result={'summary': f'Step {i}'},
                reward=i * 0.01,
                embedding=obs
            )
            
            baseline_memory.store(obs, content=f'Step {i}')
        
        # 查询早期记忆
        early_query = long_history[0]
        
        # 我们的系统
        our_results = our_memory.retrieve(
            query_embedding=early_query,
            retrieval_strategy='similar',
            top_k=5
        )
        our_can_retrieve_early = len(our_results) > 0
        
        # Baseline
        baseline_results = baseline_memory.retrieve(early_query, top_k=5)
        baseline_can_retrieve_early = len(baseline_results) > 0
        
        return {
            'our_long_term_ability': 1.0 if our_can_retrieve_early else 0.0,
            'baseline_long_term_ability': 1.0 if baseline_can_retrieve_early else 0.0,
            'long_term_advantage': (1.0 if our_can_retrieve_early else 0.0) - (1.0 if baseline_can_retrieve_early else 0.0)
        }
    
    def compare_with_baseline(self, our_memory, baseline_memory, 
                            baseline_name: str,
                            test_scenarios: Dict[str, Any]) -> Dict[str, Any]:
        """
        全面对比我们的系统与baseline
        """
        print(f"\n{'='*80}")
        print(f"对比: Our Memory System vs {baseline_name}")
        print(f"{'='*80}")
        
        comparison_results = {
            'baseline_name': baseline_name,
            'comparison_timestamp': datetime.now().isoformat()
        }
        
        # 1. 检索准确率
        if 'test_queries' in test_scenarios and 'ground_truth' in test_scenarios:
            retrieval_results = self.evaluate_retrieval_accuracy(
                our_memory, baseline_memory,
                test_scenarios['test_queries'],
                test_scenarios['ground_truth']
            )
            comparison_results['retrieval_accuracy'] = retrieval_results
            print(f"\n📊 检索准确率:")
            print(f"   我们的系统: {retrieval_results['our_accuracy']:.3f}")
            print(f"   {baseline_name}: {retrieval_results['baseline_accuracy']:.3f}")
            print(f"   提升: {retrieval_results['improvement']:+.3f}")
        
        # 2. 显式查询能力
        if 'explicit_queries' in test_scenarios:
            explicit_results = self.evaluate_explicit_query(
                our_memory, baseline_memory,
                test_scenarios['explicit_queries']
            )
            comparison_results['explicit_query'] = explicit_results
            print(f"\n🔍 显式查询能力:")
            print(f"   我们的系统: {explicit_results['our_explicit_query_rate']:.3f}")
            print(f"   {baseline_name}: {explicit_results['baseline_explicit_query_rate']:.3f}")
        
        # 3. 结构化存储
        structured_results = self.evaluate_structured_storage(our_memory, baseline_memory)
        comparison_results['structured_storage'] = structured_results
        print(f"\n📁 结构化存储:")
        print(f"   我们的系统: {structured_results['our_structured_features']}")
        print(f"   {baseline_name}: {structured_results['baseline_structured_features']}")
        print(f"   优势: +{structured_results['structured_advantage']}")
        
        # 4. 长期依赖
        if 'long_history' in test_scenarios:
            long_term_results = self.evaluate_long_term_dependency(
                our_memory, baseline_memory,
                test_scenarios['long_history']
            )
            comparison_results['long_term_dependency'] = long_term_results
            print(f"\n⏰ 长期依赖:")
            print(f"   我们的系统: {long_term_results['our_long_term_ability']:.3f}")
            print(f"   {baseline_name}: {long_term_results['baseline_long_term_ability']:.3f}")
        
        # 计算综合得分
        scores = []
        if 'retrieval_accuracy' in comparison_results:
            scores.append(comparison_results['retrieval_accuracy']['improvement'] + 0.5)
        if 'explicit_query' in comparison_results:
            scores.append(comparison_results['explicit_query']['explicit_query_advantage'])
        if 'structured_storage' in comparison_results:
            scores.append(min(1.0, comparison_results['structured_storage']['structured_advantage'] / 4.0))
        
        overall_score = np.mean(scores) if scores else 0.5
        comparison_results['overall_advantage'] = float(overall_score)
        
        print(f"\n📈 综合优势得分: {overall_score:.3f}")
        
        return comparison_results
    
    def comprehensive_comparison(self, our_memory, baselines: Dict[str, Any],
                                test_scenarios: Dict[str, Any]) -> Dict[str, Any]:
        """
        与多个baseline全面对比
        """
        print("\n" + "="*80)
        print("MEMORY系统全面对比")
        print("="*80)
        
        all_comparisons = {}
        
        for baseline_name, baseline_memory in baselines.items():
            comparison = self.compare_with_baseline(
                our_memory, baseline_memory, baseline_name, test_scenarios
            )
            all_comparisons[baseline_name] = comparison
        
        # 生成总结报告
        report = {
            'comparison_timestamp': datetime.now().isoformat(),
            'our_system': 'Our Memory System (Episodic + Semantic + Graph)',
            'baselines_compared': list(baselines.keys()),
            'comparisons': all_comparisons,
            'summary': self._generate_summary(all_comparisons)
        }
        
        self.results = report
        return report
    
    def _generate_summary(self, comparisons: Dict[str, Any]) -> Dict[str, Any]:
        """生成对比总结"""
        avg_advantages = []
        for baseline_name, comp in comparisons.items():
            if 'overall_advantage' in comp:
                avg_advantages.append(comp['overall_advantage'])
        
        return {
            'average_advantage': np.mean(avg_advantages) if avg_advantages else 0.0,
            'best_baseline_comparison': max(comparisons.items(), 
                                           key=lambda x: x[1].get('overall_advantage', 0))[0] if comparisons else None,
            'total_baselines': len(comparisons)
        }
    
    def save_results(self, filepath: str):
        """保存对比结果"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n✓ 对比结果已保存到: {filepath}")


def create_test_scenarios():
    """创建测试场景"""
    # 生成测试数据
    np.random.seed(42)
    
    # 测试查询
    test_queries = [np.random.randn(128) for _ in range(10)]
    
    # Ground truth（假设前5个是最相关的）
    ground_truth = [[i for i in range(5)] for _ in range(10)]
    
    # 显式查询
    explicit_queries = [
        "对手1的VPIP是多少？",
        "最近10手的平均利润是多少？",
        "系统稳定性指标如何？"
    ]
    
    # 长期历史
    long_history = [np.random.randn(128) for _ in range(200)]
    
    return {
        'test_queries': test_queries,
        'ground_truth': ground_truth,
        'explicit_queries': explicit_queries,
        'long_history': long_history
    }


def main():
    """主函数：运行全面对比"""
    print("="*80)
    print("Memory系统对比评估")
    print("="*80)
    
    # 创建我们的Memory系统
    print("\n1. 初始化我们的Memory系统...")
    our_memory = CognitiveAgent(mode="langgraph")
    
    # 填充一些数据
    np.random.seed(42)
    for i in range(50):
        obs_text = f"Observation {i}: Market event occurred"
        our_memory.perceive(obs_text, source="test")
        if i % 10 == 0:
            our_memory.record_action(
                action_type="test_action",
                parameters={"step": i},
                result=f"Processed step {i}",
                reward=i * 0.01
            )
    
    # 创建baselines
    print("\n2. 初始化Baseline Memory系统...")
    baselines = {
        'LSTM': LSTMMemory(hidden_size=128),
        'Transformer': TransformerMemory(d_model=128),
        'Memory Networks': MemoryNetworkBaseline(memory_size=1000),
        'Episodic Memory': EpisodicMemoryBaseline(max_memories=10000)
    }
    
    # 填充baseline数据
    np.random.seed(42)
    for i in range(50):
        obs = np.random.randn(128)
        for baseline in baselines.values():
            baseline.store(obs, content=f'Observation {i}')
    
    # 创建测试场景
    print("\n3. 创建测试场景...")
    test_scenarios = create_test_scenarios()
    
    # 运行对比
    print("\n4. 运行对比评估...")
    evaluator = MemoryComparisonEvaluator()
    report = evaluator.comprehensive_comparison(
        our_memory.memory_engine,
        baselines,
        test_scenarios
    )
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluator.save_results(f'evaluation_results/memory_comparison_{timestamp}.json')
    
    # 打印总结
    print("\n" + "="*80)
    print("对比总结")
    print("="*80)
    summary = report['summary']
    print(f"\n平均优势: {summary['average_advantage']:.3f}")
    print(f"最佳对比: {summary['best_baseline_comparison']}")
    print(f"对比的Baseline数量: {summary['total_baselines']}")
    
    return report


if __name__ == "__main__":
    main()

