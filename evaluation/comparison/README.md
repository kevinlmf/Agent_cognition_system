# Evaluation Comparison Module

## 📁 文件结构

```
evaluation/
├── comparison/                    # Baseline对比模块
│   ├── __init__.py
│   ├── baseline_memory.py       # Baseline实现（LSTM、Transformer等）
│   ├── compare_memory_systems.py # 通用对比评估器
│   ├── scenario_comparison.py   # 场景特定对比框架
│   └── README.md                # 本文件
├── evaluate_poker.py            # Poker场景评估（包含baseline对比）
├── evaluate_industrial.py       # Industrial场景评估（包含baseline对比）
├── evaluate_health.py           # Health场景评估（包含baseline对比）
└── evaluate_memory_effectiveness.py # 通用评估
```

## 🎯 设计理念

**每个场景的评估脚本都包含baseline对比**，这样可以：
1. 展示我们的Memory系统在特定场景下的优势
2. 针对不同场景选择合适的baseline
3. 生成场景特定的对比报告

## 📊 对比的Baseline

### Poker场景
- LSTM
- Transformer
- Memory Networks

### Industrial场景
- LSTM
- Transformer
- Memory Networks

### Health场景
- LSTM
- Transformer
- Episodic Memory

## 🚀 使用方法

### 运行场景评估（自动包含baseline对比）

```bash
cd /Users/mengfanlong/Downloads/Projects/MLE/Memory_System

# Poker场景
python evaluation/evaluate_poker.py

# Industrial场景
python evaluation/evaluate_industrial.py

# Health场景
python evaluation/evaluate_health.py
```

每个脚本会：
1. 运行我们的Memory系统评估
2. 运行各个baseline评估
3. 对比结果并生成报告

## 📝 结果文件

每个场景会生成两个文件：
1. `{scenario}_evaluation_*.json` - 我们的系统评估结果
2. `{scenario}_comparison_*.json` - Baseline对比结果

## 🔧 自定义对比

### 添加新的Baseline

在 `scenario_comparison.py` 的 `create_baseline_agents` 函数中添加：

```python
def create_baseline_agents(scenario_type: str = "generic"):
    baselines = {
        'Your Baseline': lambda: YourBaselineClass()
    }
    return baselines
```

### 在场景评估中使用

```python
from evaluation.comparison.scenario_comparison import ScenarioComparison, create_baseline_agents

comparison = ScenarioComparison("Your Scenario")
baseline_agents = create_baseline_agents("your_scenario_type")

comparison_results = comparison.compare_with_baselines(
    create_our_agent,
    baseline_agents,
    test_scenario,
    calculate_metrics
)
```

## 📈 对比指标

每个场景有自己的指标：

### Poker
- Hidden State Prediction
- Win Rate Improvement
- Behavior Consistency

### Industrial
- System Stability
- Throughput Improvement
- Robustness

### Health
- Future Behavior Prediction
- Personalized Policy Improvement
- Latent State Estimation

## 💡 优势

1. **场景特定** - 每个场景选择最相关的baseline
2. **统一框架** - 使用相同的对比框架，便于扩展
3. **自动对比** - 运行评估时自动进行baseline对比
4. **详细报告** - 生成包含改进幅度的详细报告

---

**开始评估和对比吧！** 🚀
