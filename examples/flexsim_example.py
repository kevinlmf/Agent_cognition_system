"""
FlexSim Simulation Environment Example
Demonstrates memory system effectiveness in simulation/optimization scenarios
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import CognitiveAgent
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple


class FlexSimAgent:
    """
    FlexSim simulation agent using perception and memory
    Simulates a manufacturing/logistics optimization scenario
    """
    
    def __init__(self):
        self.agent = CognitiveAgent(mode="langgraph")
        self.simulation_state = {
            "production_rate": 100.0,
            "queue_length": 0,
            "resource_utilization": 0.5,
            "throughput": 0.0,
            "bottlenecks": []
        }
        self.optimization_history = []
        self.performance_history = []
        
    def perceive_system_state(self, state: Dict[str, Any], events: List[str] = None):
        """
        Perceive FlexSim system state
        """
        obs_text = f"Production rate: {state.get('production_rate', 0):.1f} units/hr, "
        obs_text += f"Queue length: {state.get('queue_length', 0)}, "
        obs_text += f"Resource utilization: {state.get('resource_utilization', 0):.1%}"
        
        if state.get('bottlenecks'):
            obs_text += f", Bottlenecks: {', '.join(state['bottlenecks'])}"
        
        if events:
            obs_text += f" | Events: {'; '.join(events)}"
        
        result = self.agent.perceive(obs_text, source="flexsim")
        return result
    
    def decide_optimization(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide optimization actions based on memory
        """
        state = self.agent.get_state()
        
        # Retrieve relevant memories
        relevant_memories = state.relevant_memories
        important_memories = state.important_memories
        
        # Analyze patterns from memory
        optimization_suggestions = {
            "increase_capacity": 0.0,
            "reduce_queue": 0.0,
            "rebalance_resources": 0.0,
            "maintain": 0.0
        }
        
        for mem in relevant_memories + important_memories:
            summary = mem.get('summary', '').lower()
            content = mem.get('content', '').lower()
            importance = mem.get('importance', 0.5)
            
            # Pattern matching
            if 'bottleneck' in summary or 'bottleneck' in content:
                if 'increase' in summary or 'capacity' in summary:
                    optimization_suggestions["increase_capacity"] += importance
                elif 'rebalance' in summary or 'redistribute' in summary:
                    optimization_suggestions["rebalance_resources"] += importance
            
            if 'queue' in summary or 'queue' in content:
                if 'reduce' in summary or 'decrease' in summary:
                    optimization_suggestions["reduce_queue"] += importance
            
            if 'stable' in summary or 'optimal' in summary:
                optimization_suggestions["maintain"] += importance
        
        # Current state analysis
        queue_length = current_state.get('queue_length', 0)
        utilization = current_state.get('resource_utilization', 0)
        bottlenecks = current_state.get('bottlenecks', [])
        
        # Decision logic
        action = {"type": "maintain", "parameters": {}}
        
        if queue_length > 50:
            action = {
                "type": "reduce_queue",
                "parameters": {"target_queue": queue_length * 0.7}
            }
        elif utilization > 0.9 and bottlenecks:
            action = {
                "type": "increase_capacity",
                "parameters": {"resource": bottlenecks[0], "increase": 0.2}
            }
        elif utilization < 0.5:
            action = {
                "type": "rebalance_resources",
                "parameters": {"target_utilization": 0.7}
            }
        else:
            best_action = max(optimization_suggestions.items(), key=lambda x: x[1])
            if best_action[1] > 0.3:
                action = {
                    "type": best_action[0],
                    "parameters": {}
                }
        
        return action
    
    def apply_optimization(self, action: Dict[str, Any], current_state: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """
        Apply optimization action and return new state and reward
        """
        new_state = current_state.copy()
        reward = 0.0
        
        if action["type"] == "increase_capacity":
            resource = action["parameters"].get("resource", "general")
            increase = action["parameters"].get("increase", 0.1)
            new_state["production_rate"] *= (1 + increase)
            reward = 0.2
        
        elif action["type"] == "reduce_queue":
            target = action["parameters"].get("target_queue", new_state["queue_length"] * 0.7)
            reduction = new_state["queue_length"] - target
            new_state["queue_length"] = max(0, target)
            reward = reduction / 100.0  # Reward based on queue reduction
        
        elif action["type"] == "rebalance_resources":
            target_util = action["parameters"].get("target_utilization", 0.7)
            current_util = new_state["resource_utilization"]
            if abs(current_util - target_util) < 0.1:
                reward = 0.15
            new_state["resource_utilization"] = target_util
        
        else:  # maintain
            # Small reward for stability
            if new_state["queue_length"] < 20 and 0.6 < new_state["resource_utilization"] < 0.8:
                reward = 0.1
        
        # Update throughput
        new_state["throughput"] = new_state["production_rate"] * new_state["resource_utilization"]
        
        # Record optimization
        self.optimization_history.append({
            "action": action,
            "state_before": current_state.copy(),
            "state_after": new_state.copy(),
            "reward": reward,
            "timestamp": datetime.now()
        })
        
        self.performance_history.append({
            "throughput": new_state["throughput"],
            "queue_length": new_state["queue_length"],
            "resource_utilization": new_state["resource_utilization"]
        })
        
        # Record in memory
        self.agent.record_action(
            action_type="flexsim_optimization",
            parameters=action,
            result=f"Applied {action['type']}, throughput: {new_state['throughput']:.1f}",
            reward=reward
        )
        
        return new_state, reward
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get simulation statistics"""
        if not self.performance_history:
            return {}
        
        avg_throughput = np.mean([p["throughput"] for p in self.performance_history])
        avg_queue = np.mean([p["queue_length"] for p in self.performance_history])
        avg_utilization = np.mean([p["resource_utilization"] for p in self.performance_history])
        
        return {
            "total_optimizations": len(self.optimization_history),
            "avg_throughput": avg_throughput,
            "avg_queue_length": avg_queue,
            "avg_utilization": avg_utilization,
            "total_reward": sum([o["reward"] for o in self.optimization_history])
        }


def simulate_flexsim():
    """
    Simulate FlexSim optimization scenario
    """
    print("=" * 70)
    print("FLEXSIM SIMULATION AGENT")
    print("=" * 70)
    
    agent = FlexSimAgent()
    
    # Initial state
    current_state = {
        "production_rate": 100.0,
        "queue_length": 30,
        "resource_utilization": 0.6,
        "throughput": 60.0,
        "bottlenecks": []
    }
    
    # Simulation steps
    simulation_steps = [
        {
            "step": 1,
            "events": ["Production started", "Queue building up"],
            "state_changes": {"queue_length": 45}
        },
        {
            "step": 2,
            "events": ["Bottleneck detected at Station A"],
            "state_changes": {"bottlenecks": ["Station A"], "resource_utilization": 0.85}
        },
        {
            "step": 3,
            "events": ["Queue critical", "Throughput decreasing"],
            "state_changes": {"queue_length": 60, "throughput": 50.0}
        },
        {
            "step": 4,
            "events": ["Optimization applied"],
            "state_changes": {}
        },
        {
            "step": 5,
            "events": ["System stabilized"],
            "state_changes": {"queue_length": 25, "resource_utilization": 0.7, "throughput": 70.0}
        }
    ]
    
    print(f"\n🏭 Initial System State:")
    print(f"   Production Rate: {current_state['production_rate']:.1f} units/hr")
    print(f"   Queue Length: {current_state['queue_length']}")
    print(f"   Resource Utilization: {current_state['resource_utilization']:.1%}")
    print(f"   Throughput: {current_state['throughput']:.1f} units/hr")
    
    # Run simulation
    for step_info in simulation_steps:
        print(f"\n{'='*70}")
        print(f"📊 Simulation Step {step_info['step']}")
        print(f"{'='*70}")
        
        # Update state
        current_state.update(step_info['state_changes'])
        
        # Perceive state
        agent.perceive_system_state(current_state, step_info['events'])
        
        print(f"\n📈 Current State:")
        print(f"   Queue: {current_state['queue_length']}")
        print(f"   Utilization: {current_state['resource_utilization']:.1%}")
        print(f"   Throughput: {current_state['throughput']:.1f} units/hr")
        if current_state.get('bottlenecks'):
            print(f"   Bottlenecks: {', '.join(current_state['bottlenecks'])}")
        
        # Decide optimization
        action = agent.decide_optimization(current_state)
        print(f"\n🤖 Optimization Decision: {action['type'].upper()}")
        if action['parameters']:
            print(f"   Parameters: {action['parameters']}")
        
        # Apply optimization
        current_state, reward = agent.apply_optimization(action, current_state)
        print(f"   Reward: {reward:+.3f}")
        print(f"   New Throughput: {current_state['throughput']:.1f} units/hr")
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 SIMULATION SUMMARY")
    print(f"{'='*70}")
    
    stats = agent.get_statistics()
    print(f"\n💼 Final Statistics:")
    print(f"   Total Optimizations: {stats['total_optimizations']}")
    print(f"   Average Throughput: {stats['avg_throughput']:.1f} units/hr")
    print(f"   Average Queue Length: {stats['avg_queue_length']:.1f}")
    print(f"   Average Utilization: {stats['avg_utilization']:.1%}")
    print(f"   Total Reward: {stats['total_reward']:.3f}")
    
    # Memory statistics
    print(f"\n🧠 Memory System Stats:")
    mem_stats = agent.agent.get_statistics()
    print(f"   Total memories: {mem_stats['memory']['memory_graph']['total_memories']}")
    print(f"   Episodic: {mem_stats['memory']['memory_graph']['episodic_memories']}")
    print(f"   Semantic: {mem_stats['memory']['memory_graph']['semantic_memories']}")
    
    print(f"\n{'='*70}")
    
    return agent


if __name__ == "__main__":
    simulate_flexsim()

