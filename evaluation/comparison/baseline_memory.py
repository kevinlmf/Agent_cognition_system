"""
Baseline Memory Implementations
实现常见的memory架构用于对比：
- LSTM/RNN
- Transformer
- Memory Networks
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque

# Try to import torch (optional dependency)
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    print("Warning: PyTorch not available, using numpy implementations")


class LSTMMemory:
    """
    LSTM-based Memory Baseline
    使用LSTM的隐藏状态作为memory
    """
    
    def __init__(self, hidden_size: int = 128, num_layers: int = 1):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        if HAS_TORCH:
            self.lstm = nn.LSTM(
                input_size=128,  # 假设输入维度
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
            self.hidden = None
        else:
            # NumPy实现（简化版）
            self.hidden_state = np.zeros(hidden_size)
            self.cell_state = np.zeros(hidden_size)
            self.history = deque(maxlen=1000)
    
    def store(self, observation: np.ndarray, content: str = None):
        """存储观察（隐式存储在LSTM状态中）"""
        if HAS_TORCH:
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).unsqueeze(0)
            if self.hidden is None:
                output, self.hidden = self.lstm(obs_tensor)
            else:
                output, self.hidden = self.lstm(obs_tensor, self.hidden)
        else:
            # NumPy实现：简单更新
            self.history.append(observation)
            # 简化的状态更新
            self.hidden_state = 0.9 * self.hidden_state + 0.1 * observation[:self.hidden_size]
    
    def retrieve(self, query: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        检索memory（LSTM无法显式检索，只能使用当前隐藏状态）
        """
        if HAS_TORCH:
            if self.hidden is None:
                return []
            # LSTM只能返回当前隐藏状态
            hidden = self.hidden[0].squeeze().detach().numpy()
            return [{
                'content': 'LSTM hidden state',
                'similarity': 1.0,
                'hidden_state': hidden
            }]
        else:
            # NumPy实现
            if len(self.history) == 0:
                return []
            
            # 计算与query的相似度
            similarities = []
            for i, obs in enumerate(self.history):
                sim = np.dot(query, obs) / (np.linalg.norm(query) * np.linalg.norm(obs) + 1e-8)
                similarities.append((i, sim))
            
            # 返回top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [{
                'content': f'LSTM memory {idx}',
                'similarity': sim,
                'index': idx
            } for idx, sim in similarities[:top_k]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if HAS_TORCH:
            return {
                'type': 'LSTM',
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'has_hidden_state': self.hidden is not None
            }
        else:
            return {
                'type': 'LSTM',
                'hidden_size': self.hidden_size,
                'history_length': len(self.history),
                'hidden_state_norm': np.linalg.norm(self.hidden_state)
            }


class TransformerMemory:
    """
    Transformer-based Memory Baseline
    使用Transformer的self-attention机制
    """
    
    def __init__(self, d_model: int = 128, nhead: int = 8, num_layers: int = 2):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.memory_buffer = deque(maxlen=512)  # 限制长度
        
        if HAS_TORCH:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            # NumPy实现（简化版）
            self.attention_weights = {}
    
    def store(self, observation: np.ndarray, content: str = None):
        """存储观察到memory buffer"""
        self.memory_buffer.append(observation)
    
    def retrieve(self, query: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        使用attention机制检索
        """
        if len(self.memory_buffer) == 0:
            return []
        
        memories = np.array(list(self.memory_buffer))
        
        if HAS_TORCH:
            # 使用Transformer attention
            mem_tensor = torch.FloatTensor(memories).unsqueeze(0)
            query_tensor = torch.FloatTensor(query).unsqueeze(0).unsqueeze(0)
            
            # 计算attention
            output = self.transformer(mem_tensor)
            # 简化：返回最近的memories
            return [{
                'content': f'Transformer memory {i}',
                'similarity': 1.0 / (i + 1),
                'index': i
            } for i in range(min(top_k, len(memories)))]
        else:
            # NumPy实现：计算相似度
            similarities = []
            for i, mem in enumerate(memories):
                sim = np.dot(query, mem) / (np.linalg.norm(query) * np.linalg.norm(mem) + 1e-8)
                similarities.append((i, sim))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [{
                'content': f'Transformer memory {idx}',
                'similarity': sim,
                'index': idx
            } for idx, sim in similarities[:top_k]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'type': 'Transformer',
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'memory_buffer_size': len(self.memory_buffer),
            'max_capacity': 512
        }


class MemoryNetworkBaseline:
    """
    Memory Networks Baseline
    使用显式的外部memory
    """
    
    def __init__(self, memory_size: int = 1000, embedding_dim: int = 128):
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        self.memories = []
        self.embeddings = []
    
    def store(self, observation: np.ndarray, content: str = None):
        """存储到显式memory"""
        if len(self.memories) >= self.memory_size:
            # 移除最旧的
            self.memories.pop(0)
            self.embeddings.pop(0)
        
        self.memories.append({
            'content': content or f'Memory {len(self.memories)}',
            'embedding': observation
        })
        self.embeddings.append(observation)
    
    def retrieve(self, query: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        基于相似度检索
        """
        if len(self.embeddings) == 0:
            return []
        
        similarities = []
        for i, emb in enumerate(self.embeddings):
            sim = np.dot(query, emb) / (np.linalg.norm(query) * np.linalg.norm(emb) + 1e-8)
            similarities.append((i, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [{
            'content': self.memories[idx]['content'],
            'similarity': sim,
            'index': idx
        } for idx, sim in similarities[:top_k]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'type': 'Memory Networks',
            'memory_size': len(self.memories),
            'max_capacity': self.memory_size,
            'embedding_dim': self.embedding_dim
        }


class EpisodicMemoryBaseline:
    """
    Simple Episodic Memory Baseline
    简单的episodic memory实现（用于对比）
    """
    
    def __init__(self, max_memories: int = 10000):
        self.max_memories = max_memories
        self.memories = []
    
    def store(self, observation: np.ndarray, content: str = None, 
             reward: float = None, importance: float = 0.5):
        """存储episode"""
        if len(self.memories) >= self.max_memories:
            # 移除最旧的
            self.memories.pop(0)
        
        self.memories.append({
            'content': content or f'Episode {len(self.memories)}',
            'embedding': observation,
            'reward': reward,
            'importance': importance,
            'index': len(self.memories)
        })
    
    def retrieve(self, query: np.ndarray, top_k: int = 5) -> List[Dict]:
        """基于相似度检索"""
        if len(self.memories) == 0:
            return []
        
        similarities = []
        for mem in self.memories:
            emb = mem['embedding']
            sim = np.dot(query, emb) / (np.linalg.norm(query) * np.linalg.norm(emb) + 1e-8)
            # 结合importance
            weighted_sim = sim * (1 + mem.get('importance', 0.5))
            similarities.append((mem, weighted_sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [{
            'content': mem['content'],
            'similarity': sim,
            'importance': mem.get('importance', 0.5),
            'index': mem['index']
        } for mem, sim in similarities[:top_k]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.memories:
            return {'type': 'Episodic Memory', 'total_memories': 0}
        
        importances = [m.get('importance', 0.5) for m in self.memories]
        return {
            'type': 'Episodic Memory',
            'total_memories': len(self.memories),
            'avg_importance': np.mean(importances),
            'max_capacity': self.max_memories
        }

