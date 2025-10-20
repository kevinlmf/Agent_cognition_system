"""
Basic tests for the perception and memory system
"""
import sys
sys.path.append('.')

from perception.perception_main import PerceptionEngine
from memory.memory_main import MemoryEngine
from main import CognitiveAgent


def test_perception_engine():
    """Test basic perception functionality"""
    print("Testing PerceptionEngine...")

    engine = PerceptionEngine()

    # Test text perception
    result = engine.perceive_text("The Federal Reserve announced rate changes.")

    assert 'summary' in result
    assert 'entities' in result
    assert len(result['entities']) > 0

    print("✓ PerceptionEngine working")


def test_memory_engine():
    """Test basic memory functionality"""
    print("Testing MemoryEngine...")

    memory = MemoryEngine()

    # Store experience
    perception = {"summary": "Test event", "entities": []}
    memory_id = memory.store_experience(
        world_snapshot={},
        perception_result=perception,
        reward=0.5
    )

    assert memory_id is not None
    assert len(memory.memory_graph.memories) > 0

    # Retrieve
    recent = memory.retrieve(retrieval_strategy="recent", top_k=5)
    assert len(recent) > 0

    print("✓ MemoryEngine working")


def test_cognitive_agent():
    """Test integrated cognitive agent"""
    print("Testing CognitiveAgent...")

    agent = CognitiveAgent(mode="langgraph")

    # Perceive
    result = agent.perceive("Market data shows positive trends.")
    assert 'perception' in result

    # Query
    context = agent.query_memory("market")
    assert context is not None

    # Record action
    memory_id = agent.record_action(
        action_type="test",
        parameters={},
        reward=1.0
    )
    assert memory_id is not None

    print("✓ CognitiveAgent working")


def test_world_graph():
    """Test WorldGraph functionality"""
    print("Testing WorldGraph...")

    from perception.world_model.world_graph import WorldGraph

    wg = WorldGraph()

    # Add entities
    wg.add_entity("Federal Reserve", entity_type="organization")
    wg.add_entity("Interest Rate", entity_type="financial_indicator")

    # Add relation
    wg.add_relation("Federal Reserve", "controls", "Interest Rate")

    # Query
    relations = wg.get_relations("Federal Reserve")
    assert len(relations) > 0

    stats = wg.get_statistics()
    assert stats['num_entities'] == 2
    assert stats['num_relations'] == 1

    print("✓ WorldGraph working")


def test_memory_graph():
    """Test MemoryGraph functionality"""
    print("Testing MemoryGraph...")

    from memory.memory_graph import MemoryGraph
    import numpy as np

    mg = MemoryGraph()

    # Add episodic memory
    memory_id = mg.add_memory(
        memory_type="episodic",
        content="Test experience",
        summary="Test",
        importance=0.8
    )

    assert memory_id in mg.memories

    # Add semantic memory
    semantic_id = mg.add_memory(
        memory_type="semantic",
        content="Learned pattern",
        summary="Pattern",
        importance=0.9
    )

    # Link memories
    mg.add_relation(memory_id, "abstracted_to", semantic_id)

    stats = mg.get_statistics()
    assert stats['total_memories'] == 2

    print("✓ MemoryGraph working")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("RUNNING BASIC TESTS")
    print("=" * 60)
    print()

    try:
        test_world_graph()
        test_memory_graph()
        test_perception_engine()
        test_memory_engine()
        test_cognitive_agent()

        print()
        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        return True

    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
