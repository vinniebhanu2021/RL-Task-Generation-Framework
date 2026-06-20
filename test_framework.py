"""
Simple test script to verify the RL text generation framework works correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import numpy as np
from src.envs.text_generation_env import TextGenerationEnv, TextGenConfig
from src.agents.text_ppo_agent import TextPPOAgent


def test_environment():
    """Test basic environment functionality."""
    print("Testing environment...")
    
    config = TextGenConfig(
        vocab_size=10,
        max_sequence_length=5,
        target_word="love",
        reward_type="binary"
    )
    
    env = TextGenerationEnv(config)
    
    # Test reset
    obs, info = env.reset()
    assert isinstance(obs, dict)
    assert 'sequence' in obs
    assert 'step' in obs
    assert 'done' in obs
    print("✅ Environment reset works")
    
    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    print("✅ Environment step works")
    
    env.close()
    print("✅ Environment test passed")


def test_agent():
    """Test basic agent functionality."""
    print("Testing agent...")
    
    config = TextGenConfig(
        vocab_size=10,
        max_sequence_length=5,
        target_word="love"
    )
    
    env = TextGenerationEnv(config)
    agent = TextPPOAgent(env, learning_rate=0.001, n_steps=32, batch_size=8, n_epochs=2)
    
    # Test rollout collection
    rollout_data = agent.collect_rollouts()
    assert 'observations' in rollout_data
    assert 'actions' in rollout_data
    assert 'rewards' in rollout_data
    print("✅ Agent rollout collection works")
    
    # Test update
    update_metrics = agent.update(rollout_data)
    assert 'policy_loss' in update_metrics
    assert 'value_loss' in update_metrics
    assert 'entropy' in update_metrics
    print("✅ Agent update works")
    
    env.close()
    print("✅ Agent test passed")


def test_training():
    """Test short training run."""
    print("Testing training...")
    
    config = TextGenConfig(
        vocab_size=10,
        max_sequence_length=5,
        target_word="love"
    )
    
    env = TextGenerationEnv(config)
    agent = TextPPOAgent(env, learning_rate=0.001, n_steps=32, batch_size=8, n_epochs=2)
    
    # Train for a few updates
    metrics = agent.train(100)  # Very short training
    
    assert isinstance(metrics, dict)
    assert 'policy_loss' in metrics
    assert 'value_loss' in metrics
    assert 'entropy' in metrics
    print("✅ Training works")
    
    env.close()
    print("✅ Training test passed")


def main():
    """Run all tests."""
    print("🧪 Running RL Text Generation Tests")
    print("=" * 50)
    
    try:
        test_environment()
        test_agent()
        test_training()
        
        print("=" * 50)
        print("🎉 All tests passed!")
        print("The RL text generation framework is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
