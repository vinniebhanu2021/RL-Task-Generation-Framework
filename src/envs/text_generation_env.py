"""
Text Generation Environment for Reinforcement Learning.

This module provides a Gymnasium-compatible environment for training RL agents
to generate text sequences with specific properties or constraints.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TextGenConfig:
    """Configuration for text generation environment."""
    vocab_size: int = 1000
    max_sequence_length: int = 20
    target_word: Optional[str] = None
    target_word_index: Optional[int] = None
    reward_type: str = "binary"  # "binary", "length_penalty", "semantic"
    length_penalty: float = 0.1
    early_stop_reward: float = 0.0
    invalid_action_penalty: float = -0.1


class TextGenerationEnv(gym.Env):
    """
    A Gymnasium environment for text generation using reinforcement learning.
    
    The agent learns to generate sequences of tokens that satisfy certain criteria,
    such as ending with a specific word or following grammatical patterns.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        config: TextGenConfig,
        vocab: Optional[List[str]] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the text generation environment.
        
        Args:
            config: Configuration object for the environment
            vocab: Optional vocabulary list. If None, uses default vocabulary
            render_mode: Rendering mode for visualization
        """
        super().__init__()
        
        self.config = config
        self.render_mode = render_mode
        
        # Set up vocabulary
        if vocab is None:
            self.vocab = [
                'hello', 'world', 'i', 'love', 'you', 'bye', 'good', 'morning',
                'night', 'day', 'sun', 'moon', 'star', 'sky', 'earth', 'water',
                'fire', 'wind', 'tree', 'flower', 'bird', 'cat', 'dog', 'fish',
                'car', 'house', 'book', 'music', 'art', 'dream', 'hope', 'peace',
                'joy', 'smile', 'laugh', 'cry', 'think', 'feel', 'know', 'see',
                'hear', 'touch', 'taste', 'smell', 'walk', 'run', 'jump', 'fly',
                'swim', 'dance', 'sing', 'play', 'work', 'rest', 'sleep', 'wake'
            ]
        else:
            self.vocab = vocab
            
        # Ensure vocab size matches config
        if len(self.vocab) != config.vocab_size:
            logger.warning(f"Vocabulary size ({len(self.vocab)}) doesn't match config ({config.vocab_size})")
            self.vocab = self.vocab[:config.vocab_size]
            
        self.vocab_size = len(self.vocab)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        # Set target word
        if config.target_word and config.target_word in self.word_to_idx:
            self.target_idx = self.word_to_idx[config.target_word]
        elif config.target_word_index is not None:
            self.target_idx = config.target_word_index
        else:
            self.target_idx = 0  # Default to first word
            
        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.vocab_size)
        self.observation_space = spaces.Dict({
            'sequence': spaces.Box(
                low=0, high=self.vocab_size-1, 
                shape=(config.max_sequence_length,), 
                dtype=np.int32
            ),
            'step': spaces.Box(low=0, high=config.max_sequence_length-1, dtype=np.int32),
            'done': spaces.Box(low=0, high=1, dtype=np.int32)
        })
        
        # Initialize state
        self.reset()
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Initial observation and info dictionary
        """
        super().reset(seed=seed)
        
        # Initialize sequence with special start token (index 0)
        self.sequence = np.zeros(self.config.max_sequence_length, dtype=np.int32)
        self.current_step = 0
        self.done = False
        
        observation = self._get_observation()
        info = {"sequence_length": 0, "target_word": self.idx_to_word[self.target_idx]}
        
        return observation, info
        
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Token index to add to sequence
            
        Returns:
            Observation, reward, terminated, truncated, info
        """
        if self.done:
            logger.warning("Episode already finished. Call reset() before step().")
            return self._get_observation(), 0.0, True, False, {}
            
        # Validate action
        if not self.action_space.contains(action):
            logger.warning(f"Invalid action {action}. Using action 0 instead.")
            action = 0
            
        # Add action to sequence
        if self.current_step < self.config.max_sequence_length:
            self.sequence[self.current_step] = action
            self.current_step += 1
            
        # Check if episode is done
        terminated = self._is_episode_done()
        truncated = self.current_step >= self.config.max_sequence_length
        
        # Calculate reward
        reward = self._calculate_reward(terminated, truncated)
        
        # Update done status
        self.done = terminated or truncated
        
        observation = self._get_observation()
        info = {
            "sequence_length": self.current_step,
            "current_word": self.idx_to_word[action],
            "target_word": self.idx_to_word[self.target_idx],
            "sequence_text": self._sequence_to_text()
        }
        
        return observation, reward, terminated, truncated, info
        
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        return {
            'sequence': self.sequence.copy(),
            'step': np.array([self.current_step], dtype=np.int32),
            'done': np.array([1 if self.done else 0], dtype=np.int32)
        }
        
    def _is_episode_done(self) -> bool:
        """Check if episode should terminate."""
        if self.current_step == 0:
            return False
            
        # Check if last word is target word
        last_word_idx = self.sequence[self.current_step - 1]
        return last_word_idx == self.target_idx
        
    def _calculate_reward(self, terminated: bool, truncated: bool) -> float:
        """Calculate reward based on current state."""
        if terminated:
            return 1.0  # Success reward
        elif truncated:
            return self.config.early_stop_reward
        else:
            # Intermediate reward based on type
            if self.config.reward_type == "length_penalty":
                return -self.config.length_penalty  # Penalty for longer sequences
            elif self.config.reward_type == "semantic":
                # Could implement semantic similarity reward here
                return 0.0
            else:  # binary
                return 0.0
                
    def _sequence_to_text(self) -> str:
        """Convert current sequence to readable text."""
        words = []
        for i in range(self.current_step):
            if self.sequence[i] < len(self.vocab):
                words.append(self.idx_to_word[self.sequence[i]])
        return " ".join(words)
        
    def render(self) -> Optional[Union[str, np.ndarray]]:
        """Render the environment."""
        if self.render_mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Sequence: {self._sequence_to_text()}")
            print(f"Target: {self.idx_to_word[self.target_idx]}")
            print(f"Done: {self.done}")
            print("-" * 50)
        elif self.render_mode == "rgb_array":
            # Could implement visual representation here
            return np.zeros((100, 100, 3), dtype=np.uint8)
        return None
        
    def close(self):
        """Clean up environment resources."""
        pass


class AdvancedTextGenEnv(TextGenerationEnv):
    """
    Advanced text generation environment with more sophisticated reward functions
    and multi-objective optimization capabilities.
    """
    
    def __init__(
        self,
        config: TextGenConfig,
        vocab: Optional[List[str]] = None,
        render_mode: Optional[str] = None,
        reward_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize advanced text generation environment.
        
        Args:
            config: Configuration object
            vocab: Vocabulary list
            render_mode: Rendering mode
            reward_weights: Weights for different reward components
        """
        super().__init__(config, vocab, render_mode)
        
        self.reward_weights = reward_weights or {
            "target_completion": 1.0,
            "length_efficiency": 0.1,
            "diversity": 0.05,
            "grammar": 0.2
        }
        
        # Track generated sequences for diversity calculation
        self.generated_sequences = []
        
    def _calculate_reward(self, terminated: bool, truncated: bool) -> float:
        """Calculate multi-component reward."""
        total_reward = 0.0
        
        # Target completion reward
        if terminated:
            total_reward += self.reward_weights["target_completion"]
        elif truncated:
            total_reward += self.config.early_stop_reward
            
        # Length efficiency reward (shorter sequences are better)
        if self.current_step > 0:
            efficiency = 1.0 / self.current_step
            total_reward += self.reward_weights["length_efficiency"] * efficiency
            
        # Diversity reward (encourage different sequences)
        current_text = self._sequence_to_text()
        if current_text not in self.generated_sequences:
            total_reward += self.reward_weights["diversity"]
            self.generated_sequences.append(current_text)
            
        # Grammar reward (simple heuristic - could be replaced with actual grammar checker)
        grammar_score = self._calculate_grammar_score()
        total_reward += self.reward_weights["grammar"] * grammar_score
        
        return total_reward
        
    def _calculate_grammar_score(self) -> float:
        """Calculate a simple grammar score based on word patterns."""
        if self.current_step < 2:
            return 0.0
            
        # Simple heuristic: check for common word patterns
        words = [self.idx_to_word[self.sequence[i]] for i in range(self.current_step)]
        
        # Check for basic patterns
        score = 0.0
        
        # Prefer sequences that start with common words
        if words[0] in ['i', 'hello', 'good', 'the', 'a']:
            score += 0.1
            
        # Prefer sequences with verbs
        verbs = ['love', 'see', 'hear', 'walk', 'run', 'jump', 'fly', 'swim', 'dance', 'sing', 'play', 'work']
        if any(word in verbs for word in words):
            score += 0.2
            
        # Prefer sequences with nouns
        nouns = ['world', 'day', 'night', 'sun', 'moon', 'star', 'sky', 'earth', 'water', 'tree', 'flower', 'bird', 'cat', 'dog']
        if any(word in nouns for word in words):
            score += 0.1
            
        return min(score, 1.0)  # Cap at 1.0
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Reset environment and clear generated sequences history."""
        observation, info = super().reset(seed, options)
        self.generated_sequences = []
        return observation, info
