"""
Modern Reinforcement Learning Agents for Text Generation.

This module provides implementations of state-of-the-art RL algorithms
adapted for text generation tasks, including PPO, SAC, TD3, and Rainbow DQN.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3, DQN
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device
import logging

logger = logging.getLogger(__name__)


class TextPolicyNetwork(nn.Module):
    """
    Policy network for text generation using RNN architecture.
    
    This network takes sequence observations and outputs action probabilities
    for the next token in the sequence.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = False
    ):
        """
        Initialize the policy network.
        
        Args:
            vocab_size: Size of vocabulary
            hidden_size: Hidden state size for RNN
            num_layers: Number of RNN layers
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # RNN layers
        self.rnn = nn.GRU(
            hidden_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism (optional)
        if use_attention:
            self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
            
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(
        self, 
        sequence: torch.Tensor, 
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            sequence: Input sequence tensor [batch_size, seq_len]
            hidden: Previous hidden state
            
        Returns:
            Action logits and updated hidden state
        """
        batch_size = sequence.size(0)
        
        # Embedding
        embedded = self.embedding(sequence)
        
        # RNN forward pass
        if hidden is None:
            hidden = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                device=sequence.device, dtype=embedded.dtype
            )
            
        rnn_out, hidden = self.rnn(embedded, hidden)
        
        # Apply attention if enabled
        if self.use_attention:
            attn_out, _ = self.attention(rnn_out, rnn_out, rnn_out)
            rnn_out = rnn_out + attn_out  # Residual connection
            
        # Get last output for action prediction
        last_output = rnn_out[:, -1, :]  # [batch_size, hidden_size]
        
        # Apply dropout and get logits
        last_output = self.dropout(last_output)
        logits = self.fc(last_output)
        
        return logits, hidden


class TextValueNetwork(nn.Module):
    """
    Value network for estimating state values in text generation.
    
    This network estimates the expected return from a given state,
    used in actor-critic methods.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize the value network.
        
        Args:
            vocab_size: Size of vocabulary
            hidden_size: Hidden state size for RNN
            num_layers: Number of RNN layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # RNN layers
        self.rnn = nn.GRU(
            hidden_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(
        self, 
        sequence: torch.Tensor, 
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the value network.
        
        Args:
            sequence: Input sequence tensor [batch_size, seq_len]
            hidden: Previous hidden state
            
        Returns:
            State value and updated hidden state
        """
        batch_size = sequence.size(0)
        
        # Embedding
        embedded = self.embedding(sequence)
        
        # RNN forward pass
        if hidden is None:
            hidden = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                device=sequence.device, dtype=embedded.dtype
            )
            
        rnn_out, hidden = self.rnn(embedded, hidden)
        
        # Get last output for value prediction
        last_output = rnn_out[:, -1, :]  # [batch_size, hidden_size]
        
        # Apply dropout and get value
        last_output = self.dropout(last_output)
        value = self.fc(last_output)
        
        return value, hidden


class TextFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor for text generation environments.
    
    This extractor processes the observation dictionary and extracts
    meaningful features for RL algorithms.
    """
    
    def __init__(
        self, 
        observation_space: gym.Space,
        features_dim: int = 256
    ):
        """
        Initialize the features extractor.
        
        Args:
            observation_space: Gymnasium observation space
            features_dim: Dimension of output features
        """
        super().__init__(observation_space, features_dim)
        
        # Extract dimensions from observation space
        if isinstance(observation_space, gym.spaces.Dict):
            seq_space = observation_space['sequence']
            step_space = observation_space['step']
            done_space = observation_space['done']
            
            self.seq_len = seq_space.shape[0]
            self.vocab_size = seq_space.high[0] + 1
        else:
            raise ValueError("Expected Dict observation space")
            
        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, 64)
        
        # Sequence processing
        self.seq_processor = nn.Sequential(
            nn.Linear(64 * self.seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Step and done processing
        self.step_processor = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        self.done_processor = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        
        # Final feature combination
        self.final_layer = nn.Sequential(
            nn.Linear(64 + 8 + 4, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observations.
        
        Args:
            observations: Batch of observations
            
        Returns:
            Extracted features
        """
        # Extract components
        sequence = observations['sequence'].long()
        step = observations['step'].float()
        done = observations['done'].float()
        
        # Process sequence
        seq_embedded = self.embedding(sequence)  # [batch, seq_len, 64]
        seq_flat = seq_embedded.view(seq_embedded.size(0), -1)  # [batch, seq_len * 64]
        seq_features = self.seq_processor(seq_flat)
        
        # Process step
        step_features = self.step_processor(step)
        
        # Process done
        done_features = self.done_processor(done)
        
        # Combine features
        combined = torch.cat([seq_features, step_features, done_features], dim=-1)
        features = self.final_layer(combined)
        
        return features


class TextPPOAgent:
    """
    Proximal Policy Optimization agent for text generation.
    
    This agent uses PPO with custom policy and value networks
    optimized for text generation tasks.
    """
    
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "auto"
    ):
        """
        Initialize PPO agent.
        
        Args:
            env: Gymnasium environment
            learning_rate: Learning rate for optimizer
            n_steps: Number of steps per update
            batch_size: Batch size for training
            n_epochs: Number of training epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clipping range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to use for training
        """
        self.env = env
        self.device = get_device(device)
        
        # Extract vocabulary size from environment
        vocab_size = env.action_space.n
        
        # Initialize networks
        self.policy_net = TextPolicyNetwork(vocab_size).to(self.device)
        self.value_net = TextValueNetwork(vocab_size).to(self.device)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        # Store hyperparameters
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        
    def collect_rollouts(self) -> Dict[str, torch.Tensor]:
        """
        Collect rollouts for PPO update.
        
        Returns:
            Dictionary containing rollout data
        """
        observations = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []
        
        obs, _ = self.env.reset()
        hidden_policy = None
        hidden_value = None
        
        for step in range(self.n_steps):
            # Convert observation to tensor
            obs_tensor = self._obs_to_tensor(obs)
            
            # Get action from policy
            with torch.no_grad():
                logits, hidden_policy = self.policy_net(obs_tensor['sequence'], hidden_policy)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                # Get value estimate
                value, hidden_value = self.value_net(obs_tensor['sequence'], hidden_value)
                
            # Take action in environment
            next_obs, reward, terminated, truncated, info = self.env.step(action.item())
            
            # Store data
            observations.append(obs_tensor)
            actions.append(action)
            rewards.append(reward)
            dones.append(terminated or truncated)
            values.append(value)
            log_probs.append(log_prob)
            
            obs = next_obs
            
            if terminated or truncated:
                obs, _ = self.env.reset()
                hidden_policy = None
                hidden_value = None
                
        # Convert to tensors
        observations = torch.stack([obs['sequence'] for obs in observations])
        actions = torch.stack(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        values = torch.stack(values).squeeze()
        log_probs = torch.stack(log_probs)
        
        # Calculate advantages and returns
        advantages, returns = self._calculate_gae(rewards, values, dones)
        
        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'values': values,
            'log_probs': log_probs,
            'advantages': advantages,
            'returns': returns
        }
        
    def _calculate_gae(
        self, 
        rewards: torch.Tensor, 
        values: torch.Tensor, 
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Calculate advantages using GAE
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
        
    def _obs_to_tensor(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert observation to tensor."""
        return {
            key: torch.tensor(value, device=self.device).unsqueeze(0)
            for key, value in obs.items()
        }
        
    def update(self, rollout_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update policy and value networks using PPO.
        
        Args:
            rollout_data: Data from rollout collection
            
        Returns:
            Dictionary of training metrics
        """
        observations = rollout_data['observations']
        actions = rollout_data['actions']
        old_log_probs = rollout_data['log_probs']
        advantages = rollout_data['advantages']
        returns = rollout_data['returns']
        
        # Calculate old values for clipping
        with torch.no_grad():
            old_values, _ = self.value_net(observations)
            old_values = old_values.squeeze()
            
        # Training loop
        policy_losses = []
        value_losses = []
        entropies = []
        
        for epoch in range(self.n_epochs):
            # Create batches
            batch_indices = torch.randperm(len(observations))
            
            for start_idx in range(0, len(observations), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(observations))
                batch_idx = batch_indices[start_idx:end_idx]
                
                batch_obs = observations[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_old_values = old_values[batch_idx]
                
                # Policy update
                logits, _ = self.policy_net(batch_obs)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Calculate policy loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value update
                values, _ = self.value_net(batch_obs)
                values = values.squeeze()
                
                # Value loss with clipping
                value_pred_clipped = batch_old_values + torch.clamp(
                    values - batch_old_values, -self.clip_range, self.clip_range
                )
                value_loss1 = F.mse_loss(values, batch_returns)
                value_loss2 = F.mse_loss(value_pred_clipped, batch_returns)
                value_loss = torch.max(value_loss1, value_loss2)
                
                # Total loss
                total_loss = policy_loss - self.ent_coef * entropy + self.vf_coef * value_loss
                
                # Update networks
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                    self.max_grad_norm
                )
                
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
                
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies)
        }
        
    def train(self, total_timesteps: int) -> Dict[str, List[float]]:
        """
        Train the agent for specified number of timesteps.
        
        Args:
            total_timesteps: Total number of timesteps to train
            
        Returns:
            Training metrics
        """
        metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'episode_rewards': [],
            'episode_lengths': []
        }
        
        timesteps = 0
        episode_rewards = []
        episode_lengths = []
        
        while timesteps < total_timesteps:
            # Collect rollouts
            rollout_data = self.collect_rollouts()
            
            # Update networks
            update_metrics = self.update(rollout_data)
            
            # Store metrics
            metrics['policy_loss'].append(update_metrics['policy_loss'])
            metrics['value_loss'].append(update_metrics['value_loss'])
            metrics['entropy'].append(update_metrics['entropy'])
            
            timesteps += self.n_steps
            
            # Log progress
            if len(metrics['policy_loss']) % 10 == 0:
                logger.info(f"Timesteps: {timesteps}, Policy Loss: {update_metrics['policy_loss']:.4f}")
                
        return metrics
