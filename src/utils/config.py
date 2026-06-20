"""
Configuration management for RL text generation project.

This module provides YAML-based configuration management with validation
and default settings for all components.
"""

from typing import Dict, Any, Optional, List
import yaml
import os
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """Configuration for text generation environment."""
    vocab_size: int = 64
    max_sequence_length: int = 20
    target_word: Optional[str] = "you"
    target_word_index: Optional[int] = None
    reward_type: str = "binary"  # "binary", "length_penalty", "semantic"
    length_penalty: float = 0.1
    early_stop_reward: float = 0.0
    invalid_action_penalty: float = -0.1
    use_advanced_env: bool = False
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "target_completion": 1.0,
        "length_efficiency": 0.1,
        "diversity": 0.05,
        "grammar": 0.2
    })


@dataclass
class AgentConfig:
    """Configuration for RL agent."""
    algorithm: str = "ppo"  # "ppo", "sac", "td3", "dqn", "rainbow"
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 100000
    gamma: float = 0.99
    tau: float = 0.005
    target_update_interval: int = 1000
    
    # PPO specific
    n_steps: int = 2048
    n_epochs: int = 10
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # SAC specific
    target_entropy: Optional[float] = None
    learning_starts: int = 1000
    
    # DQN specific
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    target_update_interval: int = 1000
    
    # Network architecture
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    use_attention: bool = False


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    total_timesteps: int = 100000
    eval_frequency: int = 10000
    eval_episodes: int = 10
    save_frequency: int = 50000
    log_frequency: int = 1000
    
    # Early stopping
    early_stopping: bool = False
    patience: int = 10
    min_improvement: float = 0.01
    
    # Checkpointing
    save_best_model: bool = True
    save_replay_buffer: bool = False


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "logs/training.log"
    
    # TensorBoard
    use_tensorboard: bool = True
    tensorboard_log_dir: str = "tensorboard_logs"
    
    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = "rl-text-generation"
    wandb_entity: Optional[str] = None
    
    # Visualization
    plot_frequency: int = 1000
    save_plots: bool = True
    plot_dir: str = "plots"


@dataclass
class Config:
    """Main configuration class."""
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # General settings
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda"
    num_envs: int = 1
    verbose: int = 1


class ConfigManager:
    """Manager for loading, saving, and validating configurations."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = Config()
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
            
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                
            self._update_config_from_dict(config_dict)
            logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
            
    def save_config(self, config_path: str) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            config_path: Path to save configuration
        """
        try:
            config_dict = self._config_to_dict()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise
            
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config object to dictionary."""
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, dict):
                return {k: dataclass_to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [dataclass_to_dict(item) for item in obj]
            else:
                return obj
                
        return dataclass_to_dict(self.config)
        
    def _update_config_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update config object from dictionary."""
        def update_dataclass(obj, data):
            if hasattr(obj, '__dataclass_fields__'):
                for field_name, field_value in data.items():
                    if hasattr(obj, field_name):
                        current_value = getattr(obj, field_name)
                        if hasattr(current_value, '__dataclass_fields__'):
                            update_dataclass(current_value, field_value)
                        else:
                            setattr(obj, field_name, field_value)
                            
        update_dataclass(self.config, config_dict)
        
    def validate_config(self) -> List[str]:
        """
        Validate configuration and return list of warnings/errors.
        
        Returns:
            List of validation messages
        """
        warnings = []
        
        # Validate environment config
        if self.config.environment.vocab_size <= 0:
            warnings.append("vocab_size must be positive")
            
        if self.config.environment.max_sequence_length <= 0:
            warnings.append("max_sequence_length must be positive")
            
        if self.config.environment.target_word and self.config.environment.target_word_index:
            warnings.append("Both target_word and target_word_index specified, target_word will be used")
            
        # Validate agent config
        if self.config.agent.learning_rate <= 0:
            warnings.append("learning_rate must be positive")
            
        if self.config.agent.batch_size <= 0:
            warnings.append("batch_size must be positive")
            
        if not 0 <= self.config.agent.gamma <= 1:
            warnings.append("gamma must be between 0 and 1")
            
        # Validate training config
        if self.config.training.total_timesteps <= 0:
            warnings.append("total_timesteps must be positive")
            
        if self.config.training.eval_frequency <= 0:
            warnings.append("eval_frequency must be positive")
            
        return warnings
        
    def get_default_config_path(self) -> str:
        """Get path to default configuration file."""
        return os.path.join("configs", "default.yaml")
        
    def create_default_config(self) -> None:
        """Create default configuration file."""
        default_path = self.get_default_config_path()
        self.save_config(default_path)
        logger.info(f"Default configuration created at {default_path}")


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or create default.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration object
    """
    manager = ConfigManager(config_path)
    
    if config_path is None:
        default_path = manager.get_default_config_path()
        if not os.path.exists(default_path):
            manager.create_default_config()
        else:
            manager.load_config(default_path)
            
    # Validate configuration
    warnings = manager.validate_config()
    if warnings:
        logger.warning(f"Configuration warnings: {warnings}")
        
    return manager.config


# Predefined configurations for common use cases
def get_quick_test_config() -> Config:
    """Get configuration for quick testing."""
    config = Config()
    config.environment.vocab_size = 20
    config.environment.max_sequence_length = 10
    config.training.total_timesteps = 10000
    config.training.eval_frequency = 2000
    config.agent.n_steps = 512
    config.agent.batch_size = 32
    return config


def get_production_config() -> Config:
    """Get configuration for production training."""
    config = Config()
    config.environment.vocab_size = 1000
    config.environment.max_sequence_length = 50
    config.training.total_timesteps = 1000000
    config.training.eval_frequency = 50000
    config.agent.n_steps = 4096
    config.agent.batch_size = 128
    config.agent.use_attention = True
    config.logging.use_wandb = True
    return config


def get_research_config() -> Config:
    """Get configuration for research experiments."""
    config = Config()
    config.environment.use_advanced_env = True
    config.agent.algorithm = "ppo"
    config.agent.use_attention = True
    config.training.total_timesteps = 2000000
    config.logging.use_tensorboard = True
    config.logging.use_wandb = True
    config.logging.save_plots = True
    return config
