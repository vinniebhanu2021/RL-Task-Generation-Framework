"""
Logging utilities for RL text generation project.

This module provides comprehensive logging setup with support for
TensorBoard, Weights & Biases, and file logging.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.utils.config import LoggingConfig


class MetricsLogger:
    """Logger for training metrics with multiple backends."""
    
    def __init__(self, config: LoggingConfig, log_dir: str):
        """
        Initialize metrics logger.
        
        Args:
            config: Logging configuration
            log_dir: Directory for log files
        """
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard
        self.tb_writer = None
        if config.use_tensorboard and TENSORBOARD_AVAILABLE:
            tb_log_dir = self.log_dir / "tensorboard"
            self.tb_writer = SummaryWriter(str(tb_log_dir))
            
        # Initialize Weights & Biases
        self.wandb_run = None
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                dir=str(self.log_dir)
            )
            self.wandb_run = wandb.run
            
        # File logging
        self.metrics_file = self.log_dir / "metrics.json"
        self.metrics_history = []
        
    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """
        Log metrics to all configured backends.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current training step
        """
        # Add timestamp
        metrics['step'] = step
        metrics['timestamp'] = time.time()
        
        # Log to TensorBoard
        if self.tb_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)
                    
        # Log to Weights & Biases
        if self.wandb_run:
            wandb.log(metrics, step=step)
            
        # Log to file
        self.metrics_history.append(metrics.copy())
        
        # Save metrics to file periodically
        if len(self.metrics_history) % 100 == 0:
            self._save_metrics()
            
    def log_text(self, text: str, step: int, tag: str = "generated_text") -> None:
        """
        Log generated text.
        
        Args:
            text: Generated text to log
            step: Current training step
            tag: Tag for the text
        """
        if self.tb_writer:
            self.tb_writer.add_text(tag, text, step)
            
        if self.wandb_run:
            wandb.log({tag: wandb.Html(f"<p>{text}</p>")}, step=step)
            
    def log_histogram(self, values: list, step: int, tag: str) -> None:
        """
        Log histogram of values.
        
        Args:
            values: Values to create histogram from
            step: Current training step
            tag: Tag for the histogram
        """
        if self.tb_writer:
            self.tb_writer.add_histogram(tag, values, step)
            
        if self.wandb_run:
            wandb.log({tag: wandb.Histogram(values)}, step=step)
            
    def _save_metrics(self) -> None:
        """Save metrics history to file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
            
    def close(self) -> None:
        """Close all logging backends."""
        if self.tb_writer:
            self.tb_writer.close()
            
        if self.wandb_run:
            wandb.finish()
            
        # Save final metrics
        self._save_metrics()


def setup_logging(config: LoggingConfig, log_dir: str) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        config: Logging configuration
        log_dir: Directory for log files
        
    Returns:
        Configured logger
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure logging level
    log_level = getattr(logging, config.log_level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if config.log_to_file:
        log_file = log_path / config.log_file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
    return root_logger


class TrainingLogger:
    """High-level training logger with episode tracking."""
    
    def __init__(self, config: LoggingConfig, log_dir: str):
        """
        Initialize training logger.
        
        Args:
            config: Logging configuration
            log_dir: Directory for log files
        """
        self.config = config
        self.metrics_logger = MetricsLogger(config, log_dir)
        self.logger = logging.getLogger(__name__)
        
        # Episode tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        # Training metrics
        self.training_step = 0
        self.episode_count = 0
        
    def log_episode_step(self, reward: float, done: bool, info: Dict[str, Any]) -> None:
        """
        Log a single step within an episode.
        
        Args:
            reward: Reward received
            done: Whether episode is done
            info: Additional info from environment
        """
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        if done:
            self.log_episode_end(info)
            
    def log_episode_end(self, info: Dict[str, Any]) -> None:
        """
        Log the end of an episode.
        
        Args:
            info: Episode info from environment
        """
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_lengths.append(self.current_episode_length)
        
        # Check if episode was successful
        success = self.current_episode_reward > 0
        self.episode_successes.append(success)
        
        self.episode_count += 1
        
        # Log episode metrics
        episode_metrics = {
            'episode_reward': self.current_episode_reward,
            'episode_length': self.current_episode_length,
            'episode_success': success,
            'episode_count': self.episode_count
        }
        
        # Add info from environment
        if 'sequence_text' in info:
            episode_metrics['sequence_text'] = info['sequence_text']
            
        self.metrics_logger.log_metrics(episode_metrics, self.episode_count)
        
        # Log generated text
        if 'sequence_text' in info:
            self.metrics_logger.log_text(
                info['sequence_text'], 
                self.episode_count, 
                'generated_sequence'
            )
            
        # Reset for next episode
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        # Log progress periodically
        if self.episode_count % 100 == 0:
            self._log_progress()
            
    def log_training_update(self, update_metrics: Dict[str, float]) -> None:
        """
        Log training update metrics.
        
        Args:
            update_metrics: Metrics from training update
        """
        self.training_step += 1
        
        # Add episode statistics
        if self.episode_rewards:
            update_metrics.update({
                'avg_episode_reward': sum(self.episode_rewards[-100:]) / min(100, len(self.episode_rewards)),
                'avg_episode_length': sum(self.episode_lengths[-100:]) / min(100, len(self.episode_lengths)),
                'success_rate': sum(self.episode_successes[-100:]) / min(100, len(self.episode_successes))
            })
            
        self.metrics_logger.log_metrics(update_metrics, self.training_step)
        
    def _log_progress(self) -> None:
        """Log training progress summary."""
        if not self.episode_rewards:
            return
            
        recent_rewards = self.episode_rewards[-100:]
        recent_lengths = self.episode_lengths[-100:]
        recent_successes = self.episode_successes[-100:]
        
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        avg_length = sum(recent_lengths) / len(recent_lengths)
        success_rate = sum(recent_successes) / len(recent_successes)
        
        self.logger.info(
            f"Episode {self.episode_count}: "
            f"Avg Reward: {avg_reward:.3f}, "
            f"Avg Length: {avg_length:.1f}, "
            f"Success Rate: {success_rate:.3f}"
        )
        
    def close(self) -> None:
        """Close the logger."""
        self.metrics_logger.close()
        
        # Log final statistics
        if self.episode_rewards:
            self.logger.info(f"Training completed:")
            self.logger.info(f"  Total episodes: {self.episode_count}")
            self.logger.info(f"  Average reward: {sum(self.episode_rewards) / len(self.episode_rewards):.3f}")
            self.logger.info(f"  Average length: {sum(self.episode_lengths) / len(self.episode_lengths):.1f}")
            self.logger.info(f"  Success rate: {sum(self.episode_successes) / len(self.episode_successes):.3f}")


def create_experiment_logger(
    experiment_name: str,
    config: LoggingConfig,
    base_log_dir: str = "logs"
) -> TrainingLogger:
    """
    Create a logger for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment
        config: Logging configuration
        base_log_dir: Base directory for logs
        
    Returns:
        Configured training logger
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(base_log_dir) / f"{experiment_name}_{timestamp}"
    
    return TrainingLogger(config, str(log_dir))
