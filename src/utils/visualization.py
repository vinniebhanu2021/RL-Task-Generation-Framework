"""
Visualization utilities for RL text generation project.

This module provides comprehensive visualization tools for training metrics,
learning curves, and generated text analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import logging

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


class TrainingVisualizer:
    """Visualizer for training metrics and results."""
    
    def __init__(self, log_dir: str):
        """
        Initialize training visualizer.
        
        Args:
            log_dir: Directory containing training logs
        """
        self.log_dir = Path(log_dir)
        self.metrics_data = self._load_metrics()
        
    def _load_metrics(self) -> pd.DataFrame:
        """Load metrics from log directory."""
        metrics_file = self.log_dir / "metrics.json"
        
        if not metrics_file.exists():
            logger.warning(f"Metrics file not found: {metrics_file}")
            return pd.DataFrame()
            
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                
            return pd.DataFrame(metrics)
            
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
            return pd.DataFrame()
            
    def plot_training_curves(
        self, 
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """
        Plot training curves for rewards, losses, and other metrics.
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size
        """
        if self.metrics_data.empty:
            logger.warning("No metrics data available for plotting")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Training Curves', fontsize=16, fontweight='bold')
        
        # Episode rewards
        if 'episode_reward' in self.metrics_data.columns:
            episode_rewards = self.metrics_data['episode_reward'].dropna()
            if not episode_rewards.empty:
                # Smooth the curve
                window_size = min(50, len(episode_rewards) // 10)
                if window_size > 1:
                    smoothed = episode_rewards.rolling(window=window_size, center=True).mean()
                    axes[0, 0].plot(smoothed.index, smoothed.values, alpha=0.8, linewidth=2)
                else:
                    axes[0, 0].plot(episode_rewards.index, episode_rewards.values, alpha=0.8, linewidth=2)
                axes[0, 0].set_title('Episode Rewards')
                axes[0, 0].set_xlabel('Episode')
                axes[0, 0].set_ylabel('Reward')
                axes[0, 0].grid(True, alpha=0.3)
                
        # Episode lengths
        if 'episode_length' in self.metrics_data.columns:
            episode_lengths = self.metrics_data['episode_length'].dropna()
            if not episode_lengths.empty:
                window_size = min(50, len(episode_lengths) // 10)
                if window_size > 1:
                    smoothed = episode_lengths.rolling(window=window_size, center=True).mean()
                    axes[0, 1].plot(smoothed.index, smoothed.values, alpha=0.8, linewidth=2, color='orange')
                else:
                    axes[0, 1].plot(episode_lengths.index, episode_lengths.values, alpha=0.8, linewidth=2, color='orange')
                axes[0, 1].set_title('Episode Lengths')
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Length')
                axes[0, 1].grid(True, alpha=0.3)
                
        # Success rate
        if 'episode_success' in self.metrics_data.columns:
            successes = self.metrics_data['episode_success'].dropna()
            if not successes.empty:
                # Calculate rolling success rate
                window_size = min(100, len(successes) // 5)
                if window_size > 1:
                    success_rate = successes.rolling(window=window_size, center=True).mean()
                    axes[0, 2].plot(success_rate.index, success_rate.values, alpha=0.8, linewidth=2, color='green')
                else:
                    axes[0, 2].plot(successes.index, successes.values, alpha=0.8, linewidth=2, color='green')
                axes[0, 2].set_title('Success Rate')
                axes[0, 2].set_xlabel('Episode')
                axes[0, 2].set_ylabel('Success Rate')
                axes[0, 2].set_ylim(0, 1)
                axes[0, 2].grid(True, alpha=0.3)
                
        # Training losses (if available)
        loss_columns = [col for col in self.metrics_data.columns if 'loss' in col.lower()]
        if loss_columns:
            for i, col in enumerate(loss_columns[:3]):  # Plot up to 3 loss curves
                row = 1
                col_idx = i
                if col_idx >= 3:
                    break
                    
                losses = self.metrics_data[col].dropna()
                if not losses.empty:
                    axes[row, col_idx].plot(losses.index, losses.values, alpha=0.8, linewidth=2)
                    axes[row, col_idx].set_title(f'{col.replace("_", " ").title()}')
                    axes[row, col_idx].set_xlabel('Training Step')
                    axes[row, col_idx].set_ylabel('Loss')
                    axes[row, col_idx].grid(True, alpha=0.3)
                    
        # Hide unused subplots
        for i in range(len(loss_columns), 3):
            axes[1, i].set_visible(False)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")
        else:
            plt.show()
            
    def plot_reward_distribution(
        self, 
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """
        Plot distribution of episode rewards.
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size
        """
        if self.metrics_data.empty or 'episode_reward' not in self.metrics_data.columns:
            logger.warning("No reward data available for plotting")
            return
            
        rewards = self.metrics_data['episode_reward'].dropna()
        if rewards.empty:
            logger.warning("No reward data available for plotting")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Reward Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Histogram
        axes[0, 0].hist(rewards, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Reward Distribution')
        axes[0, 0].set_xlabel('Reward')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot
        axes[0, 1].boxplot(rewards, vert=True)
        axes[0, 1].set_title('Reward Box Plot')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cumulative distribution
        sorted_rewards = np.sort(rewards)
        cumulative = np.arange(1, len(sorted_rewards) + 1) / len(sorted_rewards)
        axes[1, 0].plot(sorted_rewards, cumulative, linewidth=2)
        axes[1, 0].set_title('Cumulative Distribution')
        axes[1, 0].set_xlabel('Reward')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Reward over time (smoothed)
        window_size = min(100, len(rewards) // 10)
        if window_size > 1:
            smoothed = rewards.rolling(window=window_size, center=True).mean()
            axes[1, 1].plot(smoothed.index, smoothed.values, alpha=0.8, linewidth=2)
        else:
            axes[1, 1].plot(rewards.index, rewards.values, alpha=0.8, linewidth=2)
        axes[1, 1].set_title('Reward Over Time (Smoothed)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Reward distribution plot saved to {save_path}")
        else:
            plt.show()
            
    def plot_sequence_length_distribution(
        self, 
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """
        Plot distribution of sequence lengths.
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size
        """
        if self.metrics_data.empty or 'episode_length' not in self.metrics_data.columns:
            logger.warning("No sequence length data available for plotting")
            return
            
        lengths = self.metrics_data['episode_length'].dropna()
        if lengths.empty:
            logger.warning("No sequence length data available for plotting")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Sequence Length Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Histogram
        axes[0, 0].hist(lengths, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
        axes[0, 0].set_title('Length Distribution')
        axes[0, 0].set_xlabel('Sequence Length')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot
        axes[0, 1].boxplot(lengths, vert=True)
        axes[0, 1].set_title('Length Box Plot')
        axes[0, 1].set_ylabel('Sequence Length')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Length over time
        window_size = min(100, len(lengths) // 10)
        if window_size > 1:
            smoothed = lengths.rolling(window=window_size, center=True).mean()
            axes[1, 0].plot(smoothed.index, smoothed.values, alpha=0.8, linewidth=2, color='orange')
        else:
            axes[1, 0].plot(lengths.index, lengths.values, alpha=0.8, linewidth=2, color='orange')
        axes[1, 0].set_title('Length Over Time (Smoothed)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Sequence Length')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Length vs Reward scatter
        if 'episode_reward' in self.metrics_data.columns:
            rewards = self.metrics_data['episode_reward'].dropna()
            if len(rewards) == len(lengths):
                axes[1, 1].scatter(lengths, rewards, alpha=0.6, s=20)
                axes[1, 1].set_title('Length vs Reward')
                axes[1, 1].set_xlabel('Sequence Length')
                axes[1, 1].set_ylabel('Reward')
                axes[1, 1].grid(True, alpha=0.3)
                
                # Add correlation coefficient
                corr = np.corrcoef(lengths, rewards)[0, 1]
                axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                              transform=axes[1, 1].transAxes, fontsize=10,
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Sequence length distribution plot saved to {save_path}")
        else:
            plt.show()
            
    def plot_learning_curves_comparison(
        self, 
        other_log_dirs: List[str],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """
        Compare learning curves across multiple experiments.
        
        Args:
            other_log_dirs: List of other log directories to compare
            save_path: Path to save the plot
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Learning Curves Comparison', fontsize=16, fontweight='bold')
        
        # Collect data from all experiments
        experiments = [self.metrics_data]
        experiment_names = ['Current']
        
        for log_dir in other_log_dirs:
            try:
                metrics_file = Path(log_dir) / "metrics.json"
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    experiments.append(pd.DataFrame(metrics))
                    experiment_names.append(Path(log_dir).name)
            except Exception as e:
                logger.warning(f"Failed to load metrics from {log_dir}: {e}")
                
        # Plot comparison curves
        colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
        
        for i, (data, name, color) in enumerate(zip(experiments, experiment_names, colors)):
            if data.empty:
                continue
                
            # Episode rewards
            if 'episode_reward' in data.columns:
                rewards = data['episode_reward'].dropna()
                if not rewards.empty:
                    window_size = min(50, len(rewards) // 10)
                    if window_size > 1:
                        smoothed = rewards.rolling(window=window_size, center=True).mean()
                        axes[0, 0].plot(smoothed.index, smoothed.values, 
                                       label=name, color=color, alpha=0.8, linewidth=2)
                    else:
                        axes[0, 0].plot(rewards.index, rewards.values, 
                                       label=name, color=color, alpha=0.8, linewidth=2)
                        
            # Success rate
            if 'episode_success' in data.columns:
                successes = data['episode_success'].dropna()
                if not successes.empty:
                    window_size = min(100, len(successes) // 5)
                    if window_size > 1:
                        success_rate = successes.rolling(window=window_size, center=True).mean()
                        axes[0, 1].plot(success_rate.index, success_rate.values, 
                                      label=name, color=color, alpha=0.8, linewidth=2)
                    else:
                        axes[0, 1].plot(successes.index, successes.values, 
                                      label=name, color=color, alpha=0.8, linewidth=2)
                        
        # Configure subplots
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Success Rate')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Hide unused subplots
        axes[1, 0].set_visible(False)
        axes[1, 1].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Learning curves comparison saved to {save_path}")
        else:
            plt.show()
            
    def generate_summary_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a text summary report of training results.
        
        Args:
            save_path: Path to save the report
            
        Returns:
            Summary report text
        """
        if self.metrics_data.empty:
            return "No training data available for summary."
            
        report = []
        report.append("=" * 60)
        report.append("TRAINING SUMMARY REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Basic statistics
        if 'episode_reward' in self.metrics_data.columns:
            rewards = self.metrics_data['episode_reward'].dropna()
            if not rewards.empty:
                report.append("REWARD STATISTICS:")
                report.append(f"  Total episodes: {len(rewards)}")
                report.append(f"  Average reward: {rewards.mean():.3f}")
                report.append(f"  Median reward: {rewards.median():.3f}")
                report.append(f"  Min reward: {rewards.min():.3f}")
                report.append(f"  Max reward: {rewards.max():.3f}")
                report.append(f"  Std deviation: {rewards.std():.3f}")
                report.append("")
                
        if 'episode_length' in self.metrics_data.columns:
            lengths = self.metrics_data['episode_length'].dropna()
            if not lengths.empty:
                report.append("SEQUENCE LENGTH STATISTICS:")
                report.append(f"  Average length: {lengths.mean():.1f}")
                report.append(f"  Median length: {lengths.median():.1f}")
                report.append(f"  Min length: {lengths.min():.1f}")
                report.append(f"  Max length: {lengths.max():.1f}")
                report.append(f"  Std deviation: {lengths.std():.1f}")
                report.append("")
                
        if 'episode_success' in self.metrics_data.columns:
            successes = self.metrics_data['episode_success'].dropna()
            if not successes.empty:
                success_rate = successes.mean()
                report.append("SUCCESS STATISTICS:")
                report.append(f"  Success rate: {success_rate:.3f}")
                report.append(f"  Successful episodes: {successes.sum()}")
                report.append(f"  Failed episodes: {len(successes) - successes.sum()}")
                report.append("")
                
        # Learning progress
        if 'episode_reward' in self.metrics_data.columns and len(rewards) > 100:
            early_rewards = rewards[:len(rewards)//3]
            late_rewards = rewards[-len(rewards)//3:]
            
            report.append("LEARNING PROGRESS:")
            report.append(f"  Early episodes (first 1/3): {early_rewards.mean():.3f}")
            report.append(f"  Late episodes (last 1/3): {late_rewards.mean():.3f}")
            report.append(f"  Improvement: {late_rewards.mean() - early_rewards.mean():.3f}")
            report.append("")
            
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Summary report saved to {save_path}")
            
        return report_text
