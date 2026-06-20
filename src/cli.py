"""
Command-line interface for RL text generation project.

This module provides a comprehensive CLI for training, evaluating,
and visualizing RL agents for text generation tasks.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import json
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import load_config, Config, get_quick_test_config, get_production_config
from src.envs.text_generation_env import TextGenerationEnv, AdvancedTextGenEnv, TextGenConfig
from src.agents.text_ppo_agent import TextPPOAgent
from src.utils.logging_utils import setup_logging
from src.utils.visualization import TrainingVisualizer


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="RL Text Generation Training and Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train an RL agent')
    train_parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file'
    )
    train_parser.add_argument(
        '--algorithm', '-a',
        choices=['ppo', 'sac', 'td3', 'dqn'],
        help='RL algorithm to use'
    )
    train_parser.add_argument(
        '--timesteps', '-t',
        type=int,
        help='Total training timesteps'
    )
    train_parser.add_argument(
        '--target-word',
        type=str,
        help='Target word for text generation'
    )
    train_parser.add_argument(
        '--vocab-size',
        type=int,
        help='Vocabulary size'
    )
    train_parser.add_argument(
        '--max-length',
        type=int,
        help='Maximum sequence length'
    )
    train_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='outputs',
        help='Output directory for models and logs'
    )
    train_parser.add_argument(
        '--seed',
        type=int,
        help='Random seed'
    )
    train_parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for training'
    )
    train_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser('eval', help='Evaluate a trained agent')
    eval_parser.add_argument(
        '--model-path', '-m',
        type=str,
        required=True,
        help='Path to trained model'
    )
    eval_parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file'
    )
    eval_parser.add_argument(
        '--episodes', '-e',
        type=int,
        default=100,
        help='Number of episodes to evaluate'
    )
    eval_parser.add_argument(
        '--render',
        action='store_true',
        help='Render episodes during evaluation'
    )
    eval_parser.add_argument(
        '--output-file',
        type=str,
        help='File to save evaluation results'
    )
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize training results')
    viz_parser.add_argument(
        '--log-dir',
        type=str,
        required=True,
        help='Directory containing training logs'
    )
    viz_parser.add_argument(
        '--output-dir',
        type=str,
        default='plots',
        help='Directory to save plots'
    )
    viz_parser.add_argument(
        '--format',
        choices=['png', 'pdf', 'svg'],
        default='png',
        help='Plot format'
    )
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run interactive demo')
    demo_parser.add_argument(
        '--model-path', '-m',
        type=str,
        help='Path to trained model'
    )
    demo_parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file'
    )
    demo_parser.add_argument(
        '--episodes', '-e',
        type=int,
        default=10,
        help='Number of demo episodes'
    )
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument(
        '--create-default',
        action='store_true',
        help='Create default configuration file'
    )
    config_parser.add_argument(
        '--validate',
        type=str,
        help='Validate configuration file'
    )
    config_parser.add_argument(
        '--preset',
        choices=['quick-test', 'production', 'research'],
        help='Use predefined configuration preset'
    )
    
    return parser


def train_command(args: argparse.Namespace) -> None:
    """Execute training command."""
    # Load configuration
    if args.config:
        config = load_config(args.config)
    elif args.preset:
        if args.preset == 'quick-test':
            config = get_quick_test_config()
        elif args.preset == 'production':
            config = get_production_config()
        else:
            config = get_production_config()  # Default to production
    else:
        config = load_config()
    
    # Override config with command line arguments
    if args.algorithm:
        config.agent.algorithm = args.algorithm
    if args.timesteps:
        config.training.total_timesteps = args.timesteps
    if args.target_word:
        config.environment.target_word = args.target_word
    if args.vocab_size:
        config.environment.vocab_size = args.vocab_size
    if args.max_length:
        config.environment.max_sequence_length = args.max_length
    if args.seed:
        config.seed = args.seed
    if args.device:
        config.device = args.device
    if args.verbose:
        config.verbose = 2
    
    # Setup logging
    log_dir = os.path.join(args.output_dir, 'logs')
    setup_logging(config.logging, log_dir)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting training...")
    logger.info(f"Configuration: {config}")
    
    # Create environment
    env_config = TextGenConfig(
        vocab_size=config.environment.vocab_size,
        max_sequence_length=config.environment.max_sequence_length,
        target_word=config.environment.target_word,
        reward_type=config.environment.reward_type,
        length_penalty=config.environment.length_penalty,
        early_stop_reward=config.environment.early_stop_reward
    )
    
    if config.environment.use_advanced_env:
        env = AdvancedTextGenEnv(env_config, render_mode="human" if args.verbose else None)
    else:
        env = TextGenerationEnv(env_config, render_mode="human" if args.verbose else None)
    
    # Create agent
    if config.agent.algorithm == 'ppo':
        agent = TextPPOAgent(
            env=env,
            learning_rate=config.agent.learning_rate,
            n_steps=config.agent.n_steps,
            batch_size=config.agent.batch_size,
            n_epochs=config.agent.n_epochs,
            gamma=config.agent.gamma,
            gae_lambda=config.agent.gae_lambda,
            clip_range=config.agent.clip_range,
            ent_coef=config.agent.ent_coef,
            vf_coef=config.agent.vf_coef,
            max_grad_norm=config.agent.max_grad_norm,
            device=config.device
        )
    else:
        raise ValueError(f"Algorithm {config.agent.algorithm} not implemented yet")
    
    # Train agent
    start_time = time.time()
    metrics = agent.train(config.training.total_timesteps)
    training_time = time.time() - start_time
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Save model and metrics
    model_dir = os.path.join(args.output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save metrics
    metrics_file = os.path.join(args.output_dir, 'training_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Training metrics saved to {metrics_file}")
    logger.info("Training completed successfully!")


def eval_command(args: argparse.Namespace) -> None:
    """Execute evaluation command."""
    logger = logging.getLogger(__name__)
    logger.info("Starting evaluation...")
    
    # Load configuration
    config = load_config(args.config) if args.config else load_config()
    
    # Create environment
    env_config = TextGenConfig(
        vocab_size=config.environment.vocab_size,
        max_sequence_length=config.environment.max_sequence_length,
        target_word=config.environment.target_word,
        reward_type=config.environment.reward_type
    )
    
    if config.environment.use_advanced_env:
        env = AdvancedTextGenEnv(env_config, render_mode="human" if args.render else None)
    else:
        env = TextGenerationEnv(env_config, render_mode="human" if args.render else None)
    
    # Load agent (implementation depends on algorithm)
    # For now, assume PPO
    agent = TextPPOAgent(env=env, device=config.device)
    
    # TODO: Load trained model weights
    logger.warning("Model loading not implemented yet")
    
    # Evaluate agent
    episode_rewards = []
    episode_lengths = []
    success_rate = 0
    
    for episode in range(args.episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Get action from agent
            with torch.no_grad():
                obs_tensor = {key: torch.tensor(value, device=agent.device).unsqueeze(0) 
                             for key, value in obs.items()}
                logits, _ = agent.policy_net(obs_tensor['sequence'])
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            if args.render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if episode_reward > 0:  # Success
            success_rate += 1
            
        if args.render:
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    success_rate /= args.episodes
    
    # Print results
    print(f"\nEvaluation Results:")
    print(f"Episodes: {args.episodes}")
    print(f"Average Reward: {sum(episode_rewards) / len(episode_rewards):.3f}")
    print(f"Average Length: {sum(episode_lengths) / len(episode_lengths):.1f}")
    print(f"Success Rate: {success_rate:.3f}")
    
    # Save results
    if args.output_file:
        results = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'success_rate': success_rate,
            'average_reward': sum(episode_rewards) / len(episode_rewards),
            'average_length': sum(episode_lengths) / len(episode_lengths)
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {args.output_file}")


def visualize_command(args: argparse.Namespace) -> None:
    """Execute visualization command."""
    logger = logging.getLogger(__name__)
    logger.info("Starting visualization...")
    
    # Create visualizer
    visualizer = TrainingVisualizer(args.log_dir)
    
    # Create plots
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot training curves
    visualizer.plot_training_curves(
        save_path=os.path.join(args.output_dir, f'training_curves.{args.format}')
    )
    
    # Plot reward distribution
    visualizer.plot_reward_distribution(
        save_path=os.path.join(args.output_dir, f'reward_distribution.{args.format}')
    )
    
    # Plot sequence length distribution
    visualizer.plot_sequence_length_distribution(
        save_path=os.path.join(args.output_dir, f'sequence_length_distribution.{args.format}')
    )
    
    logger.info(f"Plots saved to {args.output_dir}")


def demo_command(args: argparse.Namespace) -> None:
    """Execute demo command."""
    logger = logging.getLogger(__name__)
    logger.info("Starting interactive demo...")
    
    # Load configuration
    config = load_config(args.config) if args.config else load_config()
    
    # Create environment
    env_config = TextGenConfig(
        vocab_size=config.environment.vocab_size,
        max_sequence_length=config.environment.max_sequence_length,
        target_word=config.environment.target_word,
        reward_type=config.environment.reward_type
    )
    
    env = TextGenerationEnv(env_config, render_mode="human")
    
    print(f"\n🎯 Target word: {config.environment.target_word}")
    print(f"📚 Vocabulary size: {config.environment.vocab_size}")
    print(f"📏 Max sequence length: {config.environment.max_sequence_length}")
    print(f"🎮 Running {args.episodes} demo episodes...\n")
    
    # Run demo episodes
    for episode in range(args.episodes):
        print(f"\n--- Episode {episode + 1} ---")
        obs, _ = env.reset()
        
        # Random policy for demo (replace with trained agent)
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            print(f"Action: {info['current_word']} | Reward: {reward:.2f}")
        
        print(f"Final sequence: '{info['sequence_text']}'")
        print(f"Success: {'✅' if reward > 0 else '❌'}")
    
    print(f"\n🎉 Demo completed!")


def config_command(args: argparse.Namespace) -> None:
    """Execute configuration command."""
    if args.create_default:
        from src.utils.config import ConfigManager
        manager = ConfigManager()
        manager.create_default_config()
        print("Default configuration created at configs/default.yaml")
        
    elif args.validate:
        from src.utils.config import ConfigManager
        manager = ConfigManager(args.validate)
        warnings = manager.validate_config()
        
        if warnings:
            print("Configuration warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        else:
            print("Configuration is valid!")
            
    elif args.preset:
        config = None
        if args.preset == 'quick-test':
            config = get_quick_test_config()
        elif args.preset == 'production':
            config = get_production_config()
        elif args.preset == 'research':
            config = get_production_config()  # Use production as base
            
        if config:
            from src.utils.config import ConfigManager
            manager = ConfigManager()
            manager.config = config
            
            preset_path = f"configs/{args.preset}.yaml"
            manager.save_config(preset_path)
            print(f"Configuration preset '{args.preset}' saved to {preset_path}")
    else:
        print("Use --help to see available configuration options")


def main():
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'train':
            train_command(args)
        elif args.command == 'eval':
            eval_command(args)
        elif args.command == 'visualize':
            visualize_command(args)
        elif args.command == 'demo':
            demo_command(args)
        elif args.command == 'config':
            config_command(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose if hasattr(args, 'verbose') else False:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
