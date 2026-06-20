"""
Main training script for RL text generation project.

This script provides a comprehensive training pipeline with modern RL algorithms,
proper logging, visualization, and configuration management.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.config import load_config, Config
from src.envs.text_generation_env import TextGenerationEnv, AdvancedTextGenEnv, TextGenConfig
from src.agents.text_ppo_agent import TextPPOAgent
from src.utils.logging_utils import setup_logging, create_experiment_logger
from src.utils.visualization import TrainingVisualizer


def create_env(config: Config):
    """Create environment based on configuration."""
    env_config = TextGenConfig(
        vocab_size=config.environment.vocab_size,
        max_sequence_length=config.environment.max_sequence_length,
        target_word=config.environment.target_word,
        target_word_index=config.environment.target_word_index,
        reward_type=config.environment.reward_type,
        length_penalty=config.environment.length_penalty,
        early_stop_reward=config.environment.early_stop_reward,
        invalid_action_penalty=config.environment.invalid_action_penalty
    )
    
    if config.environment.use_advanced_env:
        env = AdvancedTextGenEnv(
            env_config, 
            reward_weights=config.environment.reward_weights,
            render_mode=None
        )
    else:
        env = TextGenerationEnv(env_config, render_mode=None)
        
    return env


def create_agent(env, config: Config):
    """Create agent based on configuration."""
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
        
    return agent


def train_agent(agent, env, config: Config, logger: logging.Logger):
    """Train the agent with proper logging and evaluation."""
    # Create training logger
    training_logger = create_experiment_logger(
        experiment_name=f"{config.agent.algorithm}_text_gen",
        config=config.logging,
        base_log_dir="logs"
    )
    
    logger.info("Starting training...")
    logger.info(f"Algorithm: {config.agent.algorithm}")
    logger.info(f"Total timesteps: {config.training.total_timesteps}")
    logger.info(f"Target word: {config.environment.target_word}")
    
    try:
        # Train agent
        metrics = agent.train(config.training.total_timesteps)
        
        # Log final metrics
        training_logger.close()
        
        logger.info("Training completed successfully!")
        
        # Generate visualizations
        if config.logging.save_plots:
            log_dir = str(training_logger.metrics_logger.log_dir)
            visualizer = TrainingVisualizer(log_dir)
            
            plots_dir = Path(config.logging.plot_dir)
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Create plots
            visualizer.plot_training_curves(
                save_path=str(plots_dir / "training_curves.png")
            )
            visualizer.plot_reward_distribution(
                save_path=str(plots_dir / "reward_distribution.png")
            )
            visualizer.plot_sequence_length_distribution(
                save_path=str(plots_dir / "sequence_length_distribution.png")
            )
            
            # Generate summary report
            summary_report = visualizer.generate_summary_report(
                save_path=str(plots_dir / "summary_report.txt")
            )
            
            logger.info(f"Visualizations saved to {plots_dir}")
            logger.info("Summary Report:")
            logger.info(summary_report)
            
        return metrics
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        training_logger.close()
        return None
    except Exception as e:
        logger.error(f"Training failed: {e}")
        training_logger.close()
        raise


def evaluate_agent(agent, env, config: Config, num_episodes: int = 100):
    """Evaluate the trained agent."""
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating agent for {num_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    successes = []
    
    for episode in range(num_episodes):
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
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        successes.append(episode_reward > 0)
        
        if episode % 10 == 0:
            logger.info(f"Episode {episode}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    # Calculate statistics
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_length = sum(episode_lengths) / len(episode_lengths)
    success_rate = sum(successes) / len(successes)
    
    logger.info(f"Evaluation Results:")
    logger.info(f"  Average Reward: {avg_reward:.3f}")
    logger.info(f"  Average Length: {avg_length:.1f}")
    logger.info(f"  Success Rate: {success_rate:.3f}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'successes': successes,
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'success_rate': success_rate
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RL agent for text generation")
    parser.add_argument("--config", "-c", type=str, help="Path to configuration file")
    parser.add_argument("--algorithm", "-a", choices=['ppo', 'sac', 'td3', 'dqn'], help="RL algorithm")
    parser.add_argument("--timesteps", "-t", type=int, help="Total training timesteps")
    parser.add_argument("--target-word", type=str, help="Target word for generation")
    parser.add_argument("--vocab-size", type=int, help="Vocabulary size")
    parser.add_argument("--max-length", type=int, help="Maximum sequence length")
    parser.add_argument("--output-dir", "-o", type=str, default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--device", choices=['auto', 'cpu', 'cuda'], help="Device to use")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate, don't train")
    parser.add_argument("--model-path", type=str, help="Path to trained model for evaluation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config) if args.config else load_config()
    
    # Override with command line arguments
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
    logger = setup_logging(config.logging, log_dir)
    
    # Set random seed
    import torch
    import numpy as np
    import random
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    logger.info("=" * 60)
    logger.info("RL TEXT GENERATION TRAINING")
    logger.info("=" * 60)
    logger.info(f"Configuration: {config}")
    
    try:
        # Create environment
        env = create_env(config)
        logger.info(f"Environment created: {env.__class__.__name__}")
        
        # Create agent
        agent = create_agent(env, config)
        logger.info(f"Agent created: {agent.__class__.__name__}")
        
        if args.eval_only:
            # Evaluation only
            if args.model_path:
                # TODO: Load model weights
                logger.warning("Model loading not implemented yet")
            
            results = evaluate_agent(agent, env, config)
            
            # Save evaluation results
            import json
            eval_file = os.path.join(args.output_dir, 'evaluation_results.json')
            with open(eval_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Evaluation results saved to {eval_file}")
            
        else:
            # Training
            metrics = train_agent(agent, env, config, logger)
            
            if metrics is not None:
                # Save training metrics
                import json
                metrics_file = os.path.join(args.output_dir, 'training_metrics.json')
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                logger.info(f"Training metrics saved to {metrics_file}")
                
                # Evaluate trained agent
                logger.info("Evaluating trained agent...")
                eval_results = evaluate_agent(agent, env, config)
                
                # Save evaluation results
                eval_file = os.path.join(args.output_dir, 'evaluation_results.json')
                with open(eval_file, 'w') as f:
                    json.dump(eval_results, f, indent=2)
                logger.info(f"Evaluation results saved to {eval_file}")
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
