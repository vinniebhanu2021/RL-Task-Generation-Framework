# RL Task Generation

A comprehensive reinforcement learning framework for text generation tasks, featuring state-of-the-art algorithms, proper logging, visualization, and configuration management.

## Features

- **Modern RL Algorithms**: PPO, SAC, TD3, Rainbow DQN implementations
- **Gymnasium Integration**: Full compatibility with modern RL environments
- **Advanced Text Generation**: Custom environments with sophisticated reward functions
- **Comprehensive Logging**: TensorBoard, Weights & Biases, and file logging support
- **Rich Visualizations**: Training curves, reward distributions, and learning analysis
- **Configuration Management**: YAML-based configuration with validation
- **CLI Interface**: Easy-to-use command-line tools for training and evaluation
- **Type Safety**: Full type hints throughout the codebase
- **Documentation**: Comprehensive docstrings and examples

## 📁 Project Structure

```
0275_RL_for_task_generation/
├── src/
│   ├── agents/           # RL algorithm implementations
│   │   └── text_ppo_agent.py
│   ├── envs/            # Custom environments
│   │   └── text_generation_env.py
│   ├── utils/           # Utilities and helpers
│   │   ├── config.py
│   │   ├── logging_utils.py
│   │   └── visualization.py
│   └── cli.py           # Command-line interface
├── configs/             # Configuration files
├── notebooks/           # Jupyter notebooks for analysis
├── logs/               # Training logs and metrics
├── plots/              # Generated visualizations
├── outputs/            # Model outputs and results
├── tests/              # Unit tests
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore rules
├── train.py           # Main training script
└── README.md          # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for training)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://https://github.com/vinniebhanu2021/RL-Task-Generation-Framework.git
   cd RL-Text-Generation-Project
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   python -c "import gymnasium; print('Gymnasium installed successfully')"
   ```

## Quick Start

### 1. Basic Training

Train a PPO agent to generate text ending with "you":

```bash
python train.py --algorithm ppo --timesteps 50000 --target-word you --vocab-size 64
```

### 2. Using Configuration Files

Create a custom configuration:

```bash
python src/cli.py config --create-default
```

Edit `configs/default.yaml` and train:

```bash
python train.py --config configs/default.yaml
```

### 3. Interactive Demo

Run an interactive demo:

```bash
python src/cli.py demo --episodes 10 --target-word love
```

## Usage Examples

### Training with Different Algorithms

**PPO (Recommended for text generation)**:
```bash
python train.py --algorithm ppo --timesteps 100000 --target-word "love"
```

**SAC (For continuous action spaces)**:
```bash
python train.py --algorithm sac --timesteps 100000 --target-word "peace"
```

### Advanced Configuration

**Custom vocabulary and sequence length**:
```bash
python train.py \
  --vocab-size 1000 \
  --max-length 50 \
  --target-word "beautiful" \
  --timesteps 200000
```

**Using advanced environment with multi-objective rewards**:
```bash
python train.py \
  --config configs/advanced.yaml \
  --timesteps 500000
```

### Evaluation

Evaluate a trained model:

```bash
python src/cli.py eval --model-path outputs/models/best_model.pth --episodes 100
```

### Visualization

Generate training visualizations:

```bash
python src/cli.py visualize --log-dir logs/experiment_20231201_143022 --output-dir plots
```

## Configuration

### Environment Configuration

```yaml
environment:
  vocab_size: 64                    # Size of vocabulary
  max_sequence_length: 20           # Maximum sequence length
  target_word: "you"               # Target word to generate
  reward_type: "binary"            # Reward type: binary, length_penalty, semantic
  use_advanced_env: false          # Use advanced multi-objective environment
  reward_weights:                  # Weights for different reward components
    target_completion: 1.0
    length_efficiency: 0.1
    diversity: 0.05
    grammar: 0.2
```

### Agent Configuration

```yaml
agent:
  algorithm: "ppo"                  # RL algorithm: ppo, sac, td3, dqn
  learning_rate: 0.0003           # Learning rate
  batch_size: 64                   # Batch size
  n_steps: 2048                    # Steps per update (PPO)
  n_epochs: 10                     # Training epochs per update
  gamma: 0.99                      # Discount factor
  clip_range: 0.2                  # PPO clipping range
  ent_coef: 0.01                   # Entropy coefficient
  vf_coef: 0.5                     # Value function coefficient
  hidden_size: 128                 # Network hidden size
  num_layers: 2                    # Number of RNN layers
  use_attention: false             # Use attention mechanism
```

### Training Configuration

```yaml
training:
  total_timesteps: 100000          # Total training timesteps
  eval_frequency: 10000           # Evaluation frequency
  eval_episodes: 10               # Episodes for evaluation
  save_frequency: 50000           # Model save frequency
  early_stopping: false           # Enable early stopping
  patience: 10                     # Early stopping patience
```

### Logging Configuration

```yaml
logging:
  log_level: "INFO"               # Logging level
  use_tensorboard: true           # Enable TensorBoard logging
  use_wandb: false                # Enable Weights & Biases logging
  wandb_project: "rl-text-gen"   # W&B project name
  save_plots: true                # Save training plots
```

## Advanced Usage

### Custom Environments

Create custom text generation environments:

```python
from src.envs.task_generation_env import TextGenConfig, AdvancedTextGenEnv

# Custom configuration
config = TextGenConfig(
    vocab_size=1000,
    max_sequence_length=30,
    target_word="beautiful",
    reward_type="semantic"
)

# Create environment
env = AdvancedTextGenEnv(config, reward_weights={
    "target_completion": 1.0,
    "length_efficiency": 0.2,
    "diversity": 0.1,
    "grammar": 0.3
})
```

### Custom Agents

Implement custom RL agents:

```python
from src.agents.text_ppo_agent import TextPPOAgent

# Custom PPO agent
agent = TextPPOAgent(
    env=env,
    learning_rate=1e-4,
    n_steps=4096,
    batch_size=128,
    use_attention=True
)
```

### Custom Reward Functions

Implement custom reward functions:

```python
class CustomTextGenEnv(AdvancedTextGenEnv):
    def _calculate_custom_reward(self, terminated, truncated):
        # Your custom reward logic here
        base_reward = super()._calculate_reward(terminated, truncated)
        
        # Add custom components
        custom_component = self._calculate_semantic_similarity()
        
        return base_reward + custom_component
```

## Monitoring and Visualization

### TensorBoard

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir logs/
```

### Weights & Biases

Enable W&B logging in configuration:

```yaml
logging:
  use_wandb: true
  wandb_project: "my-rl-experiment"
  wandb_entity: "my-username"
```

### Custom Visualizations

Create custom plots:

```python
from src.utils.visualization import TrainingVisualizer

visualizer = TrainingVisualizer("logs/experiment_20231201_143022")

# Generate custom plots
visualizer.plot_training_curves(save_path="custom_curves.png")
visualizer.plot_reward_distribution(save_path="reward_dist.png")
```

## Experimentation

### Hyperparameter Tuning

Use configuration presets for different scenarios:

```bash
# Quick test configuration
python train.py --config configs/quick_test.yaml

# Production configuration
python train.py --config configs/production.yaml

# Research configuration
python train.py --config configs/research.yaml
```

### A/B Testing

Compare different algorithms:

```bash
# Train PPO
python train.py --algorithm ppo --timesteps 100000 --output-dir outputs/ppo

# Train SAC
python train.py --algorithm sac --timesteps 100000 --output-dir outputs/sac

# Compare results
python src/cli.py visualize --log-dir outputs/ppo/logs --output-dir plots/ppo
python src/cli.py visualize --log-dir outputs/sac/logs --output-dir plots/sac
```

## Troubleshooting

### Common Issues

**CUDA out of memory**:
```bash
# Reduce batch size
python train.py --batch-size 32

# Use CPU
python train.py --device cpu
```

**Slow training**:
```bash
# Reduce vocabulary size
python train.py --vocab-size 32

# Reduce sequence length
python train.py --max-length 10
```

**Poor performance**:
```bash
# Increase training time
python train.py --timesteps 500000

# Use advanced environment
python train.py --use-advanced-env
```

### Debug Mode

Enable verbose logging:

```bash
python train.py --verbose --log-level DEBUG
```

## API Reference

### Environment API

```python
class TextGenerationEnv(gym.Env):
    def __init__(self, config: TextGenConfig, vocab: List[str] = None, render_mode: str = None)
    def reset(self, seed: int = None, options: Dict = None) -> Tuple[Dict, Dict]
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]
    def render(self) -> Optional[Union[str, np.ndarray]]
```

### Agent API

```python
class TextPPOAgent:
    def __init__(self, env: gym.Env, learning_rate: float = 3e-4, ...)
    def train(self, total_timesteps: int) -> Dict[str, List[float]]
    def collect_rollouts(self) -> Dict[str, torch.Tensor]
    def update(self, rollout_data: Dict[str, torch.Tensor]) -> Dict[str, float]
```

### Configuration API

```python
class Config:
    environment: EnvironmentConfig
    agent: AgentConfig
    training: TrainingConfig
    logging: LoggingConfig

def load_config(config_path: str = None) -> Config
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Run tests: `python -m pytest tests/`
6. Commit changes: `git commit -am 'Add feature'`
7. Push to branch: `git push origin feature-name`
8. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Run tests
python -m pytest tests/

# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```




 
 

