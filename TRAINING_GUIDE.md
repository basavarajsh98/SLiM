# SLiM Dynamic Training System

This guide explains how to use the refactored training system that supports multiple experiment types dynamically.

## Overview

The training system has been refactored to be more flexible and configurable. It now supports:
- **Default experiments** (multi-state steering with LoRA)
- **Detoxification experiments** (single-state steering)
- **Emotion steering experiments** (dual-state steering)
- **Custom experiment configurations**

## Key Features

1. **Dynamic Experiment Types**: Switch between different experiment types without code changes
2. **Configurable Parameters**: All training parameters can be customized via config files
3. **Automatic Evaluation**: Built-in evaluation with different state configurations
4. **Flexible Model Setup**: Automatic model configuration based on experiment type
5. **Progress Monitoring**: Enhanced logging and evaluation during training

## Quick Start

### 1. Basic Usage

```bash
# Run with default configuration
python src/train.py

# Run specific experiment type
python src/run_experiment.py
```

### 2. Configuration

Edit `config/config.yaml` to set your experiment type and parameters:

```yaml
# Set experiment type
experiment_type: detoxification  # Options: default, detoxification, emotion_steering, multi_state_steering, language_steering, topic_steering

# General parameters
epochs: 10
batch_size: 4
learning_rate: 2e-5

# Experiment-specific parameters
detoxification:
  epochs: 50
  num_states: 1
  save_model_path: "results/SLiM_detoxification"

multi_state_steering:
  epochs: 50
  num_states: 3
  save_model_path: "results/SLiM_multi_state_steering"

language_steering:
  epochs: 50
  num_states: 2
  save_model_path: "results/SLiM_language_steering"

topic_steering:
  epochs: 50
  num_states: 4
  save_model_path: "results/SLiM_topic_steering"
```

## Experiment Types

### Default (Multi-State Steering)
- **Purpose**: Multi-dimensional state control (language, sentiment, topic)
- **Model**: GPT-2 with LoRA fine-tuning
- **States**: 5-dimensional state vectors
- **Use Case**: Complex multi-attribute text generation

### Detoxification
- **Purpose**: Control toxicity levels in generated text
- **Model**: Custom SLiMedNet
- **States**: Single float value (0.0 to 1.0)
- **Use Case**: Safe text generation with controlled toxicity

### Emotion Steering
- **Purpose**: Control emotional states in generated text
- **Model**: Custom SLiMedNet
- **States**: 2-dimensional vectors (emotion, language)
- **Use Case**: Emotion-aware text generation

### Multi-State Steering
- **Purpose**: Control multiple aspects (topic, language, rating) simultaneously
- **Model**: Custom SLiMedNet
- **States**: 3-dimensional vectors (language, topic, rating)
- **Use Case**: Multi-attribute content generation with topic and language control

### Language Steering
- **Purpose**: Control language generation (English vs German)
- **Model**: Custom SLiMedNet
- **States**: Binary language states (English/German)
- **Use Case**: Bilingual text generation with language control

### Topic Steering
- **Purpose**: Control topic-specific content generation
- **Model**: Custom SLiMedNet
- **States**: Topic-specific state vectors
- **Use Case**: Topic-aware text generation

## Configuration Parameters

### General Parameters
- `experiment_type`: Type of experiment to run
- `epochs`: Number of training epochs
- `batch_size`: Training batch size
- `learning_rate`: Learning rate for optimizer
- `accumulation_steps`: Gradient accumulation steps
- `max_steps`: Maximum training steps (optional)
- `eval_frequency`: How often to run evaluation
- `prompt_text`: Text prompt for evaluation generation

### Experiment-Specific Parameters
Each experiment type can have its own configuration section in the YAML file.

## Running Experiments

### Method 1: Direct Execution
```bash
python src/train.py
```

### Method 2: Using run_experiment.py
```python
from src.run_experiment import run_detoxification_experiment
from src.run_experiment import run_emotion_steering_experiment
from src.run_experiment import run_multi_state_steering_experiment
from src.run_experiment import run_language_steering_experiment
from src.run_experiment import run_topic_steering_experiment

# Run detoxification experiment
run_detoxification_experiment()

# Run emotion steering experiment
run_emotion_steering_experiment()

# Run multi-state steering experiment
run_multi_state_steering_experiment()

# Run language steering experiment
run_language_steering_experiment()

# Run topic steering experiment
run_topic_steering_experiment()
```

### Method 3: Custom Configuration
```python
from src.train import main

custom_config = {
    "experiment_type": "detoxification",
    "epochs": 20,
    "max_steps": 500,
    "save_model_path": "my_custom_path"
}

main(experiment_type="detoxification", custom_config=custom_config)
```

## Model Architecture

The system automatically selects the appropriate model architecture based on the experiment type:

- **Default**: GPT-2 + LoRA + SLiMedNet wrapper
- **Detoxification**: Custom SLiMedNet (no base model)
- **Emotion Steering**: Custom SLiMedNet (no base model)
- **Multi-State Steering**: Custom SLiMedNet (no base model)
- **Language Steering**: Custom SLiMedNet (no base model)
- **Topic Steering**: Custom SLiMedNet (no base model)

## Evaluation

During training, the system automatically evaluates the model by:
1. Generating text with baseline (no state)
2. Generating text with different state configurations
3. Displaying results for comparison

Evaluation frequency and state samples are configurable.

## Output

Training outputs include:
- Progress logs with loss and perplexity
- Regular evaluation results
- Model checkpoints saved at configurable intervals
- Final model and training statistics

## Customization

### Adding New Experiment Types
1. Add new experiment type to `config.yaml`
2. Update the `setup_training_components` function in `train.py`
3. Add experiment-specific logic to the training pipeline

### Custom State Configurations
Modify the `run_evaluation` function to include your desired state samples for evaluation.

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce batch size or use gradient accumulation
3. **Configuration Errors**: Check YAML syntax and parameter names

### Debug Mode
Set `verbose=True` in the training function to get detailed logging.

## Examples

See `src/run_experiment.py` for complete examples of running different experiment types.

## Support

For issues or questions, check the configuration files and ensure all dependencies are properly installed.
