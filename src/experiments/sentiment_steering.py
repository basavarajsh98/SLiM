import os
import warnings
from collections import Counter
from random import shuffle

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from src.model import SLiMedNet
from src.inference import generate_text

warnings.filterwarnings("ignore")
from config.config import get_config

config = get_config()
# Default configuration for sentiment steering experiment
DEFAULT_CONFIG = {
    "num_states": 2,
    "save_model_path": "resources/checkpoints/sentiment_steering/sentiments",
    "prompt_text": "I think",
}


def get_sentiment_evaluation_samples():
    """
    Get specialized evaluation samples for sentiment steering experiments.

    Returns:
        list: List of sentiment states with descriptions for evaluation
    """
    return [
        (torch.FloatTensor([1, 0]), "Positive Sentiment"),
        (torch.FloatTensor([0, 1]), "Negative Sentiment"),
        (torch.FloatTensor([0.8, 0.2]), "Mostly Positive"),
        (torch.FloatTensor([0.2, 0.8]), "Mostly Negative"),
        (torch.FloatTensor([0.5, 0.5]), "Neutral Sentiment"),
        (torch.FloatTensor([0.9, 0.1]), "Very Positive"),
        (torch.FloatTensor([0.1, 0.9]), "Very Negative"),
    ]


def get_sentiment_state_samples():
    """
    Get state samples for sentiment steering experiments (for use in training).

    Returns:
        list: List of sentiment state tensors for evaluation during training
    """
    return [
        torch.FloatTensor([1, 0]),  # Positive
        torch.FloatTensor([0, 1]),  # Negative
    ]


def run_sentiment_steering_evaluation(
    model, tokenizer, device, prompt_text="I think", verbose=True
):
    """
    Run specialized evaluation for sentiment steering experiments.

    Args:
        model: The trained model
        tokenizer: The tokenizer
        device: Device to run evaluation on
        prompt_text: Text prompt for generation
        verbose: Whether to print results
    """
    if not verbose:
        return

    print("\n" + "=" * 60)
    print("SENTIMENT STEERING EVALUATION RESULTS")
    print("=" * 60)

    # Baseline generation (no state)
    baseline_text = generate_text(
        model, tokenizer, prompt_text, max_length=40, state_tensor=None
    )
    print(f"Baseline (no state): {baseline_text}")
    print("-" * 60)

    # Get specialized evaluation samples
    eval_samples = get_sentiment_evaluation_samples()

    # Generate text for each sentiment state
    for i, (state_tensor, description) in enumerate(eval_samples):
        state_tensor = state_tensor.unsqueeze(0).to(device)

        generated_text = generate_text(
            model, tokenizer, prompt_text, max_length=40, state_tensor=state_tensor
        )

        print(f"State {i + 1} ({description}): {generated_text}")

    print("=" * 60 + "\n")


def balance_dataset(texts, min_count=500):
    """
    Balance the dataset by downsampling to the minimum count across sentiment categories.

    Args:
        texts: List of (text, label) tuples
        min_count: Minimum count per sentiment category

    Returns:
        list: Balanced list of (text, label) tuples
    """
    # Count the number of samples in each sentiment category
    counts = Counter([tuple(label) for _, label in texts])
    print("min_count: ", min_count)
    balanced_texts = []
    sentiment_buckets = {sentiment: [] for sentiment in counts.keys()}

    # Organize samples by sentiment
    for text, label in texts:
        sentiment_buckets[tuple(label)].append((text, tuple(label)))

    # Downsample to the minimum count in each category
    for sentiment, samples in sentiment_buckets.items():
        shuffle(samples)  # Shuffle to avoid any ordering bias
        balanced_texts.extend(samples[:min_count])

    balanced_texts = [(text, list(map(int, label))) for text, label in balanced_texts]
    shuffle(balanced_texts)  # Shuffle again to mix sentiments in the final dataset
    return balanced_texts


def prepare_dataset(
    max_sequence_length=64, path="resources/datasets/sentiments.pth"
):
    """
    Prepare dataset for sentiment steering experiment.

    Args:
        max_sequence_length: Maximum sequence length for tokenization
        path: Path to save/load the processed dataset

    Returns:
        samples: List of (tokens, state) tuples
        tokenizer: The tokenizer used for processing
    """

    tokenizer = AutoTokenizer.from_pretrained(config.get('base_model'))
    tokenizer.pad_token = tokenizer.eos_token
    label_mapping = {0: [1, 0], 1: [0, 1]}

    if not os.path.exists(path):
        print("Loading dataset from HuggingFace...")
        dataset = load_dataset("contemmcm/sentiment140")  # ['en', 'de']
        texts = []

        # Prepare texts and labels with one-hot encoding
        for item in dataset["complete"]:
            texts.append((item["text"], label_mapping.get(item["label"])))

        # Balance the dataset
        print("Balancing dataset...")
        texts = balance_dataset(texts)
        print("Sample: ", texts[0])
        print("Total Size: ", len(texts))

        # Tokenize balanced texts
        print("Tokenizing started...")
        encoded_texts = tokenizer(
            [text for text, _ in texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_sequence_length + 1,
        )["input_ids"]
        print("Done.")

        # Create final samples
        samples = [(tokens, state) for tokens, (_, state) in zip(encoded_texts, texts)]
        torch.save(samples, path)
    else:
        print(f"Found tokenized dataset at {path}!\nImporting...")
        samples = torch.load(path)
        print("Done.")
        print("Total Size: ", len(samples))

    return samples, tokenizer


def get_sentiment_steering_config(custom_config=None):
    """
    Get configuration for sentiment steering experiment.

    Args:
        custom_config: Optional custom configuration to override defaults

    Returns:
        dict: Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    if custom_config:
        config.update(custom_config)
    return config


def run_sentiment_steering_experiment(custom_config=None):
    """
    Run sentiment steering experiment using the dynamic training system.

    Args:
        custom_config: Optional custom configuration
    """
    from src.train import main

    # Get configuration
    config = get_sentiment_steering_config(custom_config)

    # Set experiment type and run training
    config["experiment_type"] = "sentiment_steering"

    print("Starting Sentiment Steering Experiment...")
    print(f"Configuration: {config}")

    # Run training using the dynamic system
    main(experiment_type="sentiment_steering", custom_config=config)


def evaluate_sentiment_steering_model(model_path, prompt_text="I think", verbose=True):
    """
    Evaluate a trained sentiment steering model independently.

    Args:
        model_path: Path to the trained model
        prompt_text: Text prompt for generation
        verbose: Whether to print results
    """

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SLiMedNet(num_states=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.get('base_model'))
    tokenizer.pad_token = tokenizer.eos_token

    # Run evaluation
    run_sentiment_steering_evaluation(model, tokenizer, device, prompt_text, verbose)


if __name__ == "__main__":
    run_sentiment_steering_experiment()
