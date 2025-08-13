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
# Default configuration for language steering experiment
DEFAULT_CONFIG = {
    "num_states": 2,
    "save_model_path": "resources/checkpoints/language_steering/languages",
    "prompt_text": "I feel",
}


def get_language_evaluation_samples():
    """
    Get specialized evaluation samples for language steering experiments.

    Returns:
        list: List of language states with descriptions for evaluation
    """
    return [
        (torch.FloatTensor([1, 0]), "English"),
        (torch.FloatTensor([0, 1]), "German"),
    ]


def get_language_state_samples():
    """
    Get state samples for language steering experiments (for use in training).

    Returns:
        list: List of language state tensors for evaluation during training
    """
    return [
        torch.FloatTensor([1, 0]),  # English
        torch.FloatTensor([0, 1]),  # German
    ]


def run_language_steering_evaluation(
    model, tokenizer, device, prompt_text="I feel", verbose=True
):
    """
    Run specialized evaluation for language steering experiments.

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
    print("LANGUAGE STEERING EVALUATION RESULTS")
    print("=" * 60)

    # Baseline generation (no state)
    baseline_text = generate_text(
        model, tokenizer, prompt_text, max_length=40, state_tensor=None
    )
    print(f"Baseline (no state): {baseline_text}")
    print("-" * 60)

    # Get specialized evaluation samples
    eval_samples = get_language_evaluation_samples()

    # Generate text for each language state
    for i, (state_tensor, description) in enumerate(eval_samples):
        state_tensor = state_tensor.unsqueeze(0).to(device)

        generated_text = generate_text(
            model, tokenizer, prompt_text, max_length=40, state_tensor=state_tensor
        )

        print(f"State {i + 1} ({description}): {generated_text}")

    print("=" * 60 + "\n")


def prepare_dataset(max_sequence_length=64, path="resources/datasets/languages.pth"):
    """
    Prepare dataset for language steering experiment.

    Args:
        max_sequence_length: Maximum sequence length for tokenization
        path: Path to save/load the processed dataset

    Returns:
        samples: List of (tokens, state) tuples
        tokenizer: The tokenizer used for processing
    """

    tokenizer = AutoTokenizer.from_pretrained(config.get('base_model'))
    tokenizer.pad_token = tokenizer.eos_token

    if not os.path.exists(path):
        print("Loading dataset from HuggingFace...")
        dataset = load_dataset("ma2za/many_emotions", "raw")  # ['en', 'de']
        texts1 = []

        # Prepare texts and labels with one-hot encoding for English
        for item in dataset["en"]:
            label = item["label"]
            if label in [6]:
                texts1.append((item["text"], [1, 0]))
        texts1 = texts1[:500]

        # Prepare texts and labels with one-hot encoding for German
        texts2 = []
        for item in dataset["de"]:
            label = item["label"]
            if label in [6]:
                texts2.append((item["text"], [0, 1]))
        texts2 = texts2[:500]

        texts = texts1 + texts2

        print("Sample: ", texts[0])
        print("Sample: ", texts[-1])
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


def get_language_steering_config(custom_config=None):
    """
    Get configuration for language steering experiment.

    Args:
        custom_config: Optional custom configuration to override defaults

    Returns:
        dict: Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    if custom_config:
        config.update(custom_config)
    return config


def run_language_steering_experiment(custom_config=None):
    """
    Run language steering experiment using the dynamic training system.

    Args:
        custom_config: Optional custom configuration
    """
    from src.train import main

    # Get configuration
    config = get_language_steering_config(custom_config)

    # Set experiment type and run training
    config["experiment_type"] = "language_steering"

    print("Starting Language Steering Experiment...")
    print(f"Configuration: {config}")

    # Run training using the dynamic system
    main(experiment_type="language_steering", custom_config=config)


def evaluate_language_steering_model(model_path, prompt_text="I feel", verbose=True):
    """
    Evaluate a trained language steering model independently.

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
    run_language_steering_evaluation(model, tokenizer, device, prompt_text, verbose)


if __name__ == "__main__":
    run_language_steering_experiment()
