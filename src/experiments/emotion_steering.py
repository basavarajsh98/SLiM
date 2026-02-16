import os
import warnings
from collections import Counter
from random import shuffle

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from src.model import SLiMedNet
from src.inference import generate_text

warnings.filterwarnings("ignore")
from config.config import get_config

config = get_config()
# Default configuration for emotion steering experiment
DEFAULT_CONFIG = {
    "num_states": 5,
    "save_model_path": "resources/checkpoints/emotion_steering/emotions",
    "prompt_text": "i feel",
}


def get_emotion_evaluation_samples():
    """
    Get specialized evaluation samples for emotion steering experiments.

    Returns:
        list: List of emotion states with descriptions for evaluation
    """
    return [
        (torch.FloatTensor([1, 0, 0, 0, 0]), "Anger"),
        (torch.FloatTensor([0, 1, 0, 0, 0]), "Fear"),
        (torch.FloatTensor([0, 0, 1, 0, 0]), "Joy"),
        (torch.FloatTensor([0, 0, 0, 1, 0]), "Love"),
        (torch.FloatTensor([0, 0, 0, 0, 1]), "Sadness"),
    ]


def get_emotion_state_samples():
    """
    Get state samples for emotion steering experiments (for use in training).

    Returns:
        list: List of emotion state tensors for evaluation during training
    """
    return [
        torch.FloatTensor([1, 0, 0, 0, 0]),  # Anger
        torch.FloatTensor([0, 1, 0, 0, 0]),  # Fear
        torch.FloatTensor([0, 0, 1, 0, 0]),  # Joy
        torch.FloatTensor([0, 0, 0, 1, 0]),  # Love
        torch.FloatTensor([0, 0, 0, 0, 1]),  # Sadness
    ]


def run_emotion_steering_evaluation(
    model, tokenizer, device, prompt_text="i feel", verbose=True
):
    """
    Run specialized evaluation for emotion steering experiments.

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
    print("EMOTION STEERING EVALUATION RESULTS")
    print("=" * 60)

    # Baseline generation (no state)
    baseline_text = generate_text(
        model, tokenizer, prompt_text, state_tensor=None
    )
    print(f"Baseline (no state): {baseline_text}")
    print("-" * 60)

    # Get specialized evaluation samples
    eval_samples = get_emotion_evaluation_samples()

    # Generate text for each emotion state
    for i, (state_tensor, description) in enumerate(eval_samples):
        state_tensor = state_tensor.unsqueeze(0).to(device)

        generated_text = generate_text(
            model, tokenizer, prompt_text, state_tensor=state_tensor
        )

        print(f"State {i + 1} ({description}): {generated_text}")

    print("=" * 60 + "\n")


def one_hot_encode_emotions(emotion, num_classes=5):
    """
    Convert emotion label to one-hot encoded vector.

    Args:
        emotion: Emotion label (0-4 for anger, fear, joy, love, sadness)
        num_classes: Number of emotion classes

    Returns:
        numpy.ndarray: One-hot encoded vector
    """
    one_hot_vector = np.zeros(num_classes, dtype=int)
    one_hot_vector[[0, 1, 2, 3, 4].index(emotion)] = 1
    return one_hot_vector


def balance_dataset(texts, min_count=100):
    """
    Balance the dataset by downsampling to the minimum count across emotions.

    Args:
        texts: List of (text, label) tuples
        min_count: Minimum count per emotion category

    Returns:
        list: Balanced list of (text, label) tuples
    """
    # Count the number of samples in each emotion category
    counts = Counter([tuple(label) for _, label in texts])
    print("min_count: ", min_count)
    balanced_texts = []
    emotion_buckets = {emotion: [] for emotion in counts.keys()}

    # Organize samples by emotion
    for text, label in texts:
        emotion_buckets[tuple(label)].append((text, tuple(label)))

    # Downsample to the minimum count in each category
    for emotion, samples in emotion_buckets.items():
        shuffle(samples)  # Shuffle to avoid any ordering bias
        balanced_texts.extend(samples[:min_count])

    balanced_texts = [(text, list(map(int, label))) for text, label in balanced_texts]
    shuffle(balanced_texts)  # Shuffle again to mix emotions in the final dataset
    return balanced_texts


def prepare_dataset(max_sequence_length=64, path="resources/datasets/emotions.pth"):
    """
    Prepare dataset for emotion steering experiment.

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
        dataset = load_dataset(
            "ma2za/many_emotions", "raw"
        )  # ['anger', 'fear', 'joy', 'love', 'sadness']
        texts = []

        # Prepare texts and labels with one-hot encoding
        for item in dataset["en"]:
            label = item["label"]
            if label in [0, 1, 2, 3, 4]:
                one_hot_vector = one_hot_encode_emotions(label)
                texts.append((item["text"], one_hot_vector))

        # Balance the dataset
        print("Balancing dataset...")
        texts = balance_dataset(texts)

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
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(samples, path)
    else:
        print(f"Found tokenized dataset at {path}!\nImporting...")
        samples = torch.load(path)
        print("Done.")
        print("Total Size: ", len(samples))

    return samples, tokenizer


def get_emotion_steering_config(custom_config=None):
    """
    Get configuration for emotion steering experiment.

    Args:
        custom_config: Optional custom configuration to override defaults

    Returns:
        dict: Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    if custom_config:
        config.update(custom_config)
    return config


def run_emotion_steering_experiment(custom_config=None):
    """
    Run emotion steering experiment using the dynamic training system.

    Args:
        custom_config: Optional custom configuration
    """
    from src.train import main

    # Get configuration
    config = get_emotion_steering_config(custom_config)

    # Set experiment type and run training
    config["experiment_type"] = "emotion_steering"

    print("Starting Emotion Steering Experiment...")
    print(f"Configuration: {config}")

    # Run training using the dynamic system
    main(experiment_type="emotion_steering", custom_config=config)


def evaluate_emotion_steering_model(model_path, prompt_text="i feel", verbose=True):
    """
    Evaluate a trained emotion steering model independently.

    Args:
        model_path: Path to the trained model
        prompt_text: Text prompt for generation
        verbose: Whether to print results
    """

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SLiMedNet(num_states=5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.get('base_model'))
    tokenizer.pad_token = tokenizer.eos_token

    # Run evaluation
    run_emotion_steering_evaluation(model, tokenizer, device, prompt_text, verbose)


if __name__ == "__main__":
    run_emotion_steering_experiment()
