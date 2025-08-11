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
# Default configuration for topic steering experiment
DEFAULT_CONFIG = {
    "num_states": 10,
    "save_model_path": "resources/checkpoints/topics_steering/topics",
    "prompt_text": "I like",
}

MAX_SAMPLES_PER_TOPIC = 500

# Topics to index mapping
TOPIC_TO_INDEX = {
    "gourmet-food": 0,
    "video-game": 1,
    "clothing": 2,
    "beauty": 3,
    "arts": 4,
    "book": 5,
    "jewelry": 6,
    "shoe": 7,
    "musical-instrument": 8,
    "electronics": 9,
}

# Index to topic mapping for evaluation
INDEX_TO_TOPIC = {v: k for k, v in TOPIC_TO_INDEX.items()}


def get_topic_evaluation_samples():
    """
    Get specialized evaluation samples for topic steering experiments.

    Returns:
        list: List of topic states with descriptions for evaluation
    """
    return [
        (torch.FloatTensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), "Gourmet Food"),
        (torch.FloatTensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]), "Video Games"),
        (torch.FloatTensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]), "Clothing"),
        (torch.FloatTensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]), "Beauty"),
        (torch.FloatTensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), "Arts"),
        (torch.FloatTensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]), "Books"),
        (torch.FloatTensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]), "Jewelry"),
        (torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]), "Shoes"),
        (torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]), "Musical Instruments"),
        (torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), "Electronics"),
        (torch.FloatTensor([0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0]), "Food & Games"),
        (torch.FloatTensor([0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0]), "Fashion"),
        (
            torch.FloatTensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            "All Topics",
        ),
    ]


def get_topic_state_samples():
    """
    Get state samples for topic steering experiments (for use in training).

    Returns:
        list: List of topic state tensors for evaluation during training
    """
    return [
        torch.FloatTensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # Gourmet Food
        torch.FloatTensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),  # Video Games
        torch.FloatTensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),  # Clothing
        torch.FloatTensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),  # Beauty
        torch.FloatTensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),  # Arts
        torch.FloatTensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),  # Books
        torch.FloatTensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),  # Jewelry
        torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),  # Shoes
        torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),  # Musical Instruments
        torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),  # Electronics
    ]


def run_topic_steering_evaluation(
    model, tokenizer, device, prompt_text="I like", verbose=True
):
    """
    Run specialized evaluation for topic steering experiments.

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
    print("TOPIC STEERING EVALUATION RESULTS")
    print("=" * 60)

    # Baseline generation (no state)
    baseline_text = generate_text(
        model, tokenizer, prompt_text, max_length=40, state_tensor=None
    )
    print(f"Baseline (no state): {baseline_text}")
    print("-" * 60)

    # Get specialized evaluation samples
    eval_samples = get_topic_evaluation_samples()

    # Generate text for each topic state
    for i, (state_tensor, description) in enumerate(eval_samples):
        state_tensor = state_tensor.unsqueeze(0).to(device)

        generated_text = generate_text(
            model, tokenizer, prompt_text, max_length=40, state_tensor=state_tensor
        )

        print(f"State {i + 1} ({description}): {generated_text}")

    print("=" * 60 + "\n")


def balance_dataset(texts, min_count=500):
    """
    Balance the dataset by downsampling to the minimum count across topic categories.

    Args:
        texts: List of (text, label) tuples
        min_count: Minimum count per topic category

    Returns:
        list: Balanced list of (text, label) tuples
    """
    # Count the number of samples in each topic category
    counts = Counter([tuple(label) for _, label in texts])
    print("min_count: ", min_count)
    balanced_texts = []
    topic_buckets = {topic: [] for topic in counts.keys()}

    # Organize samples by topic
    for text, label in texts:
        topic_buckets[tuple(label)].append((text, tuple(label)))

    # Downsample to the minimum count in each category
    for topic, samples in topic_buckets.items():
        shuffle(samples)  # Shuffle to avoid any ordering bias
        balanced_texts.extend(samples[:min_count])

    balanced_texts = [(text, list(map(int, label))) for text, label in balanced_texts]
    shuffle(balanced_texts)  # Shuffle again to mix topics in the final dataset
    return balanced_texts


def one_hot_encode(label, num_labels=10):
    """
    Convert topic label to one-hot encoded vector.

    Args:
        label: Topic label (0-9)
        num_labels: Number of topic classes

    Returns:
        numpy.ndarray: One-hot encoded vector
    """
    if label < 0 or label >= num_labels:
        raise ValueError("Label must be between 0 and 9.")
    one_hot_vector = np.zeros(num_labels)
    one_hot_vector[label] = 1
    return one_hot_vector


def prepare_dataset(max_sequence_length=64, path="resources/datasets/topics.pth"):
    """
    Prepare dataset for topic steering experiment.

    Args:
        max_sequence_length: Maximum sequence length for tokenization
        path: Path to save/load the processed dataset

    Returns:
        samples: List of (tokens, state) tuples
        tokenizer: The tokenizer used for processing
    """

    tokenizer = AutoTokenizer.from_pretrained(config.get("base_model"))
    tokenizer.pad_token = tokenizer.eos_token

    if not os.path.exists(path):
        print("Loading dataset from HuggingFace...")
        datasets = [
            (
                "gourmet-food",
                load_dataset("contemmcm/amazon_reviews_2013", "gourmet-food"),
            ),
            ("video-game", load_dataset("contemmcm/amazon_reviews_2013", "video-game")),
            ("clothing", load_dataset("contemmcm/amazon_reviews_2013", "clothing")),
            ("beauty", load_dataset("contemmcm/amazon_reviews_2013", "beauty")),
            ("arts", load_dataset("contemmcm/amazon_reviews_2013", "arts")),
            ("book", load_dataset("contemmcm/amazon_reviews_2013", "book")),
            ("jewelry", load_dataset("contemmcm/amazon_reviews_2013", "jewelry")),
            ("shoe", load_dataset("contemmcm/amazon_reviews_2013", "shoe")),
            (
                "musical-instrument",
                load_dataset("contemmcm/amazon_reviews_2013", "musical-instrument"),
            ),
            (
                "electronics",
                load_dataset("contemmcm/amazon_reviews_2013", "electronics"),
            ),
        ]

        texts = []
        for topic_name, dataset in datasets:
            topic_index = TOPIC_TO_INDEX[topic_name]
            print(f"Processing {topic_name}...")
            for item in dataset["train"]:
                if len(item["text"]) > 10:  # Filter out very short texts
                    one_hot_label = one_hot_encode(topic_index)
                    texts.append((item["text"], one_hot_label))

        # Balance the dataset
        print("Balancing dataset...")
        texts = balance_dataset(texts)
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


def get_topic_steering_config(custom_config=None):
    """
    Get configuration for topic steering experiment.

    Args:
        custom_config: Optional custom configuration to override defaults

    Returns:
        dict: Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    if custom_config:
        config.update(custom_config)
    return config


def run_topic_steering_experiment(custom_config=None):
    """
    Run topic steering experiment using the dynamic training system.

    Args:
        custom_config: Optional custom configuration
    """
    from src.train import main

    # Get configuration
    config = get_topic_steering_config(custom_config)

    # Set experiment type and run training
    config["experiment_type"] = "topic_steering"

    print("Starting Topic Steering Experiment...")
    print(f"Configuration: {config}")

    # Run training using the dynamic system
    main(experiment_type="topic_steering", custom_config=config)


def evaluate_topic_steering_model(model_path, prompt_text="I like", verbose=True):
    """
    Evaluate a trained topic steering model independently.

    Args:
        model_path: Path to the trained model
        prompt_text: Text prompt for generation
        verbose: Whether to print results
    """

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SLiMedNet(num_states=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.get("base_model"))
    tokenizer.pad_token = tokenizer.eos_token

    # Run evaluation
    run_topic_steering_evaluation(model, tokenizer, device, prompt_text, verbose)


if __name__ == "__main__":
    run_topic_steering_experiment()
