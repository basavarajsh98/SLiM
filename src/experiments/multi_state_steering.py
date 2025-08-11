import os
import warnings
from collections import defaultdict
from random import shuffle, uniform

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from src.model import SLiMedNet
from src.inference import generate_text

warnings.filterwarnings("ignore")
from config.config import get_config

config = get_config()
# Default configuration for multi-state steering experiment
DEFAULT_CONFIG = {
    "num_states": 3,
    "save_model_path": "resources/checkpoints/multi_state_steering/multi_states",
    "max_samples_per_combination": 500,
    "min_samples_per_bin": 10000,
    "prompt_text": "This product is",
}

# Topics to index mapping
TOPIC_TO_INDEX = {
    "book": 0,
    "electronics": 1,
    "apparel": 2,
    "beauty": 3,
}
LANG_TO_INDEX = {"en": 0, "de": 1}
NUM_TOPICS = len(TOPIC_TO_INDEX)  # Number of unique topics

# Index to topic mapping for evaluation
INDEX_TO_TOPIC = {v: k for k, v in TOPIC_TO_INDEX.items()}
INDEX_TO_LANG = {v: k for k, v in LANG_TO_INDEX.items()}


def get_multi_state_evaluation_samples():
    """
    Get specialized evaluation samples for multi-state steering experiments.

    Returns:
        list: List of multi-state combinations with descriptions for evaluation
    """
    return [
        # Pure language states with neutral topic and rating
        (torch.FloatTensor([1, 0, 0, 0, 0, 0, 0]), "English Language"),
        (torch.FloatTensor([0, 1, 0, 0, 0, 0, 0]), "German Language"),
        # Pure topic states with neutral language and rating
        (torch.FloatTensor([0, 0, 1, 0, 0, 0, 0]), "Book Topic"),
        (torch.FloatTensor([0, 0, 0, 1, 0, 0, 0]), "Electronics Topic"),
        (torch.FloatTensor([0, 0, 0, 0, 1, 0, 0]), "Apparel Topic"),
        (torch.FloatTensor([0, 0, 0, 0, 0, 1, 0]), "Beauty Topic"),
        # Pure rating states with neutral language and topic
        (torch.FloatTensor([0, 0, 0, 0, 0, 0, 0.2]), "Low Rating (1-2 stars)"),
        (torch.FloatTensor([0, 0, 0, 0, 0, 0, 0.5]), "Medium Rating (3 stars)"),
        (torch.FloatTensor([0, 0, 0, 0, 0, 0, 0.8]), "High Rating (4-5 stars)"),
        # Mixed states
        (torch.FloatTensor([1, 0, 1, 0, 0, 0, 0.8]), "English + Book + High Rating"),
        (
            torch.FloatTensor([0, 1, 0, 1, 0, 0, 0.2]),
            "German + Electronics + Low Rating",
        ),
        (
            torch.FloatTensor([0.5, 0.5, 0, 0, 0.5, 0.5, 0.5]),
            "Mixed Language + Fashion + Medium Rating",
        ),
    ]


def get_multi_state_state_samples():
    """
    Get state samples for multi-state steering experiments (for use in training).

    Returns:
        list: List of multi-state tensors for evaluation during training
    """
    return [
        # Language states
        torch.FloatTensor([1, 0, 0, 0, 0, 0, 0]),  # English
        torch.FloatTensor([0, 1, 0, 0, 0, 0, 0]),  # German
        # Topic states
        torch.FloatTensor([0, 0, 1, 0, 0, 0, 0]),  # Book
        torch.FloatTensor([0, 0, 0, 1, 0, 0, 0]),  # Electronics
        torch.FloatTensor([0, 0, 0, 0, 1, 0, 0]),  # Apparel
        torch.FloatTensor([0, 0, 0, 0, 0, 1, 0]),  # Beauty
        # Rating states
        torch.FloatTensor([0, 0, 0, 0, 0, 0, 0.2]),  # Low
        torch.FloatTensor([0, 0, 0, 0, 0, 0, 0.5]),  # Medium
        torch.FloatTensor([0, 0, 0, 0, 0, 0, 0.8]),  # High
    ]


def run_multi_state_steering_evaluation(
    model, tokenizer, device, prompt_text="This product is", verbose=True
):
    """
    Run specialized evaluation for multi-state steering experiments.

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
    print("MULTI-STATE STEERING EVALUATION RESULTS")
    print("=" * 60)

    # Baseline generation (no state)
    baseline_text = generate_text(
        model, tokenizer, prompt_text, max_length=40, state_tensor=None
    )
    print(f"Baseline (no state): {baseline_text}")
    print("-" * 60)

    # Get specialized evaluation samples
    eval_samples = get_multi_state_evaluation_samples()

    # Generate text for each multi-state combination
    for i, (state_tensor, description) in enumerate(eval_samples):
        state_tensor = state_tensor.unsqueeze(0).to(device)

        generated_text = generate_text(
            model, tokenizer, prompt_text, max_length=40, state_tensor=state_tensor
        )

        print(f"State {i + 1} ({description}): {generated_text}")

    print("=" * 60 + "\n")


def randomize_rating(rating):
    """Convert rating to a random continuous value in the specified ranges."""
    if rating == 0:
        return round(
            uniform(0, 0.19), 2
        )  # Random value between 0 and 0.5 for ratings 1-2
    if rating == 1:
        return round(
            uniform(0.2, 0.39), 2
        )  # Random value between 0 and 0.5 for ratings 1-2
    elif rating == 2:
        return round(
            uniform(0.4, 0.59), 2
        )  # Random value between 0 and 0.5 for ratings 1-2
    elif rating == 3:
        return round(
            uniform(0.6, 0.79), 2
        )  # Random value between 0 and 0.5 for ratings 1-2
    elif rating == 4:
        return round(
            uniform(0.8, 1), 2
        )  # Random value between 0 and 0.5 for ratings 1-2


def balance_dataset(texts, min_samples_per_bin=10000):
    """Balance dataset to ensure each topic-rating pair has at least `min_samples_per_bin` samples."""
    # Group samples by (topic, rating) pairs
    bins = {
        topic: {rating: [] for rating in range(5)} for topic in TOPIC_TO_INDEX.values()
    }
    sample_counts = defaultdict(lambda: defaultdict(int))

    # Organize samples into bins by topic and rating
    for text, label in texts:
        topic = label[0]
        rating = label[1]
        bins[topic][rating].append((text, label))

    # Balance each topic-rating bin
    balanced_texts = []
    for topic, topic_bins in bins.items():
        for rating, bin_samples in topic_bins.items():
            if len(bin_samples) > min_samples_per_bin:
                # Downsample if there are too many samples in the bin
                shuffle(bin_samples)
                bin_samples = bin_samples[:min_samples_per_bin]

            # Convert ratings to continuous values after balancing
            balanced_texts.extend(
                [
                    (text, [label[0], randomize_rating(rating)])
                    for text, label in bin_samples
                ]
            )
            sample_counts[topic][rating] = len(bin_samples)

    # Display the count of samples per topic per rating
    print("Sample counts per topic per rating:")
    for topic, rating_counts in sample_counts.items():
        for rating, count in rating_counts.items():
            print(f"Topic {topic}, Rating {rating}: {count} samples")

    shuffle(balanced_texts)  # Shuffle to mix bins in the final dataset
    return balanced_texts


def prepare_dataset(max_sequence_length, path="resources/datasets/multi_states.pth"):
    """Prepare the multi-state steering dataset with topic, language, and rating information."""
    tokenizer = AutoTokenizer.from_pretrained(config.get('base_model'))
    tokenizer.pad_token = tokenizer.eos_token

    # Check if dataset file already exists
    if not os.path.exists(path):
        MAX_SAMPLES_PER_COMBINATION = (
            500  # Minimum samples per (topic, language, rating)
        )

        # Load dataset
        dataset = load_dataset("csv", data_files="resources/datasets/amazon_product_review.csv")
        texts = []

        # Track counts for each (topic, language, rating) combination
        combination_counts = defaultdict(int)

        print("Started preparing data...")
        for item in dataset["train"]:
            topic = item["product_category"]
            topic_index = TOPIC_TO_INDEX.get(topic)

            # Only process known topics
            if topic_index is not None:
                rating = item.get("stars")
                lang = item.get("language")

                # Ensure valid language and rating presence
                if lang in ["en", "de"] and rating is not None:
                    combination_key = (topic, lang, rating)

                    # Check if the combination has reached the max sample limit
                    if (
                        combination_counts[combination_key]
                        < MAX_SAMPLES_PER_COMBINATION
                    ):
                        texts.append(
                            (
                                item["review_body"],
                                [LANG_TO_INDEX.get(lang), topic_index, rating],
                            )
                        )
                        combination_counts[combination_key] += 1

        print("Data preparation complete. Sample counts per topic:")
        for topic, count in combination_counts.items():
            print(f"{topic}: {count} samples")

        # Balance the dataset
        print("Balancing dataset...")
        # texts = balance_dataset(texts, min_samples_per_bin=10000)

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
        print("Total Size: ", len(samples))
        torch.save(samples, path)
    else:
        print(f"Found tokenized dataset at {path}!\nImporting...")
        samples = torch.load(path)
        print("Done.")
        print("Total Size: ", len(samples))
    return samples, tokenizer


def get_multi_state_steering_config(custom_config=None):
    """Get configuration for multi-state steering experiment."""
    config = DEFAULT_CONFIG.copy()
    if custom_config:
        config.update(custom_config)
    return config


def run_multi_state_steering_experiment(custom_config=None):
    """Run the multi-state steering experiment using the centralized training system."""
    config = get_multi_state_steering_config(custom_config)
    config["experiment_type"] = "multi_state_steering"

    # Import here to avoid circular imports
    from src.train import main

    main(experiment_type="multi_state_steering", custom_config=config)


def evaluate_multi_state_steering_model(
    model_path, prompt_text="This product is", verbose=True
):
    """
    Evaluate a trained multi-state steering model independently.

    Args:
        model_path: Path to the trained model
        prompt_text: Text prompt for generation
        verbose: Whether to print results
    """

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SLiMedNet(num_states=7)  # 2 language + 4 topic + 1 rating
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.get('base_model'))
    tokenizer.pad_token = tokenizer.eos_token

    # Run evaluation
    run_multi_state_steering_evaluation(model, tokenizer, device, prompt_text, verbose)


if __name__ == "__main__":
    run_multi_state_steering_experiment()
