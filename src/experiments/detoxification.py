import os
import warnings
from random import shuffle

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from src.model import SLiMedNet
from src.inference import generate_text

warnings.filterwarnings("ignore")

# Default configuration for detoxification experiment
DEFAULT_CONFIG = {
    "num_states": 1,
    "max_sequence_length": 64,
    "batch_size": 2,
    "accumulation_steps": 4,
    "learning_rate": 2e-5,
    "epochs": 50,
    "max_steps": None,
    "save_model_path": "resources/checkpoints/SLiM_detoxification_wo_500",
    "eval_frequency": 50,
    "prompt_text": "I think",
}


def get_detoxification_evaluation_samples():
    """
    Get specialized evaluation samples for detoxification experiments.

    Returns:
        list: List of toxicity values with descriptions for evaluation
    """
    return [
        (0.0, "Non-toxic"),
        (0.1, "Very low toxicity"),
        (0.25, "Low toxicity"),
        (0.5, "Medium toxicity"),
        (0.75, "High toxicity"),
        (0.9, "Very high toxicity"),
        (1.0, "Maximum toxicity"),
    ]


def get_detoxification_state_samples():
    """
    Get state samples for detoxification experiments (for use in training).

    Returns:
        list: List of toxicity values for evaluation during training
    """
    return [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]


def run_detoxification_evaluation(
    model, tokenizer, device, prompt_text="I think", verbose=True
):
    """
    Run specialized evaluation for detoxification experiments.

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
    print("DETOXIFICATION EVALUATION RESULTS")
    print("=" * 60)

    # Baseline generation (no state)
    baseline_text = generate_text(
        model, tokenizer, prompt_text, max_length=40, state_tensor=None
    )
    print(f"Baseline (no state): {baseline_text}")
    print("-" * 60)

    # Get specialized evaluation samples
    eval_samples = get_detoxification_evaluation_samples()

    # Generate text for each toxicity level
    for toxicity_value, description in eval_samples:
        state_tensor = torch.FloatTensor([toxicity_value]).unsqueeze(0).to(device)

        generated_text = generate_text(
            model, tokenizer, prompt_text, max_length=40, state_tensor=state_tensor
        ).strip()

        print(f"Toxicity {toxicity_value:.2f} ({description}): {generated_text}")

    print("=" * 60 + "\n")


def balance_dataset(texts, limit_per_bin=500):
    # Define bins for toxicity scores
    bins = {
        "0.0-0.09": [],
        "0.1-0.19": [],
        "0.2-0.29": [],
        "0.3-0.39": [],
        "0.4-0.49": [],
        "0.5-0.59": [],
        "0.6-0.69": [],
        "0.7-0.79": [],
        "0.8-0.89": [],
        "0.9-1.0": [],
    }

    # Organize texts by toxicity bins
    for text, label in texts:
        toxicity_score = label[0]
        if 0 <= toxicity_score <= 0.09:
            bins["0.0-0.09"].append((text, label))
        elif 0.1 <= toxicity_score <= 0.19:
            bins["0.1-0.19"].append((text, label))
        elif 0.2 <= toxicity_score <= 0.29:
            bins["0.2-0.29"].append((text, label))
        elif 0.3 <= toxicity_score <= 0.39:
            bins["0.3-0.39"].append((text, label))
        elif 0.4 <= toxicity_score <= 0.49:
            bins["0.4-0.49"].append((text, label))
        elif 0.5 <= toxicity_score <= 0.59:
            bins["0.5-0.59"].append((text, label))
        elif 0.6 <= toxicity_score <= 0.69:
            bins["0.6-0.69"].append((text, label))
        elif 0.7 <= toxicity_score <= 0.79:
            bins["0.7-0.79"].append((text, label))
        elif 0.8 <= toxicity_score <= 0.89:
            bins["0.8-0.89"].append((text, label))
        elif 0.9 <= toxicity_score <= 1.0:
            bins["0.9-1.0"].append((text, label))

    # Limit each bin to a maximum of `limit_per_bin` samples
    balanced_texts = []
    for bin_samples in bins.values():
        shuffle(bin_samples)  # Shuffle to avoid any ordering bias
        balanced_texts.extend(bin_samples[:limit_per_bin])

    shuffle(balanced_texts)  # Shuffle again to mix samples from different bins
    return balanced_texts


def prepare_dataset(
    max_sequence_length=64, path="resources/datasets/detoxification_500.pth"
):
    """
    Prepare dataset for detoxification experiment.

    Args:
        max_sequence_length: Maximum sequence length for tokenization
        path: Path to save/load the processed dataset

    Returns:
        samples: List of (tokens, state) tuples
        tokenizer: The tokenizer used for processing
    """

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    if not os.path.exists(path):
        print("Loading dataset from HuggingFace...")
        dataset = load_dataset("google/civil_comments", split="train")
        texts = []

        for item in dataset:
            label = item["toxicity"]
            texts.append((item["text"], [round(label, 2)]))

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


def get_detoxification_config(custom_config=None):
    """
    Get configuration for detoxification experiment.

    Args:
        custom_config: Optional custom configuration to override defaults

    Returns:
        dict: Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    if custom_config:
        config.update(custom_config)
    return config


def run_detoxification_experiment(custom_config=None):
    """
    Run detoxification experiment using the dynamic training system.

    Args:
        custom_config: Optional custom configuration
    """
    from src.train import main

    # Get configuration
    config = get_detoxification_config(custom_config)

    # Set experiment type and run training
    config["experiment_type"] = "detoxification"

    print("Starting Detoxification Experiment...")
    print(f"Configuration: {config}")

    # Run training using the dynamic system
    main(experiment_type="detoxification", custom_config=config)


def evaluate_detoxification_model(model_path, prompt_text="I think", verbose=True):
    """
    Evaluate a trained detoxification model independently.

    Args:
        model_path: Path to the trained model
        prompt_text: Text prompt for generation
        verbose: Whether to print results
    """

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SLiMedNet(num_states=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Run evaluation
    run_detoxification_evaluation(model, tokenizer, device, prompt_text, verbose)


if __name__ == "__main__":
    # Example usage with custom configuration
    custom_config = {
        "epochs": 5,  # Reduced for demo
        "max_steps": 100,  # Limit steps for demo
        "eval_frequency": 25,
        "prompt_text": "I think",
        "save_model_path": "results/SLiM_detoxification_demo",
    }

    # Run the experiment
    run_detoxification_experiment(custom_config)

    # Example of independent evaluation (uncomment to use)
    # evaluate_detoxification_model("results/SLiM_detoxification_demo", "I think")
