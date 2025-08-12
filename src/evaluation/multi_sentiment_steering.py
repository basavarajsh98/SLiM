import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import pipeline

from config.config import get_config
from src.inference import generate_text, get_state_tensor, load_model_and_tokenizer

warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, roc_curve

config = get_config()


NUM_STATES = 3
STATE_MAPPING = "multi_state_mapping"


# Initialize the classifier
def get_distilroberta_classifier():
    """Get the emotion-english-distilroberta-base classifier."""
    classifier = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        return_all_scores=False,
        device=config["device"],
    )
    return classifier


classifier = get_distilroberta_classifier()


# Define evaluation function
def evaluate_generations_with_auc(model, tokenizer, states, num_generations=100):
    # Dictionary to store binary labels and scores per state for AUC calculation
    all_true_labels = []
    all_predicted_scores = {state: [] for state in states}

    for state in states:
        # Create state tensor
        state_tensor = get_state_tensor(STATE_MAPPING, state, device)

        generated_texts = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt="I feel",
            state_tensor=state_tensor,
            num_generations=num_generations,
        )
        # Get the classification results for each generated text
        classifications = classifier(generated_texts)

        for classification in classifications:
            # Store the true label as binary array for multi-class ROC (one-hot encoding)
            true_label = [1 if state == target_state else 0 for target_state in states]
            all_true_labels.append(true_label)

            # Store the predicted scores for each state
            for target_state in states:
                score = 0
                score = (
                    classification["score"]
                    if classification["label"].lower() == target_state
                    else 0
                )
                all_predicted_scores[target_state].append(score)

    # Plot ROC curves for each state
    fig, ax = plt.subplots(figsize=(10, 8))
    all_true_labels = np.array(all_true_labels)

    for state in states:
        # Compute ROC curve and AUC for each state in a one-vs-rest fashion
        fpr, tpr, _ = roc_curve(
            all_true_labels[:, states.index(state)], all_predicted_scores[state]
        )
        auc = roc_auc_score(
            all_true_labels[:, states.index(state)], all_predicted_scores[state]
        )

        # Plot ROC curve
        ax.plot(fpr, tpr, label=f"{state.capitalize()} (AUC = {auc:.2f})")

    # Plot settings
    ax.plot([0, 1], [0, 1], "k--")  # Diagonal line
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    # ax.set_title("ROC Curves with AUC for Positive and Negative Sentiment")
    ax.legend(loc="lower right")
    plt.savefig("resources/evaluation_results/multi_state/AUC_senti_steered.png")


def evaluate_unsteered_generations_with_auc(
    model, tokenizer, states, num_generations=100
):
    # Dictionary to store binary labels and scores per state for AUC calculation
    all_true_labels = []
    all_predicted_scores = {state: [] for state in states}

    for state in states:
        # Create state tensor
        state_tensor = get_state_tensor(STATE_MAPPING, state, device)

        generated_texts = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt="I feel",
            state_tensor=None,
            num_generations=num_generations,
        )
        # Get the classification results for each generated text
        classifications = classifier(generated_texts)

        for classification in classifications:
            # Store the true label as binary array for multi-class ROC (one-hot encoding)
            true_label = [1 if state == target_state else 0 for target_state in states]
            all_true_labels.append(true_label)

            # Store the predicted scores for each state
            for target_state in states:
                score = 0
                score = (
                    classification["score"]
                    if classification["label"].lower() == target_state
                    else 0
                )
                all_predicted_scores[target_state].append(score)

    # Plot ROC curves for each state
    fig, ax = plt.subplots(figsize=(10, 8))
    all_true_labels = np.array(all_true_labels)

    for state in states:
        # Compute ROC curve and AUC for each state in a one-vs-rest fashion
        fpr, tpr, _ = roc_curve(
            all_true_labels[:, states.index(state)], all_predicted_scores[state]
        )
        auc = roc_auc_score(
            all_true_labels[:, states.index(state)], all_predicted_scores[state]
        )

        # Plot ROC curve
        ax.plot(fpr, tpr, label=f"{state.capitalize()} (AUC = {auc:.2f})")

    # Plot settings
    ax.plot([0, 1], [0, 1], "k--")  # Diagonal line
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    # ax.set_title("ROC Curves with AUC for Positive and Negative Sentiment")
    ax.legend(loc="lower right")
    plt.savefig("resources/evaluation_results/multi_state/AUC_senti_steered.png")


if __name__ == "__main__":
    print("Loading checkpoint...\n")
    checkpoint = "./resources/checkpoints/multi_state_steering/multi_states.pth"
    model, tokenizer = load_model_and_tokenizer(checkpoint, NUM_STATES)
    device = torch.device(config["device"])
    model = model.to(device)
    # Define state list and device
    states = ["positive", "negative"]
    # Run evaluation
    evaluate_generations_with_auc(model, tokenizer, states)
    evaluate_unsteered_generations_with_auc(model, tokenizer, states)
