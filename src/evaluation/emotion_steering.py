import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import pipeline

from config.config import get_config
from src.inference import generate_text, get_state_tensor, load_model_and_tokenizer

warnings.filterwarnings("ignore")
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

config = get_config()

NUM_STATES = 5
STATE_MAPPING = "emotion_mapping"


def get_distilroberta_classifier():
    """Get the emotion-english-distilroberta-base classifier."""
    classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True,
        device=config["device"],
    )
    return classifier


classifier = get_distilroberta_classifier()


# Define evaluation function
def evaluate_generations_with_auc(
    model, tokenizer, states, num_generations=100, num_runs=5, steered=False
):
    unknown_class_idx = len(states)
    states.append("Other")
    all_aucs = {state: [] for state in states}
    all_precisions = {state: [] for state in states}
    all_recalls = {state: [] for state in states}
    all_f1s = {state: [] for state in states}
    for run in range(num_runs):
        print(f"Run {run + 1} of {num_runs}")
        all_true_labels = []
        all_predicted_scores = {
            state: [] for state in states
        }  # Dictionary to store scores for each state
        y_pred = []
        y_true = []
        for state in states[:-1]:
            # Create state tensor
            state_tensor = (
                get_state_tensor(STATE_MAPPING, state, device) if steered else None
            )

            generated_texts = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt="I feel so",
                state_tensor=state_tensor,
                num_generations=num_generations,
            )
            # Get the classification probabilities for each generated text
            classifications = classifier(generated_texts)

            for i, classification in enumerate(classifications):
                # Store the true label as binary array for multi-class ROC (one-hot encoding style)
                true_label = [
                    1 if state == target_state else 0 for target_state in states[:-1]
                ]
                all_true_labels.append(true_label)

                # Store predicted scores for each state
                for target_state in states[:-1]:
                    score_state = "joy" if target_state == "love" else target_state
                    score = next(
                        (
                            x["score"]
                            for x in classification
                            if x["label"].lower() == score_state
                        ),
                        0,
                    )
                    all_predicted_scores[target_state].append(score)
                y_true.append(states.index(state))

                # Predicted label: Find the state with the highest classification score
                predicted_label = max(classification, key=lambda x: x["score"])[
                    "label"
                ].lower()

                if predicted_label == "joy" and state == "love":
                    predicted_label = "love"
                # If the predicted label is not in states, treat it as "unknown"
                if predicted_label in states:
                    predicted_label_idx = states.index(predicted_label)
                else:
                    predicted_label_idx = unknown_class_idx  # Assign to "unknown"

                y_pred.append(predicted_label_idx)
            # Generate word cloud for current state
            # generate_emotionrdcloud(generated_texts, state)

        # Calculate and plot ROC curves for each state
        fig, ax = plt.subplots(figsize=(10, 8))
        all_true_labels = np.array(all_true_labels)

        for idx, state in enumerate(states[:-1]):
            # Compute ROC curve and AUC for each state in a one-vs-rest fashion
            fpr, tpr, _ = roc_curve(
                all_true_labels[:, states.index(state)], all_predicted_scores[state]
            )
            auc = roc_auc_score(
                all_true_labels[:, states.index(state)], all_predicted_scores[state]
            )
            all_aucs[state].append(auc)

            # Plot ROC curve
            ax.plot(fpr, tpr, label=f"{state.capitalize()} (AUC = {auc:.2f})")

            true_state_labels = [1 if label == idx else 0 for label in y_true]
            predicted_state_labels = [1 if label == idx else 0 for label in y_pred]

            # Calculate Precision, Recall, and F1
            precision = precision_score(true_state_labels, predicted_state_labels)
            recall = recall_score(true_state_labels, predicted_state_labels)
            f1 = f1_score(true_state_labels, predicted_state_labels)

            all_precisions[state].append(precision)
            all_recalls[state].append(recall)
            all_f1s[state].append(f1)

    for state in states[:-1]:
        auc_mean = np.mean(all_aucs[state])
        auc_std = np.std(all_aucs[state])

        precision_mean = np.mean(all_precisions[state])
        precision_std = np.std(all_precisions[state])

        recall_mean = np.mean(all_recalls[state])
        recall_std = np.std(all_recalls[state])

        f1_mean = np.mean(all_f1s[state])
        f1_std = np.std(all_f1s[state])

        print(f"\nState: {state.capitalize()}")
        print(f"  AUC: {auc_mean:.2f} ± {auc_std:.2f}")
        print(f"  Precision: {precision_mean:.2f} ± {precision_std:.2f}")
        print(f"  Recall: {recall_mean:.2f} ± {recall_std:.2f}")
        print(f"  F1-score: {f1_mean:.2f} ± {f1_std:.2f}")

    # Plot settings
    ax.plot([0, 1], [0, 1], "k--")  # Diagonal line
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    # ax.set_title("ROC Curves with AUC for Positive and Negative Sentiment")
    ax.legend(loc="lower right")
    mode = "steered" if steered else "unsteered"
    plt.savefig(
        f"resources/evaluation_results/emotion/AUC_emotion_{mode}_evaluation_new.png"
    )


def evaluate_unsteered_generations_with_auc(
    model, tokenizer, states, num_generations=100
):
    results = {
        state: [] for state in states
    }  # Store binary labels and scores per state for AUC calculation
    all_true_labels = []
    all_predicted_scores = {
        state: [] for state in states
    }  # Dictionary to store scores for each state
    y_pred = []
    y_true = []
    unknown_class_idx = len(states)
    states.append("Other")
    for state in states[:-1]:  # Skip the last "unknown" class in the loop
        # Create state tensor
        state_tensor = get_state_tensor(STATE_MAPPING, state, device)

        generated_texts = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt="I feel so",
            state_tensor=None,
            num_generations=num_generations,
        )
        # Get the classification probabilities for each generated text
        classifications = classifier(generated_texts)

        for i, classification in enumerate(classifications):
            # Store the true label as binary array for multi-class ROC (one-hot encoding style)
            true_label = [
                1 if state == target_state else 0 for target_state in states[:-1]
            ]
            all_true_labels.append(true_label)

            # Store predicted scores for each state
            for target_state in states[:-1]:
                score_state = "joy" if target_state == "love" else target_state
                score = next(
                    (
                        x["score"]
                        for x in classification
                        if x["label"].lower() == score_state
                    ),
                    0,
                )
                all_predicted_scores[target_state].append(score)

            y_true.append(states.index(state))

            # Predicted label: Find the state with the highest classification score
            predicted_label = max(classification, key=lambda x: x["score"])[
                "label"
            ].lower()

            if predicted_label == "joy" and state == "love":
                predicted_label = "love"
            # If the predicted label is not in states, treat it as "unknown"
            if predicted_label in states:
                predicted_label_idx = states.index(predicted_label)
            else:
                predicted_label_idx = unknown_class_idx  # Assign to "unknown"

            y_pred.append(predicted_label_idx)

    # Calculate and plot ROC curves for each state
    fig, ax = plt.subplots(figsize=(10, 8))
    all_true_labels = np.array(all_true_labels)

    for idx, state in enumerate(states[:-1]):
        # Compute ROC curve and AUC for each state in a one-vs-rest fashion
        fpr, tpr, _ = roc_curve(
            all_true_labels[:, states.index(state)], all_predicted_scores[state]
        )
        auc = roc_auc_score(
            all_true_labels[:, states.index(state)], all_predicted_scores[state]
        )
        auc = roc_auc_score(
            all_true_labels[:, states.index(state)], all_predicted_scores[state]
        )
        # precision, recall, _ = precision_recall_curve(all_true_labels[:, states.index(state)], all_predicted_scores[state])
        # Plot ROC curve
        ax.plot(fpr, tpr, label=f"{state.capitalize()} (AUC = {auc:.2f})")
        # Print AUC, Precision, and Recall
        # print(f"State: {state.capitalize()}")
        print(f"  AUC: {auc:.2f}")

        true_state_labels = [1 if label == idx else 0 for label in y_true]
        predicted_state_labels = [1 if label == idx else 0 for label in y_pred]

        # Calculate Precision, Recall, and F1
        precision = precision_score(true_state_labels, predicted_state_labels)
        recall = recall_score(true_state_labels, predicted_state_labels)
        f1 = f1_score(true_state_labels, predicted_state_labels)

        print(f"State: {state.capitalize()}")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall: {recall:.2f}")
        print(f"  F1-score: {f1:.2f}\n")

    # Configure plot settings
    ax.plot([0, 1], [0, 1], "k--")  # Diagonal line
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    # ax.set_title("ROC Curves with AUC for Each Emotion")
    ax.legend(loc="lower right")
    plt.savefig("resources/evaluation_results/emotion/AUC_emo_unsteered_26.11.png")


if __name__ == "__main__":
    print("Loading checkpoint...\n")
    checkpoint = "./resources/checkpoints/emotion_steering/emotions.pth"
    model, tokenizer = load_model_and_tokenizer(checkpoint, NUM_STATES)
    device = torch.device(config["device"])
    model = model.to(device)
    # Define state list and device
    states1 = ["anger", "fear", "joy", "sadness", "love"]
    # Run evaluation
    evaluate_generations_with_auc(model, tokenizer, states1, steered=True)
    evaluate_generations_with_auc(model, tokenizer, states1, steered=False)
