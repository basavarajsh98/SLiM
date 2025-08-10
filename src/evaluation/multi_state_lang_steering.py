import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

from config.config import get_config
from src.inference import generate_text, get_state_tensor, load_model_and_tokenizer

warnings.filterwarnings("ignore")
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

config = get_config()

NUM_STATES = 3
STATE_MAPPING = "multi_state_mapping"


# Compute similarity between generated text and average topic embedding
def compute_similarity(generated_sentence, avg_topic_embedding):
    generated_sentence_embeddings = model_rel.encode(generated_sentence)
    return cosine_similarity(
        generated_sentence_embeddings.reshape(1, -1), avg_topic_embedding
    )[0][0]


model_rel = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to("cuda")


# Function to get average topic embedding
def get_average_topic_embedding():
    import os

    if os.path.exists("./SLiM/average.embedding.pth"):
        return torch.load("./SLiM/average.embedding.pth")
    else:
        topics = [
            "gourmet-food",
            "video-game",
            "clothing",
            "beauty",
            "arts",
            "book",
            "jewelry",
            "shoe",
            "musical-instrument",
            "electronics",
        ]
        datasets = {
            topic: load_dataset(
                "contemmcm/amazon_reviews_2013", topic, split="complete"
            )
            for topic in topics
        }
        from random import shuffle

        avg_topic_embedding = {topic: [] for topic in topics}
        for i, topic in enumerate(datasets):
            shuffle(datasets[topic]["review/text"])
            sampled_data = datasets[topic]["review/text"][:5000]
            print(f"Topic Size: {topic}: ", len(sampled_data))
            embeddings = np.array(
                [model_rel.encode(sentence) for sentence in sampled_data]
            )
            avg_topic_embedding[topic].append(np.mean(embeddings, axis=0))
        torch.save(avg_topic_embedding, "average.embedding.pth")
        return avg_topic_embedding


# Initialize the classifier
def get_distilroberta_lang_classifier():
    """Get the emotion-english-distilroberta-base classifier."""
    classifier = pipeline(
        "text-classification",
        model="papluca/xlm-roberta-base-language-detection",
        return_all_scores=False,
    )
    return classifier


lang_classifier = get_distilroberta_lang_classifier()


# Initialize the classifier
def get_distilroberta_senti_classifier():
    """Get the emotion-english-distilroberta-base classifier."""
    classifier = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        return_all_scores=False,
    )
    return classifier


senti_classifier = get_distilroberta_senti_classifier()
# Updated evaluation functions to accumulate data and plot all ROC curves on one plot for each type


def calculate_metrics(true_labels, predicted_labels):
    """Calculate accuracy, precision, recall, F1 for a given set of true and predicted labels."""
    metrics = {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels),
        "recall": recall_score(true_labels, predicted_labels),
        "f1": f1_score(true_labels, predicted_labels),
    }
    return metrics


def evaluate_combined_states_with_metrics(
    model, tokenizer, combined_states, topic_embeddings, num_generations=100
):
    steered_results = []
    unsteered_results = []
    similarity_results = []

    for state in combined_states:
        topic, lang, rating = state.split("_")
        state_tensor = get_state_tensor(STATE_MAPPING, state, device)

        # Generate texts with steering (conditioned on state)
        generated_texts = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt="I feel",
            state_tensor=state_tensor,
            num_generations=num_generations,
        )

        # Generate texts without steering (model's natural output)
        unconditioned_texts = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt="I feel",
            state_tensor=None,
            num_generations=num_generations,
        )

        # Classify language, sentiment, and compute topic similarity for steered texts
        lang_classifications = lang_classifier(generated_texts)
        senti_classifications = senti_classifier(generated_texts)
        topic_similarity_scores = [
            compute_similarity(text, topic_embeddings[topic])
            for text in generated_texts
        ]

        # Classify language and sentiment for unsteered texts
        unsteered_lang_classifications = lang_classifier(unconditioned_texts)
        unsteered_senti_classifications = senti_classifier(unconditioned_texts)
        unconditioned_similarity_scores = [
            compute_similarity(text, topic_embeddings[topic])
            for text in unconditioned_texts
        ]

        # True labels for target state
        true_lang_label = [
            1 if lang == classification["label"] else 0
            for classification in lang_classifications
        ]
        true_rating_label = [
            1 if int(rating) in [4, 5] else 0 for _ in range(num_generations)
        ]

        # Predicted labels (binarized) for steered
        pred_lang_label = [
            1 if classification["label"] == lang else 0
            for classification in lang_classifications
        ]
        if rating in ("5", "4"):
            pred_rating_label = [
                1 if "positive" in classification["label"].lower() else 0
                for classification in senti_classifications
            ]
        if rating in ("1", "2"):
            pred_rating_label = [
                1 if "negative" in classification["label"].lower() else 0
                for classification in senti_classifications
            ]

        # Predicted labels for unsteered
        pred_unsteered_lang_label = [
            1 if classification["label"] == lang else 0
            for classification in unsteered_lang_classifications
        ]
        if rating in ("5", "4"):
            pred_unsteered_rating_label = [
                1 if "positive" in classification["label"].lower() else 0
                for classification in unsteered_senti_classifications
            ]
        if rating in ("1", "2"):
            pred_unsteered_rating_label = [
                1 if "negative" in classification["label"].lower() else 0
                for classification in unsteered_senti_classifications
            ]

        # Calculate classification metrics for steered
        lang_metrics = calculate_metrics(true_lang_label, pred_lang_label)
        rating_metrics = calculate_metrics(true_rating_label, pred_rating_label)

        # Calculate classification metrics for unsteered
        unsteered_lang_metrics = calculate_metrics(
            true_lang_label, pred_unsteered_lang_label
        )
        unsteered_rating_metrics = calculate_metrics(
            true_rating_label, pred_unsteered_rating_label
        )

        # Mean similarity scores
        avg_similarity_score = np.mean(topic_similarity_scores)
        std_conditioned_similarity_score = np.std(topic_similarity_scores)
        avg_unconditioned_similarity_score = np.mean(unconditioned_similarity_scores)
        std_unconditioned_similarity_score = np.std(unconditioned_similarity_scores)
        # Append steered metrics
        steered_results.append(
            {
                "state": state,
                "accuracy_lang": lang_metrics["accuracy"],
                "precision_lang": lang_metrics["precision"],
                "recall_lang": lang_metrics["recall"],
                "f1_lang": lang_metrics["f1"],
                "accuracy_rating": rating_metrics["accuracy"],
                "precision_rating": rating_metrics["precision"],
                "recall_rating": rating_metrics["recall"],
                "f1_rating": rating_metrics["f1"],
            }
        )

        # Append unsteered metrics
        unsteered_results.append(
            {
                "state": state,
                "accuracy_lang": unsteered_lang_metrics["accuracy"],
                "precision_lang": unsteered_lang_metrics["precision"],
                "recall_lang": unsteered_lang_metrics["recall"],
                "f1_lang": unsteered_lang_metrics["f1"],
                "accuracy_rating": unsteered_rating_metrics["accuracy"],
                "precision_rating": unsteered_rating_metrics["precision"],
                "recall_rating": unsteered_rating_metrics["recall"],
                "f1_rating": unsteered_rating_metrics["f1"],
            }
        )

        # Append similarity scores for both steered and unsteered
        similarity_results.append(
            {
                "state": state,
                "avg_similarity_score": avg_similarity_score,
                "std_conditioned_similarity_score": std_conditioned_similarity_score,
                "avg_unconditioned_similarity_score": avg_unconditioned_similarity_score,
                "std_unconditioned_similarity_score": std_unconditioned_similarity_score,
                # "avg_sentiment_score_steered": np.mean([classification["score"] for classification in senti_classifications]),
                # "avg_lang_score_steered": np.mean([classification["score"] for classification in lang_classifications]),
                # "avg_sentiment_score_unsteered": np.mean([classification["score"] for classification in unsteered_senti_classifications]),
                # "avg_lang_score_unsteered": np.mean([classification["score"] for classification in unsteered_lang_classifications]),
            }
        )

    # Convert results to DataFrames and save
    steered_df = pd.DataFrame(steered_results)
    unsteered_df = pd.DataFrame(unsteered_results)
    similarity_df = pd.DataFrame(similarity_results)

    steered_df.to_csv("steered_evaluation_results.csv", index=False)
    unsteered_df.to_csv("unsteered_evaluation_results.csv", index=False)
    similarity_df.to_csv("similarity_evaluation_results.csv", index=False)


def plot_roc_curves(true_labels, predicted_scores, title, plot_type):
    # Create ROC plot for each target label in a one-vs-rest fashion
    fig, ax = plt.subplots(figsize=(10, 8))
    true_labels = np.array(true_labels)
    for label, scores in predicted_scores.items():
        fpr, tpr, _ = roc_curve(
            true_labels[:, list(predicted_scores.keys()).index(label)], scores
        )
        auc = roc_auc_score(
            true_labels[:, list(predicted_scores.keys()).index(label)], scores
        )
        ax.plot(fpr, tpr, label=f"{label.capitalize()} (AUC = {auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.savefig(f"resources/evaluation_results/multi_state/AUC_{plot_type}_all.png")
    plt.close()


if __name__ == "__main__":
    print("Loading checkpoint...\n")
    checkpoint = "./SLiM/resources/checkpoints/SLiM_multi_state_wo_500_49.pth"
    model, tokenizer = load_model_and_tokenizer(checkpoint, NUM_STATES)
    device = torch.device(config["device"])
    model = model.to(device)
    # Load the average topic embedding
    avg_topic_embedding = get_average_topic_embedding()
    # ll the function with combined states
    combined_states = [
        "book_en_1",
        "book_en_5",
        "electronics_en_1",
        "electronics_en_5",
        "clothing_en_1",
        "clothing_en_5",
        "beauty_en_1",
        "beauty_en_5",
        "book_de_1",
        "book_de_5",
        "electronics_de_1",
        "electronics_de_5",
        "clothing_de_1",
        "clothing_de_5",
        "beauty_de_1",
        "beauty_de_5",
        "book_en_2",
        "book_en_4",
        "electronics_en_2",
        "electronics_en_4",
        "clothing_en_2",
        "clothing_en_4",
        "beauty_en_2",
        "beauty_en_4",
        "book_de_2",
        "book_de_4",
        "electronics_de_2",
        "electronics_de_4",
        "clothing_de_2",
        "clothing_de_4",
        "beauty_de_2",
        "beauty_de_4",
    ]
    evaluate_combined_states_with_metrics(
        model, tokenizer, combined_states, avg_topic_embedding
    )
