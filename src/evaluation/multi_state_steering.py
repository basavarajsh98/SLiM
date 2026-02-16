import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import pipeline

from config.config import get_config
from src.inference import generate_text, get_state_tensor, load_model_and_tokenizer

warnings.filterwarnings("ignore")
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics.pairwise import cosine_similarity

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

    if os.path.exists("./average.embedding.pth"):
        return torch.load("./average.embedding.pth")
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


# Evaluation function with similarity calculations
def evaluate_generations_with_similarity(
    model, tokenizer, states, topic_embeddings, num_generations=100
):
    similarity_scores_with_state = {state.split("_")[0]: [] for state in states}
    similarity_scores_without_state = {state.split("_")[0]: [] for state in states}

    for state in states:
        state_tensor = get_state_tensor(STATE_MAPPING, state, device)
        state = state.split("_")[0]
        # Generate texts with state tensor
        generated_texts_with_state = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt="I feel",
            state_tensor=state_tensor,
            num_generations=num_generations,
        )
        for gen_text in generated_texts_with_state:
            similarity_score = compute_similarity(gen_text, topic_embeddings[state])
            similarity_scores_with_state[state].append(similarity_score)

        # Generate texts without state tensor
        generated_texts_without_state = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt="I feel",
            state_tensor=None,
            num_generations=num_generations,
        )
        for gen_text in generated_texts_without_state:
            similarity_score = compute_similarity(gen_text, topic_embeddings[state])
            similarity_scores_without_state[state].append(similarity_score)
        print(f"Completed evaluating {state}")

    # Create DataFrames for plotting
    similarity_df_with_state = pd.DataFrame(
        [
            {"State": state, "Similarity Score": score, "Condition": "With State"}
            for state, scores in similarity_scores_with_state.items()
            for score in scores
        ]
    )

    similarity_df_without_state = pd.DataFrame(
        [
            {"State": state, "Similarity Score": score, "Condition": "Without State"}
            for state, scores in similarity_scores_without_state.items()
            for score in scores
        ]
    )

    # Combine the two DataFrames
    combined_similarity_df = pd.concat(
        [similarity_df_with_state, similarity_df_without_state]
    )

    # Calculate mean and standard deviation for each condition
    mean_scores = (
        combined_similarity_df.groupby(["State", "Condition"])
        .agg(["mean", "std"])
        .reset_index()
    )

    # Plot using a bar plot with error bars
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=mean_scores,
        x="State",
        y=("Similarity Score", "mean"),
        hue="Condition",
        palette="Set2",
        ci=None,
    )

    # Add error bars
    for i in range(len(mean_scores)):
        plt.errorbar(
            x=i // 2,
            y=mean_scores[("Similarity Score", "mean")][i],
            yerr=mean_scores[("Similarity Score", "std")][i],
            fmt="none",
            c="black",
            capsize=5,
        )

    plt.title("Average Relevance per Topic: With vs Without Steering")
    plt.ylabel("Mean Relevance Score")
    plt.xlabel("Topic")
    plt.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()  # Adjust layout to make room for the legend
    plt.savefig("resources/evaluation_results/multi_state/topic_plot.png")


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
        device=config["device"],
    )
    return classifier


senti_classifier = get_distilroberta_senti_classifier()
# Updated evaluation functions to accumulate data and plot all ROC curves on one plot for each type


def evaluate_lang_generations_with_auc(
    model, tokenizer, state_pairs, num_generations=100
):
    # Initialize lists to store all true labels and predicted scores
    all_true_labels = []
    all_predicted_scores = {
        state.split("_")[-1]: [] for state in [s for pair in state_pairs for s in pair]
    }

    for states in state_pairs:
        for state in states:
            state_tensor = get_state_tensor(STATE_MAPPING, state, device)
            generated_texts = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt="I feel",
                state_tensor=state_tensor,
                num_generations=num_generations,
            )
            classifications = lang_classifier(generated_texts)

            for classification in classifications:
                true_label = [
                    1 if state.split("_")[-1] == target_state.split("_")[-1] else 0
                    for target_state in states
                ]
                all_true_labels.append(true_label)

                for target_state in states:
                    target_label = target_state.split("_")[-1]
                    score = (
                        classification["score"]
                        if classification["label"] == target_label
                        else 0
                    )
                    all_predicted_scores[target_label].append(score)
            # Calculate AUC for the current state
            auc = roc_auc_score(
                true_label,
                [
                    all_predicted_scores[state.split("_")[-1]][i]
                    for i in range(len(true_label))
                ],
            )
            # state_aucs[state] = auc
        # Plot all ROC curves for lang in a single figure
        # plot_roc_curves(all_true_labels, all_predicted_scores, "Language Detection ROC Curves", f"lang_{states[0].split('_')[0]}")
    plot_roc_curves(
        all_true_labels, all_predicted_scores, "Language Detection ROC Curves", "lang"
    )


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
    plt.savefig("lang.png")
    plt.close()


def evaluate_senti_generations_with_auc(
    model, tokenizer, state_pairs, num_generations=100
):
    # Initialize dictionary to store metrics per topic and sentiment type
    metrics_by_topic_sentiment = {}

    for states in state_pairs:
        all_true_labels = []
        all_predicted_scores = {state: [] for state in states}
        all_predicted_labels = {state: [] for state in states}

        for state in states:
            # Separate topic and sentiment information from the state name
            topic, sentiment_label = state.split("_")[0], "_".join(state.split("_")[1:])

            # Get the state tensor and generate texts
            state_tensor = get_state_tensor(STATE_MAPPING, state, device)
            generated_texts = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt="I feel",
                state_tensor=state_tensor,
                num_generations=num_generations,
            )
            classifications = senti_classifier(generated_texts)

            # Track true labels and predicted scores
            true_label = [1 if state == target else 0 for target in states]
            for classification in classifications:
                # Add true labels
                all_true_labels.append(true_label)

                # Track prediction score for the target sentiment
                for target in states:
                    target_label = "_".join(target.split("_")[1:])
                    score = (
                        classification["score"]
                        if classification["label"].lower() == target_label.split("_")[0]
                        else 0
                    )
                    all_predicted_scores[target].append(score)
                    predicted_label = (
                        1
                        if classification["label"].lower() == target_label.split("_")[0]
                        else 0
                    )
                    all_predicted_labels[target].append(predicted_label)

        # Calculate metrics for each sentiment type in this pair
        for state in states:
            topic, sentiment_type = state.split("_")[0], "_".join(state.split("_")[1:])
            idx = states.index(state)
            true_labels = np.array(all_true_labels)

            # Calculate metrics for this topic and sentiment
            accuracy = accuracy_score(true_labels[:, idx], all_predicted_labels[state])
            f1 = f1_score(true_labels[:, idx], all_predicted_labels[state])
            auc = roc_auc_score(true_labels[:, idx], all_predicted_scores[state])

            # Initialize the structure for storing metrics if not already done
            if topic not in metrics_by_topic_sentiment:
                metrics_by_topic_sentiment[topic] = {}
            if sentiment_type not in metrics_by_topic_sentiment[topic]:
                metrics_by_topic_sentiment[topic][sentiment_type] = []

            # Append metrics to the list for this topic and sentiment
            metrics_by_topic_sentiment[topic][sentiment_type].append(
                {"accuracy": accuracy, "f1": f1, "auc": auc}
            )

    return metrics_by_topic_sentiment


def aggregate_topic_sentiment_metrics(
    all_senti_states, model, tokenizer, num_generations=100
):
    # Evaluate and collect metrics for each topic and sentiment type
    all_metrics = evaluate_senti_generations_with_auc(
        model, tokenizer, all_senti_states, num_generations
    )

    # Compute averages and standard deviations for each topic and sentiment combination
    averaged_results = {
        topic: {
            sentiment: {
                "accuracy_avg": np.mean([m["accuracy"] for m in metrics]),
                "accuracy_std": np.std([m["accuracy"] for m in metrics]),
                "f1_avg": np.mean([m["f1"] for m in metrics]),
                "f1_std": np.std([m["f1"] for m in metrics]),
                "auc_avg": np.mean([m["auc"] for m in metrics]),
            }
            for sentiment, metrics in sentiment_metrics.items()
        }
        for topic, sentiment_metrics in all_metrics.items()
    }

    return averaged_results


def unsteered_evaluate_senti_generations_with_auc(
    model, tokenizer, state_pairs, num_generations=100
):
    # Initialize dictionary to store metrics per topic and sentiment type
    metrics_by_topic_sentiment = {}

    for states in state_pairs:
        all_true_labels = []
        all_predicted_scores = {state: [] for state in states}
        all_predicted_labels = {state: [] for state in states}

        for state in states:
            # Separate topic and sentiment information from the state name
            topic, sentiment_label = state.split("_")[0], "_".join(state.split("_")[1:])

            # Get the state tensor and generate texts
            state_tensor = get_state_tensor(STATE_MAPPING, state, device)
            generated_texts = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt="I feel",
                state_tensor=None,
                num_generations=num_generations,
            )
            classifications = senti_classifier(generated_texts)

            # Track true labels and predicted scores
            true_label = [1 if state == target else 0 for target in states]
            for classification in classifications:
                # Add true labels
                all_true_labels.append(true_label)

                # Track prediction score for the target sentiment
                for target in states:
                    target_label = "_".join(target.split("_")[1:])
                    score = (
                        classification["score"]
                        if classification["label"].lower() == target_label.split("_")[0]
                        else 0
                    )
                    all_predicted_scores[target].append(score)
                    predicted_label = (
                        1
                        if classification["label"].lower() == target_label.split("_")[0]
                        else 0
                    )
                    all_predicted_labels[target].append(predicted_label)

        # Calculate metrics for each sentiment type in this pair
        for state in states:
            topic, sentiment_type = state.split("_")[0], "_".join(state.split("_")[1:])
            idx = states.index(state)
            true_labels = np.array(all_true_labels)

            # Calculate metrics for this topic and sentiment
            accuracy = accuracy_score(true_labels[:, idx], all_predicted_labels[state])
            f1 = f1_score(true_labels[:, idx], all_predicted_labels[state])
            auc = roc_auc_score(true_labels[:, idx], all_predicted_scores[state])

            # Initialize the structure for storing metrics if not already done
            if topic not in metrics_by_topic_sentiment:
                metrics_by_topic_sentiment[topic] = {}
            if sentiment_type not in metrics_by_topic_sentiment[topic]:
                metrics_by_topic_sentiment[topic][sentiment_type] = []

            # Append metrics to the list for this topic and sentiment
            metrics_by_topic_sentiment[topic][sentiment_type].append(
                {"accuracy": accuracy, "f1": f1, "auc": auc}
            )

    return metrics_by_topic_sentiment


def unsteered_aggregate_topic_sentiment_metrics(
    all_senti_states, model, tokenizer, num_generations=100
):
    # Evaluate and collect metrics for each topic and sentiment type
    all_metrics = unsteered_evaluate_senti_generations_with_auc(
        model, tokenizer, all_senti_states, num_generations
    )

    # Compute averages and standard deviations for each topic and sentiment combination
    averaged_results = {
        topic: {
            sentiment: {
                "accuracy_avg": np.mean([m["accuracy"] for m in metrics]),
                "accuracy_std": np.std([m["accuracy"] for m in metrics]),
                "f1_avg": np.mean([m["f1"] for m in metrics]),
                "f1_std": np.std([m["f1"] for m in metrics]),
                "auc_avg": np.mean([m["auc"] for m in metrics]),
            }
            for sentiment, metrics in sentiment_metrics.items()
        }
        for topic, sentiment_metrics in all_metrics.items()
    }

    return averaged_results


def evaluate_and_aggregate_metrics(
    model, tokenizer, all_senti_states, num_generations=100, num_runs=5
):
    """Evaluate and aggregate metrics for both steered and unsteered generations over multiple runs."""

    def evaluate_generations(model, tokenizer, states, is_steered, num_generations):
        """Evaluate model for a single run."""
        metrics_by_topic_sentiment = {}
        for state_group in states:
            all_true_labels = []
            all_predicted_scores = {state: [] for state in state_group}
            all_predicted_labels = {state: [] for state in state_group}

            for state in state_group:
                # Separate topic and sentiment from the state
                topic, sentiment_label = (
                    state.split("_")[0],
                    "_".join(state.split("_")[1:]),
                )

                # Generate texts
                state_tensor = (
                    get_state_tensor(STATE_MAPPING, state, device)
                    if is_steered
                    else None
                )
                generated_texts = generate_text(
                    model,
                    tokenizer,
                    prompt="I feel",
                    state_tensor=state_tensor,
                    num_generations=num_generations,
                )
                classifications = senti_classifier(generated_texts)

                # Track true labels and predictions
                true_label = [1 if state == target else 0 for target in state_group]
                for classification in classifications:
                    all_true_labels.append(true_label)

                    for target in state_group:
                        target_label = "_".join(target.split("_")[1:])
                        score = (
                            classification["score"]
                            if classification["label"].lower()
                            == target_label.split("_")[0]
                            else 0
                        )
                        all_predicted_scores[target].append(score)
                        predicted_label = (
                            1
                            if classification["label"].lower()
                            == target_label.split("_")[0]
                            else 0
                        )
                        all_predicted_labels[target].append(predicted_label)

            # Compute metrics for each state
            for state in state_group:
                topic, sentiment_type = (
                    state.split("_")[0],
                    "_".join(state.split("_")[1:]),
                )
                idx = state_group.index(state)
                true_labels = np.array(all_true_labels)

                # Calculate metrics
                accuracy = accuracy_score(
                    true_labels[:, idx], all_predicted_labels[state]
                )
                f1 = f1_score(true_labels[:, idx], all_predicted_labels[state])
                auc = roc_auc_score(true_labels[:, idx], all_predicted_scores[state])

                # Organize metrics
                if topic not in metrics_by_topic_sentiment:
                    metrics_by_topic_sentiment[topic] = {}
                if sentiment_type not in metrics_by_topic_sentiment[topic]:
                    metrics_by_topic_sentiment[topic][sentiment_type] = []

                metrics_by_topic_sentiment[topic][sentiment_type].append(
                    {"accuracy": accuracy, "f1": f1, "auc": auc}
                )

        return metrics_by_topic_sentiment

    def aggregate_metrics(metrics_by_topic_sentiment):
        """Aggregate metrics across runs."""
        return {
            topic: {
                sentiment: {
                    "accuracy_avg": np.mean([m["accuracy"] for m in metrics]),
                    "accuracy_std": np.std([m["accuracy"] for m in metrics]),
                    "f1_avg": np.mean([m["f1"] for m in metrics]),
                    "f1_std": np.std([m["f1"] for m in metrics]),
                    "auc_avg": np.mean([m["auc"] for m in metrics]),
                }
                for sentiment, metrics in sentiment_metrics.items()
            }
            for topic, sentiment_metrics in metrics_by_topic_sentiment.items()
        }

    # Collect metrics across runs
    steered_metrics_runs = []
    unsteered_metrics_runs = []
    for _ in range(num_runs):
        steered_metrics_runs.append(
            evaluate_generations(
                model,
                tokenizer,
                all_senti_states,
                is_steered=True,
                num_generations=num_generations,
            )
        )
        unsteered_metrics_runs.append(
            evaluate_generations(
                model,
                tokenizer,
                all_senti_states,
                is_steered=False,
                num_generations=num_generations,
            )
        )

    # Aggregate metrics across runs
    aggregated_steered = aggregate_metrics(steered_metrics_runs[0])
    aggregated_unsteered = aggregate_metrics(unsteered_metrics_runs[0])

    return aggregated_steered, aggregated_unsteered


if __name__ == "__main__":
    print("Loading checkpoint...\n")
    checkpoint = "./resources/checkpoints/multi_state_steering/multi_states.pth"
    model, tokenizer = load_model_and_tokenizer(checkpoint, NUM_STATES)
    device = torch.device(config["device"])
    model = model.to(device)
    # Load the average topic embedding
    avg_topic_embedding = get_average_topic_embedding()

    topic_states = ["book_en", "electronics_en", "clothing_en", "beauty_en"]
    print("Topic")
    # evaluate_generations_with_similarity(model, tokenizer, topic_states, avg_topic_embedding)

    # Language state groups for evaluation
    all_lang_states = [
        ["book_de", "book_en"],
        ["electronics_de", "electronics_en"],
        ["clothing_de", "clothing_en"],
        ["beauty_de", "beauty_en"],
    ]
    print("Lang")
    evaluate_lang_generations_with_auc(model, tokenizer, all_lang_states)

    # Sentiment state groups for evaluation
    all_senti_states = [
        ["book_negative_1", "book_positive_2"],
        ["electronics_negative_1", "electronics_positive_2"],
        ["clothing_negative_1", "clothing_positive_2"],
        ["beauty_negative_1", "beauty_positive_2"],
        ["book_negative_2", "book_positive_1"],
        ["electronics_negative_2", "electronics_positive_1"],
        ["clothing_negative_2", "clothing_positive_1"],
        ["beauty_negative_2", "beauty_positive_1"],
        ["book_negative_1", "book_positive_1"],
        ["electronics_negative_1", "electronics_positive_1"],
        ["clothing_negative_1", "clothing_positive_1"],
        ["beauty_negative_1", "beauty_positive_1"],
        ["book_negative_2", "book_positive_2"],
        ["electronics_negative_2", "electronics_positive_2"],
        ["clothing_negative_2", "clothing_positive_2"],
        ["beauty_negative_2", "beauty_positive_2"],
    ]
    print("Senti")
    # Aggregate sentiment metrics
    # Aggregate metrics by topic and sentiment
    averaged_results = aggregate_topic_sentiment_metrics(
        all_senti_states, model, tokenizer
    )

    print("Aggregated Results by Topic and Sentiment:")
    for topic, sentiments in averaged_results.items():
        print(f"\nTopic: {topic.capitalize()}")
        for senti, metrics in sentiments.items():
            print(
                f"  {senti.capitalize()}: Accuracy: {metrics['accuracy_avg']:.2f} +- {metrics['accuracy_std']:.2f}, F1: {metrics['f1_avg']:.2f} +- {metrics['f1_std']:.2f}, AUC: {metrics['auc_avg']:.2f}"
            )

    # Aggregate metrics by topic and sentiment
    unsteered_averaged_results = unsteered_aggregate_topic_sentiment_metrics(
        all_senti_states, model, tokenizer
    )

    print("Aggregated Results by Topic and Sentiment:")
    for topic, sentiments in unsteered_averaged_results.items():
        print(f"\nTopic: {topic.capitalize()}")
        for senti, metrics in sentiments.items():
            print(
                f"  {senti.capitalize()}: Accuracy: {metrics['accuracy_avg']:.2f} +- {metrics['accuracy_std']:.2f}, F1: {metrics['f1_avg']:.2f} +- {metrics['f1_std']:.2f}, AUC: {metrics['auc_avg']:.2f}"
            )

    aggregated_steered, aggregated_unsteered = evaluate_and_aggregate_metrics(
        model, tokenizer, all_senti_states
    )

    print("\nSteered Metrics:")
    for topic, sentiments in aggregated_steered.items():
        print(f"\nTopic: {topic.capitalize()}")
        for senti, metrics in sentiments.items():
            print(
                f"  {senti.capitalize()}: Accuracy: {metrics['accuracy_avg']:.2f} ± {metrics['accuracy_std']:.2f}, "
                f"F1: {metrics['f1_avg']:.2f} ± {metrics['f1_std']:.2f}, AUC: {metrics['auc_avg']:.2f}"
            )

    print("\nUnsteered Metrics:")
    for topic, sentiments in aggregated_unsteered.items():
        print(f"\nTopic: {topic.capitalize()}")
        for senti, metrics in sentiments.items():
            print(
                f"  {senti.capitalize()}: Accuracy: {metrics['accuracy_avg']:.2f} ± {metrics['accuracy_std']:.2f}, "
                f"F1: {metrics['f1_avg']:.2f} ± {metrics['f1_std']:.2f}, AUC: {metrics['auc_avg']:.2f}"
            )
