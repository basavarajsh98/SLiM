import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

from config.config import get_config
from src.inference import generate_text, get_state_tensor, load_model_and_tokenizer

warnings.filterwarnings("ignore")
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Define evaluation function
from sklearn.metrics.pairwise import cosine_similarity

config = get_config()


NUM_STATES = 10
STATE_MAPPING = "topic_mapping"


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

    if os.path.exists("./average.embedding_5k.pth"):
        return torch.load("./average.embedding_5k.pth")
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
    similarity_scores_with_state = {state: [] for state in states}
    similarity_scores_without_state = {state: [] for state in states}

    for state in states:
        state_tensor = get_state_tensor(STATE_MAPPING, state, device)
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

    # Display values on each bar
    for i in range(len(mean_scores)):
        plt.text(
            x=(i // 2) + 0.03,
            y=mean_scores[("Similarity Score", "mean")][i] + 0.01,
            s=f"{mean_scores[('Similarity Score', 'mean')][i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.title("Average Relevance per Topic: With vs Without Steering")
    plt.ylabel("Mean Relevance Score")
    plt.xlabel("Topic")
    plt.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()  # Adjust layout to make room for the legend
    plt.savefig("resources/evaluation_results/topic/topic_plot_26.11.png")

    # Calculate mean similarity scores for each state
    mean_similarity_with_state = {
        state: np.mean(scores) for state, scores in similarity_scores_with_state.items()
    }
    mean_similarity_without_state = {
        state: np.mean(scores)
        for state, scores in similarity_scores_without_state.items()
    }

    # Calculate relevance increase
    relevance_increase = {
        state: 100
        * (mean_similarity_with_state[state] - mean_similarity_without_state[state])
        / mean_similarity_without_state[state]
        for state in states
    }

    # Convert relevance increase to DataFrame for plotting
    relevance_increase_df = pd.DataFrame(
        {
            "State": list(relevance_increase.keys()),
            "Relevance Increase": list(relevance_increase.values()),
        }
    )

    # Plot the relevance increase
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=relevance_increase_df,
        x="State",
        y="Relevance Increase",
        palette="coolwarm",
    )
    plt.title("Percentage Increase in Relevance Score with State Tensor")
    plt.ylabel("Relevance Increase (%)")
    plt.xlabel("State")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("resources/evaluation_results/topic/relevance_increase_plot.png")


if __name__ == "__main__":
    print("Loading checkpoint...\n")
    checkpoint = "./resources/checkpoints/topic_steering/topics.pth"
    model, tokenizer = load_model_and_tokenizer(checkpoint, NUM_STATES)
    device = torch.device(config["device"])
    model = model.to(device)

    # Define state list and device
    # Define topics and load datasets

    states = [
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

    # Load the average topic embedding
    avg_topic_embedding = get_average_topic_embedding()

    # Run evaluation
    evaluate_generations_with_similarity(model, tokenizer, states, avg_topic_embedding)
