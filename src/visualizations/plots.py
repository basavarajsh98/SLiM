import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from config.config import get_config
from src.inference import generate_text, get_state_tensor, load_model_and_tokenizer

warnings.filterwarnings("ignore")

config = get_config()


def compute_perplexicity(text, model, tokenizer, state_tensor=None):
    """
    Calculate log probabilities of each token in the input text with optional state_tensor.
    """
    from torch.nn import CrossEntropyLoss

    loss_fct = CrossEntropyLoss(reduction="none")
    encodings = tokenizer(
        text[:100],
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=100,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)
    encoded_texts = encodings["input_ids"]
    attn_mask = encodings["attention_mask"]
    labels = encoded_texts

    with torch.no_grad():
        out_logits = model(
            encoded_texts, attention_mask=attn_mask, state_tensor=state_tensor
        )

    shift_logits = out_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

    perplexity_batch = torch.exp(
        (
            loss_fct(shift_logits.transpose(1, 2), shift_labels)
            * shift_attention_mask_batch
        ).sum(1)
        / shift_attention_mask_batch.sum(1)
    )

    ppl = perplexity_batch.item()

    return ppl


def generate_and_calculate_perplexity_ratio(
    model, tokenizer, prompt, states, num_generations=100
):
    """
    Generate text with and without emotion state input and calculate the perplexity ratio.
    """
    ratios = {}

    for state in states:
        # Get state tensor if applicable
        state_tensor = get_state_tensor(state)

        # Generate text with emotion steering vector
        generated_texts_with_state = generate_text(
            model,
            tokenizer,
            prompt,
            state_tensor=state_tensor,
            num_generations=num_generations,
        )

        # Generate text without emotion steering vector
        generated_texts_without_state = generate_text(
            model, tokenizer, prompt, state_tensor=None, num_generations=num_generations
        )

        # Calculate perplexities for each case by feeding back into the model
        perplexities_with_state = [
            compute_perplexicity(text, model, tokenizer, state_tensor=state_tensor)
            for text in generated_texts_with_state
        ]
        perplexities_without_state = [
            compute_perplexicity(text, model, tokenizer, state_tensor=None)
            for text in generated_texts_without_state
        ]

        # Calculate the average perplexity and the ratio
        avg_perplexity_with_state = np.mean(perplexities_with_state)
        avg_perplexity_without_state = np.mean(perplexities_without_state)
        ratio = avg_perplexity_with_state / avg_perplexity_without_state

        ratios[state] = ratio
        print(
            f"Completed evaluating {state}: with: {avg_perplexity_with_state} without {avg_perplexity_without_state}\n"
        )
    return ratios


def plot_perplexity_ratios(ratios):
    """
    Plot the perplexity ratios for each state.
    """
    relevance_increase_df = pd.DataFrame(
        {"State": list(ratios.keys()), "Perplexity Ratio": list(ratios.values())}
    )
    # Plot the relevance increase
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=relevance_increase_df, x="State", y="Perplexity Ratio", palette="pastel"
    )

    # Annotate bars with their values
    for index, row in relevance_increase_df.iterrows():
        plt.text(
            x=index,
            y=row["Perplexity Ratio"]
            + 0.02,  # Adjust the y-position slightly above the bar
            s=f"{row['Perplexity Ratio']:.2f}",
            ha="center",  # Center the text horizontally
            va="bottom",  # Align the text at the bottom
            fontsize=10,  # Set font size
            color="black",  # Set text color
        )

    # Add labels and title
    plt.xlabel("Emotion State")
    plt.ylabel("Average Perplexity Ratio")
    # plt.title('Perplexity Ratio for Emotion States')
    # plt.xticks(rotation=70)
    plt.xticks(rotation=45)
    # Add a baseline line at ratio=1
    plt.axhline(1, color="red", linestyle="--", label="Baseline (1)")
    # plt.ylim(0, max(values) + 0.2)  # Adjust the y-axis limit for better visualization
    plt.legend()
    plt.tight_layout()

    # Display the plot
    plt.savefig(
        "resources/evaluation_results/perplexity_ratio/perplexity_ratio_emo.png"
    )


# Example usage in your KL divergence function
def calculate_kl_divergence(logits_with_state, logits_without_state):
    # min_length = min(logits_with_state.size(1), logits_without_state.size(1))
    # logits_with_state = logits_with_state[:, :min_length, :]
    # logits_without_state = logits_without_state[:, :min_length, :]
    # Calculate log softmax (log probabilities)
    log_probs_with_state = F.log_softmax(logits_with_state, dim=-1).squeeze(0)
    log_probs_without_state = F.log_softmax(logits_without_state, dim=-1).squeeze(0)
    kl_divergence = F.kl_div(
        log_probs_with_state,
        log_probs_without_state,
        reduction="batchmean",
        log_target=True,
    )
    return kl_divergence.item()


def evaluate_kl_divergence(model, tokenizer, prompt, states, num_generations=100):
    kl_divergences = {}
    kl_divergences_per_state = {state: [] for state in states}

    for state in states:
        state_tensor = get_state_tensor(state)
        kl_divergence_values = []

        for _ in range(num_generations):
            # Generate text with and without state tensor
            generated_text_with_state = generate_text(
                model, tokenizer, prompt, state_tensor=state_tensor, num_generations=1
            )[0]
            inputs = tokenizer(
                generated_text_with_state,
                return_tensors="pt",
            ).to(device)

            # Get log probabilities for each token in both generations
            logits_with_state = model(**inputs, state_tensor=state_tensor)
            logits_without_state = model(**inputs, state_tensor=None)

            # Calculate KL divergence for each token in the generated text
            kl_div = calculate_kl_divergence(logits_with_state, logits_without_state)
            kl_divergence_values.append(kl_div)
            kl_divergences_per_state[state].append(kl_div)

        # Average KL divergence for the state
        kl_divergences[state] = sum(kl_divergence_values) / len(kl_divergence_values)

    return kl_divergences, kl_divergences_per_state


# Example usage
if __name__ == "__main__":
    prompt = " I feel"
    # states =[
    #     'gourmet-food', 'video-game',
    #      'clothing', 'beauty',
    #     'arts', 'book', 'jewelry', 'shoe',
    #     'musical-instrument', 'electronics'
    #     ]
    states = ["anger", "fear", "joy", "sadness", "love"]

    checkpoint = "./resources/checkpoints/emotion_steering/emotions.pth"
    model, tokenizer = load_model_and_tokenizer(checkpoint, NUM_STATES)
    base_model = AutoModelForCausalLM.from_pretrained(config.get('base_model')).to("cuda")
    device = torch.device(config["device"])
    model = model.to(device)
    # print("Calculating perplexity ratios...")
    ratios = generate_and_calculate_perplexity_ratio(model, tokenizer, prompt, states)

    # Plot the perplexity ratios
    plot_perplexity_ratios(ratios)
    # # print("Plot saved as 'perplexity_ratio.png'")
    kl_divergences, kl_divergences_per_state = evaluate_kl_divergence(
        model, tokenizer, prompt, states
    )
    # print("KL Divergence values per state:", kl_divergences)

    # Prepare data in a long format for Tukey’s HSD
    kl_data = []
    for state, values in kl_divergences_per_state.items():
        kl_data.extend([(state, val) for val in values])

    # Convert to DataFrame
    kl_df = pd.DataFrame(kl_data, columns=["State", "KL Divergence"])

    # Plot violin plot for KL Divergence distributions by state
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="State", y="KL Divergence", data=kl_df, palette="muted")
    # plt.title("KL Divergence Distribution Across States")
    # Annotate mean KL divergence for each state
    state_means = kl_df.groupby("State")["KL Divergence"].mean()
    for index, (state, mean_kl) in enumerate(state_means.items()):
        plt.text(
            x=index + 0.03,
            y=mean_kl + 0.02,  # Slightly above the mean value
            s=f"{mean_kl:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
        )
    plt.xlabel("State")
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.2)  # Increase space at the bottom if needed
    plt.ylabel("KL-divergence (with SLiM || without SLiM)")
    plt.savefig("KL_Divergence_Violin_Plot.png")

    # Plot density plot for each state's KL divergence
    plt.figure(figsize=(10, 6))

    for state in kl_divergences_per_state:
        sns.kdeplot(kl_divergences_per_state[state], label=state, fill=True, alpha=0.5)

    # plt.title("Density Plot of KL Divergence Across States")
    plt.xlabel("KL-divergence (with SLiM || without SLiM)")
    plt.tight_layout()
    plt.ylabel("Density")
    plt.legend(title="State")
    plt.savefig("KL_Divergence_Density_Plot.png")
