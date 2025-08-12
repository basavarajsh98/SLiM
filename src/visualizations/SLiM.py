# viosualize scale and shift and gamma param via histogram

import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from config.config import get_config
from src.dataset import SLiMed_Dataset, collate_fn
from src.inference import load_model_and_tokenizer

warnings.filterwarnings("ignore")

config = get_config()


checkpoint = "./resources/checkpoints/emotion_steering/emotions.pth"
model, tokenizer = load_model_and_tokenizer(checkpoint, num_states=5)
from src.experiments.emotion_steering import prepare_dataset

samples, tokenizer = prepare_dataset()
dataset = SLiMed_Dataset(samples)
test_loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)


# Initialize SLiM_values_by_state to store FiLM values by unique state
SLiM_values_by_state = defaultdict(
    lambda: {
        "gamma": [[] for _ in model.apply_film_at_layers],
        "scale": [[] for _ in model.apply_film_at_layers],
        "shift": [[] for _ in model.apply_film_at_layers],
    }
)
# Dictionary to store hidden states by state and layer
hidden_states_by_state = defaultdict(
    lambda: {
        "unmodulated": [[] for _ in model.apply_film_at_layers],
        "modulated": [[] for _ in model.apply_film_at_layers],
    }
)

# Capture FiLM parameters based on unique state inputs
for batch in test_loader:
    input_ids, attention_mask, state_tensor = batch
    model.reset_SLiM_values()
    model.reset_record_hidden_states()
    model(input_ids=input_ids, state_tensor=state_tensor, attention_mask=attention_mask)
    SLiM_values = model.SLiM_values
    # Organize by unique state for each batch
    batch_size = state_tensor.size(0)  # Get batch size
    for i in range(batch_size):
        # Convert each one-hot encoded state to a tuple to use as a unique key
        state_key = tuple(state_tensor[i].cpu().numpy().astype(float))
        # Append each FiLM parameter for this specific layer and state
        for key in SLiM_values:
            for layer_idx in range(len(SLiM_values[key])):
                # Ensure we handle indexing correctly
                if i < len(SLiM_values[key][layer_idx]):
                    SLiM_values_by_state[state_key][key][layer_idx].append(
                        SLiM_values[key][layer_idx][i]
                    )

        # Save hidden states separately for each unique state in the first layer only
        for layer_idx in model.apply_film_at_layers:
            hidden_states_by_state[state_key]["unmodulated"][layer_idx].append(
                model.hidden_states["unmodulated"][layer_idx][i]
            )
            hidden_states_by_state[state_key]["modulated"][layer_idx].append(
                model.hidden_states["modulated"][layer_idx][i]
            )
# Convert lists to numpy arrays for easier plotting
for state_key, values in SLiM_values_by_state.items():
    for key in values:
        for layer_idx in range(len(values[key])):
            # Concatenate to form arrays for each state and layer
            if values[key][layer_idx]:  # Check if list is non-empty
                values[key][layer_idx] = np.concatenate(values[key][layer_idx], axis=0)


# Initialize scalers for each layer to normalize unmodulated and modulated states separately
scalers = {
    layer_idx: {"unmodulated": MinMaxScaler(), "modulated": MinMaxScaler()}
    for layer_idx in model.apply_film_at_layers
}

# Plot histograms for each unique state
for state_key, values in SLiM_values_by_state.items():
    fig, axs = plt.subplots(
        len(model.apply_film_at_layers),
        2,
        figsize=(20, 2 * len(model.apply_film_at_layers)),
    )
    # fig.suptitle(f'FiLM Parameters and Normalized Hidden States for State: {state_key}')

    for i, layer_idx in enumerate(model.apply_film_at_layers):
        # FiLM parameter histograms
        # if values["gamma"][layer_idx].size > 0:
        #     axs[i, 0].hist(values["gamma"][layer_idx].flatten(), bins=50, color='skyblue', density=True)
        #     axs[i, 0].set_title(f'Gamma Layer {layer_idx}')
        #     axs[i, 0].set_xlim(0, 1)

        if values["scale"][layer_idx].size > 0:
            axs[i, 0].hist(
                values["scale"][layer_idx][0, :, :].flatten(),
                bins=50,
                color="salmon",
                density=True,
            )
            axs[i, 0].set_title(f"Scale Layer {layer_idx}")
            axs[i, 0].set_xlim(-1, 1)

        if values["shift"][layer_idx].size > 0:
            axs[i, 1].hist(
                values["shift"][layer_idx][0, :, :].flatten(),
                bins=50,
                color="lightgreen",
                density=True,
            )
            axs[i, 1].set_title(f"Shift Layer {layer_idx}")
            axs[i, 1].set_xlim(-1, 1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"normalized_hidden_states_comparison_state_{state_key}.png")
    plt.show()

# Scatter plot for each layer
# Initialize dictionaries to store accumulated values for averaging
avg_unmodulated_by_layer = {}
avg_modulated_by_layer = {}

# Plot scatter of averaged unmodulated vs. modulated states for first layer by state
for state_key, state_hidden_states in hidden_states_by_state.items():
    if state_key == (np.int64(1), np.int64(0), np.int64(0), np.int64(0), np.int64(0)):
        for layer_idx in model.apply_film_at_layers:
            # Separate handling for layer 0 (first layer), where we scatter-plot each state individually
            for layer_idx in model.apply_film_at_layers:
                unmodulated_state = np.concat(
                    state_hidden_states["unmodulated"][layer_idx], axis=0
                )
                modulated_state = np.concat(
                    state_hidden_states["modulated"][layer_idx], axis=0
                )

                # Flatten for normalization
                unmodulated_state_2d = unmodulated_state.reshape(
                    -1, unmodulated_state.shape[-1]
                )
                modulated_state_2d = modulated_state.reshape(
                    -1, modulated_state.shape[-1]
                )

                # Normalize hidden states for the specific state
                unmodulated_normalized = scalers[layer_idx][
                    "unmodulated"
                ].fit_transform(unmodulated_state_2d)
                modulated_normalized = scalers[layer_idx]["modulated"].fit_transform(
                    modulated_state_2d
                )

                # Calculate averages across dimensions
                avg_unmodulated = np.mean(unmodulated_normalized, axis=0)
                avg_modulated = np.mean(modulated_normalized, axis=0)

                # Scatter plot for this state and layer 0
                plt.figure(figsize=(8, 8))
                plt.scatter(avg_unmodulated, avg_modulated, alpha=0.5, color="purple")
                plt.xlabel("Average Unmodulated Hidden State")
                plt.ylabel("Average Modulated Hidden State")
                plt.title(
                    f"Averaged Modulated vs. Unmodulated Hidden States for Layer {layer_idx}, State {state_key}"
                )
                plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
                plt.grid(True)
                plt.axis("equal")
                plt.tight_layout()
                plt.savefig(
                    f"scatter_avg_modulated_vs_unmodulated_layer_{layer_idx}_state_{state_key}.png"
                )
                plt.show()
