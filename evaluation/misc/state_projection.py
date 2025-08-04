import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Function to collect projected states
def collect_projected_states(model, state_inputs):
    """
    Pass each state input through the model's state projection layer 
    and store the projected state.
    """
    model.eval()  # Set model to evaluation mode
    projected_states = []
    
    with torch.no_grad():
        for state in state_inputs:
            # Ensure the state tensor has the right shape
            state_tensor = torch.tensor(state).unsqueeze(0)  # Assuming state is a 1D tensor
            projected_state = model.state_proj(state_tensor)  # Apply projection
            projected_states.append(projected_state.squeeze().cpu().numpy())

    return projected_states

# Collect projected states for visualization
state_inputs = [
    [1, 0],  # Example state input 1 (one-hot encoded)
    [0, 1],  # Example state input 2
    # Add more states as needed
]
projected_states = collect_projected_states(model, state_inputs)

# Perform t-SNE dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
reduced_states = tsne.fit_transform(projected_states)

# Plot the t-SNE result
plt.figure(figsize=(8, 6))
for idx, (x, y) in enumerate(reduced_states):
    plt.scatter(x, y, label=f'State {idx}')
plt.legend()
plt.title('t-SNE of Projected State Embeddings')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()
