import torch
import torch.nn as nn
import torch.nn.functional as F


class StateBlock(nn.Module):
    def __init__(self, state_dim, target_dim, activation=nn.ReLU):
        super(StateBlock, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(state_dim, target_dim // 4),
            activation(),
            nn.Linear(target_dim // 4, target_dim // 2),
            activation(),
            nn.Linear(target_dim // 2, target_dim),
            nn.LayerNorm(target_dim),
        )

    def forward(self, state_vector):
        return self.projection(state_vector)


class SLiMedNet(nn.Module):
    def __init__(
        self,
        model,
        state_embed_dim,
        apply_SLiM_at_layers=None,
        record_SLiM=False,
        record_hidden_states=False,
    ):
        super(SLiMedNet, self).__init__()
        state_embed_dim = config.get("num_states")
        apply_SLiM_at_layers = config.get("apply_SLiM_at_layers")

        self.gpt2 = model

        self.state_proj = StateBlock(state_embed_dim, self.gpt2.config.n_embd)
        self.apply_SLiM_at_layers = (
            apply_SLiM_at_layers
            if apply_SLiM_at_layers
            else list(range(len(self.gpt2.transformer.h)))
        )
        self.gate = nn.ModuleList(
            [
                nn.Linear(state_embed_dim, 1)
                for _ in range(len(self.apply_SLiM_at_layers))
            ]
        )
        self.SLiM_scale = nn.ModuleList(
            [
                nn.Linear(self.gpt2.config.n_embd, self.gpt2.config.n_embd)
                for _ in range(len(self.apply_SLiM_at_layers))
            ]
        )
        self.SLiM_shift = nn.ModuleList(
            [
                nn.Linear(self.gpt2.config.n_embd, self.gpt2.config.n_embd)
                for _ in range(len(self.apply_SLiM_at_layers))
            ]
        )

        self.hooks = self.register_gpt_SLiM_hooks()
        self.record_SLiM = record_SLiM
        self.record_hidden_states = record_hidden_states

        if self.record_SLiM:
            self.SLiM_values = {
                "gamma": [[] for _ in range(len(self.apply_SLiM_at_layers))],
                "scale": [[] for _ in range(len(self.apply_SLiM_at_layers))],
                "shift": [[] for _ in range(len(self.apply_SLiM_at_layers))],
            }

        if self.record_hidden_states:
            self.hidden_states = {
                "unmodulated": [[] for _ in range(len(self.apply_SLiM_at_layers))],
                "modulated": [[] for _ in range(len(self.apply_SLiM_at_layers))],
            }

    def register_gpt_SLiM_hooks(self):
        hooks = []
        for i, idx in enumerate(self.apply_SLiM_at_layers):
            layer = self.gpt2.transformer.h[idx]
            hooks.append(layer.register_forward_hook(self.create_SLiM_hook(i)))
        return hooks

    def create_SLiM_hook(self, layer_idx):
        def hook(module, input, output):
            if self.current_state_embed is not None:
                projected_state = self.state_proj(self.current_state_embed)
                gate_value = torch.sigmoid(
                    self.gate[layer_idx](self.current_state_embed)
                )
                scale = torch.tanh(self.SLiM_scale[layer_idx](projected_state))
                shift = torch.tanh(self.SLiM_shift[layer_idx](projected_state))

                gate_value = torch.sigmoid(
                    self.gate[layer_idx](self.current_state_embed)
                )

                if self.record_SLiM:
                    self.SLiM_values["gamma"][layer_idx].append(
                        gate_value.detach().cpu().numpy()
                    )
                    self.SLiM_values["scale"][layer_idx].append(
                        scale.detach().cpu().numpy()
                    )
                    self.SLiM_values["shift"][layer_idx].append(
                        shift.detach().cpu().numpy()
                    )

                steered_output = (output[0] * scale + shift) * gate_value
                steered_output = F.layer_norm(steered_output, steered_output.shape[-1:])
                if self.record_hidden_states:
                    modulated_output = steered_output.detach().cpu().numpy()
                    self.hidden_states["modulated"][layer_idx].append(modulated_output)
                    unmodulated_output = output[0].detach().cpu().numpy()
                    self.hidden_states["unmodulated"][layer_idx].append(
                        unmodulated_output
                    )
                steered_output = steered_output + output[0]
                output = (steered_output,) + output[1:]
            return output

        return hook

    def forward(self, input_ids, state_tensor=None, attention_mask=None):
        if state_tensor is None:
            self.current_state_embed = None
        else:
            self.current_state_embed = state_tensor.unsqueeze(1)
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def generate(
        self, input_ids, state_tensor=None, attention_mask=None, **generate_kwargs
    ):
        if state_tensor is None:
            self.current_state_embed = None
        else:
            self.current_state_embed = state_tensor.unsqueeze(1)
        return self.gpt2.generate(
            input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs
        )

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def reset_SLiM_values(self):
        """
        Clears self.SLiM_values to ensure it starts fresh for each batch of states.
        """
        if self.record_SLiM:
            self.SLiM_values = {
                "gamma": [[] for _ in range(len(self.apply_SLiM_at_layers))],
                "scale": [[] for _ in range(len(self.apply_SLiM_at_layers))],
                "shift": [[] for _ in range(len(self.apply_SLiM_at_layers))],
            }

    def reset_record_hidden_states(self):
        """
        Clears self.SLiM_values to ensure it starts fresh for each batch of states.
        """
        if self.record_hidden_states:
            self.hidden_states = {
                "unmodulated": [[] for _ in range(len(self.apply_SLiM_at_layers))],
                "modulated": [[] for _ in range(len(self.apply_SLiM_at_layers))],
            }
