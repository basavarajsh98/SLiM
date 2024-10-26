import torch
import yaml
import os

def save_model(model, tokenizer, epoch, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
    }, f"{path}_{epoch}.pth")
    print(f"Model saved to {path}_{epoch}.pth.")

def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params} / {all_params} ({100 * trainable_params / all_params:.2f}%)")

