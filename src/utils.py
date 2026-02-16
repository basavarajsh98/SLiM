import torch
import os
from config.config import get_config

config = get_config()


def save_model(model, tokenizer, epoch, loss, perplexity, path, final=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if final:
        save_path = f"{path}.pth"
    else:
        save_path = f"{path}_{epoch}.pth"

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "tokenizer": tokenizer,
            "loss": loss,
            "perplexity": perplexity,
        },
        save_path,
    )
    print(f"Model saved to {save_path}.")


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"\nTotal trainable params: {trainable_params} || All params: {all_param} || Trainable %: {100 * trainable_params / all_param:.2f}%"
    )
