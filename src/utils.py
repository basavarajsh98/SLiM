import torch

from config.config import get_config

config = get_config()


def save_model(model, tokenizer, epoch, loss, perplexity, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "tokenizer": tokenizer,
            "loss": loss,
            "perplexity": perplexity,
        },
        f"{path}_{epoch}.pth",
    )
    print(f"Model saved to {path}_{epoch}.pth.")


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
