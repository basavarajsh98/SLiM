import time
import warnings

import torch
import torch.amp as amp
import torch.optim as optim
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

from config.config import get_config
from src.dataset import SLiMed_Dataset, collate_fn
from src.inference import generate_text
from src.model import SLiMedNet
from src.utils import print_trainable_parameters, save_model

warnings.filterwarnings("ignore")

config = get_config()


def train_model(
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler=None,
    scaler=None,
    device=None,
    accumulation_steps=4,
    epochs=3,
    max_steps=None,
    save_model_path="results/SLiM_model",
    experiment_type="default",
    tokenizer=None,
    prompt_text="i feel",
    state_samples=None,
    eval_frequency=50,
    verbose=True,
):
    """
    Dynamic training function that can handle different experiment types and configurations.

    Args:
        model: The model to train
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        scaler: Gradient scaler for mixed precision training
        device: Device to train on
        accumulation_steps: Gradient accumulation steps
        epochs: Number of training epochs
        max_steps: Maximum training steps (optional)
        save_model_path: Path to save model checkpoints
        experiment_type: Type of experiment (default, detoxification, emotion_steering, etc.)
        tokenizer: Tokenizer for text generation during evaluation
        prompt_text: Text prompt for evaluation generation
        state_samples: List of state tensors to test during evaluation
        eval_frequency: How often to run evaluation (in steps)
        verbose: Whether to print training progress
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if scaler is None:
        scaler = amp.GradScaler()

    model.train()
    step = 0
    total_loss = 0

    for epoch in range(epochs):
        running_loss = 0
        optimizer.zero_grad()

        if verbose:
            print(f"\nEpoch {epoch + 1}/{epochs}")

        for i, (input_seq, target_seq, state) in enumerate(dataloader):
            input_seq, target_seq, state = (
                input_seq.to(device),
                target_seq.to(device),
                state.to(device),
            )

            with amp.autocast(device):
                logits = model(input_seq, state)
                loss = criterion(logits.view(-1, logits.size(-1)), target_seq.view(-1))

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

                step += 1
                total_loss += loss.item()

                if verbose and (step % eval_frequency == 0 or step == max_steps):
                    avg_loss = total_loss / step
                    print(
                        f"Step {step} - Avg Loss: {avg_loss:.3f} - Current Loss: {loss.item():.3f}"
                    )

                    # Run evaluation if tokenizer is provided
                    if tokenizer is not None:
                        run_evaluation(
                            model,
                            tokenizer,
                            device,
                            prompt_text,
                            state_samples,
                            experiment_type,
                            verbose,
                        )

                # Check if max steps reached
                if max_steps and step >= max_steps:
                    if verbose:
                        print(f"Reached max steps: {max_steps}. Stopping training.")
                        final_loss = total_loss / step
                        final_ppx = torch.exp(torch.tensor(final_loss)).item()
                        print(
                            f"Final - Loss: {final_loss:.3f} Perplexity: {final_ppx:.3f}"
                        )

                    save_model(
                        model,
                        tokenizer,
                        max_steps,
                        final_loss,
                        final_ppx,
                        save_model_path,
                    )
                    return

            running_loss += loss.item()

        # End of epoch
        epoch_loss = running_loss / len(dataloader)
        epoch_ppx = torch.exp(torch.Tensor([epoch_loss])).item()

        if verbose:
            print(
                f"Epoch {epoch + 1} - Loss: {epoch_loss:.3f} - Perplexity: {epoch_ppx:.3f}"
            )

        # Save model at end of epoch
        save_model(model, tokenizer, epoch, epoch_loss, epoch_ppx, save_model_path)

        # Run evaluation at end of epoch if tokenizer provided
        if tokenizer is not None:
            run_evaluation(
                model,
                tokenizer,
                device,
                prompt_text,
                state_samples,
                experiment_type,
                verbose,
            )


def run_evaluation(
    model, tokenizer, device, prompt_text, state_samples, experiment_type, verbose=True
):
    """Run evaluation with different state configurations based on experiment type."""
    if not verbose:
        return

    # Try to use specialized evaluation if available
    if experiment_type == "detoxification":
        try:
            from src.experiments.detoxification import run_detoxification_evaluation

            run_detoxification_evaluation(
                model, tokenizer, device, prompt_text, verbose
            )
            return
        except ImportError:
            # Fall back to default evaluation if specialized function not available
            pass
    elif experiment_type == "emotion_steering":
        try:
            from src.experiments.emotion_steering import run_emotion_steering_evaluation

            run_emotion_steering_evaluation(
                model, tokenizer, device, prompt_text, verbose
            )
            return
        except ImportError:
            # Fall back to default evaluation if specialized function not available
            pass
    elif experiment_type == "language_steering":
        try:
            from src.experiments.language_steering import (
                run_language_steering_evaluation,
            )

            run_language_steering_evaluation(
                model, tokenizer, device, prompt_text, verbose
            )
            return
        except ImportError:
            # Fall back to default evaluation if specialized function not available
            pass
    elif experiment_type == "topic_steering":
        try:
            from src.experiments.topic_steering import run_topic_steering_evaluation

            run_topic_steering_evaluation(
                model, tokenizer, device, prompt_text, verbose
            )
            return
        except ImportError:
            # Fall back to default evaluation if specialized function not available
            pass
    elif experiment_type == "multi_state_steering":
        try:
            from src.experiments.multi_state_steering import (
                run_multi_state_steering_evaluation,
            )

            run_multi_state_steering_evaluation(
                model, tokenizer, device, prompt_text, verbose
            )
            return
        except ImportError:
            # Fall back to default evaluation if specialized function not available
            pass
    elif experiment_type == "sentiment_steering":
        try:
            from src.experiments.sentiment_steering import (
                run_sentiment_steering_evaluation,
            )

            run_sentiment_steering_evaluation(
                model, tokenizer, device, prompt_text, verbose
            )
            return
        except ImportError:
            # Fall back to default evaluation if specialized function not available
            pass

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    # Baseline generation
    baseline_text = generate_text(
        model, tokenizer, prompt_text, max_length=40, state_tensor=None
    )
    print(f"Baseline: {baseline_text}")

    if state_samples is None:
        # Try to get specialized state samples if available
        if experiment_type == "detoxification":
            try:
                from src.experiments.detoxification import (
                    get_detoxification_state_samples,
                )

                state_samples = get_detoxification_state_samples()
            except ImportError:
                # Fall back to default samples
                state_samples = [0.0, 0.25, 0.5, 0.75, 0.8, 0.9, 1.0]
        elif experiment_type == "emotion_steering":
            try:
                from src.experiments.emotion_steering import get_emotion_state_samples

                state_samples = get_emotion_state_samples()
            except ImportError:
                # Fall back to default emotion samples
                state_samples = [
                    torch.FloatTensor([1, 0, 0, 0, 0]),  # Anger
                    torch.FloatTensor([0, 1, 0, 0, 0]),  # Fear
                    torch.FloatTensor([0, 0, 1, 0, 0]),  # Joy
                    torch.FloatTensor([0, 0, 0, 1, 0]),  # Love
                    torch.FloatTensor([0, 0, 0, 0, 1]),  # Sadness
                ]
        elif experiment_type == "language_steering":
            try:
                from src.experiments.language_steering import get_language_state_samples

                state_samples = get_language_state_samples()
            except ImportError:
                # Fall back to default language samples
                state_samples = [
                    torch.FloatTensor([1, 0]),  # English
                    torch.FloatTensor([0, 1]),  # German
                ]
        elif experiment_type == "topic_steering":
            try:
                from src.experiments.topic_steering import get_topic_state_samples

                state_samples = get_topic_state_samples()
            except ImportError:
                # Fall back to default topic samples
                state_samples = [
                    torch.FloatTensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # Gourmet Food
                    torch.FloatTensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),  # Video Games
                    torch.FloatTensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),  # Clothing
                    torch.FloatTensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),  # Beauty
                    torch.FloatTensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),  # Arts
                ]
        elif experiment_type == "sentiment_steering":
            try:
                from src.experiments.sentiment_steering import (
                    get_sentiment_state_samples,
                )

                state_samples = get_sentiment_state_samples()
            except ImportError:
                # Fall back to default sentiment samples
                state_samples = [
                    torch.FloatTensor([1, 0]),  # Positive
                    torch.FloatTensor([0, 1]),  # Negative
                ]
        else:
            # Default multi-state samples
            state_samples = [
                torch.FloatTensor([0, 0, 1]).unsqueeze(0),  # book_en_1
                torch.FloatTensor([0, 0, 5]).unsqueeze(0),  # book_en_5
                torch.FloatTensor([1, 0, 1]).unsqueeze(0),  # book_de_1
                torch.FloatTensor([1, 0, 5]).unsqueeze(0),  # book_de_5
            ]

    # Generate text for each state
    for i, state in enumerate(state_samples):
        if isinstance(state, (int, float)):
            # For detoxification (single float values)
            state_tensor = torch.FloatTensor([state]).unsqueeze(0).to(device)
            state_label = f"{state}"
        else:
            # For multi-state experiments
            state_tensor = state.to(device)
            state_label = f"State {i + 1}"

        generated_text = generate_text(
            model, tokenizer, prompt_text, max_length=40, state_tensor=state_tensor
        ).strip()

        print(f"{state_label}: {generated_text}")

    print("=" * 50 + "\n")


def setup_training_components(config, model, experiment_type="default"):
    """Setup training components based on configuration and experiment type."""
    device = torch.device(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )

    # Setup optimizer
    if experiment_type == "detoxification":
        optimizer = optim.AdamW(
            model.parameters(), lr=float(config.get("learning_rate", 2e-5))
        )
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=float(config.get("learning_rate", 2e-5))
        )

    # Setup scheduler for detoxification
    scheduler = None
    if experiment_type == "detoxification":
        total_steps = (
            config.get("epochs", 3) * config.get("batch_size", 2)
        ) // config.get("accumulation_steps", 4)
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

    # Setup criterion and scaler
    criterion = torch.nn.CrossEntropyLoss()
    scaler = amp.GradScaler()

    return device, optimizer, scheduler, criterion, scaler


def main(experiment_type="default", custom_config=None):
    """Main training function with experiment type selection."""
    # Use custom config if provided, otherwise use default
    if custom_config:
        config.update(custom_config)

    print(f"Starting {experiment_type} experiment...")
    print("Preparing dataset...")

    # Prepare dataset based on experiment type
    if experiment_type == "detoxification":
        from src.experiments.detoxification import prepare_dataset

        samples, tokenizer = prepare_dataset(
            max_sequence_length=config.get("max_sequence_length", 64)
        )
    else:
        samples, tokenizer = prepare_dataset()

    dataset = SLiMed_Dataset(samples)
    dataloader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 2),
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    device = torch.device(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )

    print("\nPreparing model...")

    # Model setup based on experiment type
    if experiment_type == "detoxification":
        model = SLiMedNet(state_embed_dim=config.get("num_states", 1)).to(device)
    else:
        # Original setup with LoRA
        base_model = AutoModelForCausalLM.from_pretrained(
            config.get("model_name", "openai-community/gpt2")
        )
        for param in base_model.parameters():
            param.requires_grad = False
        base_model = prepare_model_for_kbit_training(base_model)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            fan_in_fan_out=True,
        )
        base_model = get_peft_model(base_model, peft_config)
        model = SLiMedNet(
            state_embed_dim=config.get("num_states", 5), model=base_model
        ).to(device)

    print_trainable_parameters(model)

    # Setup training components
    device, optimizer, scheduler, criterion, scaler = setup_training_components(
        config, model, experiment_type
    )

    print("\nBegin training...\n")
    start_time = time.time()

    # Run training
    train_model(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        accumulation_steps=config.get("accumulation_steps", 4),
        epochs=config.get("epochs", 3),
        max_steps=config.get("max_steps"),
        save_model_path=config.get("save_model_path", "results/SLiM_model"),
        experiment_type=experiment_type,
        tokenizer=tokenizer,
        prompt_text=config.get("prompt_text", "i feel"),
        eval_frequency=config.get("eval_frequency", 50),
        verbose=True,
    )

    training_time = (time.time() - start_time) / 60
    print(f"\nTraining completed in {training_time:.2f} minutes!")
    print_trainable_parameters(model)


if __name__ == "__main__":
    # You can specify experiment type here or modify the config
    experiment_type = config.get("experiment_type", "default")
    main(experiment_type)
