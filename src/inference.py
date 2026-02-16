import sys
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from colorama import init, Fore, Style
from config.config import get_config
from src.model import SLiMedNet

warnings.filterwarnings("ignore")
init(autoreset=True)

config = get_config()


def load_model_and_tokenizer(path, num_states):
    print(Fore.CYAN + f"Loading base model: {config['base_model']}...")
    model = AutoModelForCausalLM.from_pretrained(config["base_model"])
    model = SLiMedNet(state_embed_dim=num_states, model=model)
    checkpoint = torch.load(path, map_location=config["device"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(config["device"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    print(Fore.GREEN + "✅ Model & tokenizer loaded.\n")
    return model, tokenizer


def generate_text(
    model, tokenizer, prompt, state_tensor=None, num_generations=1, **kwargs
):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(config["device"])
    defaults = {
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 100,
        "num_return_sequences": num_generations,
        "no_repeat_ngram_size": 2,
        "top_k": 50,
        "top_p": 0.9,
        "temperature": 1.0,
        "do_sample": True,
    }
    generate_args = {
        **defaults,
        **kwargs,
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }
    if state_tensor is not None:
        generate_args["state_tensor"] = state_tensor

    with torch.no_grad():
        output = model.generate(**generate_args)

    return [tokenizer.decode(g, skip_special_tokens=True) for g in output]


def get_state_tensor(state_mapping, state, device):
    mapping = config[state_mapping]
    if state in mapping:
        return torch.FloatTensor(mapping[state]).unsqueeze(0).to(device)


if __name__ == "__main__":
    experiments = {
        "Setting Emotional Tone": {
            "checkpoint": "resources/checkpoints/emotion_steering/emotions.pth",
            "state_mapping": "emotion_mapping",
        },
        "Sentiment Steering": {
            "checkpoint": "resources/checkpoints/sentiment_steering/sentiments.pth",
            "state_mapping": "sentiment_mapping",
        },
        "Language Toggling": {
            "checkpoint": "resources/checkpoints/language_steering/languages.pth",
            "state_mapping": "language_mapping",
        },
        "Discussing Topic": {
            "checkpoint": "resources/checkpoints/topic_steering/languages.pth",
            "state_mapping": "topic_mapping",
        },
        "Detoxify": {
            "checkpoint": "resources/checkpoints/sentiment_steering/sentiments.pth",
            "state_mapping": "toxicity_mapping",
        },
        "Multi_State Maneuvering": {
            "checkpoint": "resources/checkpoints/multi_state_steering/multi_states.pth",
            "state_mapping": "multi_state_mapping",
        },
    }

    # Add num_states dynamically
    for exp in experiments:
        mapping_key = experiments[exp]["state_mapping"]
        mapping = config.get(mapping_key, {})
        if isinstance(mapping, dict) and mapping:
            experiments[exp]["num_states"] = len(next(iter(mapping.values())))
        else:
            experiments[exp]["num_states"] = config.get("num_states", 5)

    # Main menu
    print(Fore.YELLOW + "=" * 50)
    print(Fore.YELLOW + Style.BRIGHT + "      SLiMedNet Interactive Inference")
    print(Fore.YELLOW + "=" * 50)
    print("\nAvailable experiments:")
    for i, exp in enumerate(experiments, 1):
        print(Fore.CYAN + f"  {i}. {exp}")

    choice = input(Fore.MAGENTA + "\nSelect experiment by number: ").strip()
    try:
        exp_name = list(experiments.keys())[int(choice) - 1]
    except Exception:
        print(Fore.RED + "❌ Invalid choice. Exiting.")
        sys.exit(1)

    exp_info = experiments[exp_name]
    model, tokenizer = load_model_and_tokenizer(
        exp_info["checkpoint"], exp_info["num_states"]
    )

    # Get available states for this experiment
    available_states = list(config.get(exp_info["state_mapping"], {}).keys())

    print(Fore.YELLOW + f"🚀 Loaded {exp_name} experiment.")
    if available_states:
        print(
            Fore.BLUE
            + "📌 Available states: "
            + Fore.WHITE
            + ", ".join(available_states)
        )
    else:
        print(Fore.RED + "⚠ No states found in mapping.")

    print(Fore.BLUE + f"\nType {Fore.RED}'exit'{Fore.BLUE} at any time to quit.\n")
    while True:
        prompt = input(Fore.GREEN + "Prompt: " + Fore.WHITE).strip()
        if prompt.lower() == "exit":
            print(Fore.YELLOW + "👋 Goodbye!")
            break

        state = input(Fore.GREEN + f"State ({exp_info['state_mapping']}): " + Fore.WHITE).strip()
        if state.lower() == "exit":
            print(Fore.YELLOW + "👋 Goodbye!")
            break

        state_tensor = get_state_tensor(
            exp_info["state_mapping"], state, config["device"]
        )
        if state_tensor is None:
            print(Fore.RED + f"⚠ State '{state}' not found. Try again.\n")
            continue

        outputs = generate_text(model, tokenizer, prompt, state_tensor=state_tensor)
        print(Fore.CYAN + "\n--- Generated text ---")
        for i, out in enumerate(outputs, 1):
            print(Fore.WHITE + Style.BRIGHT + f"[{i}] " + Fore.RESET + out)
        print(Fore.CYAN + "----------------------\n")
