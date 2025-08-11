import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.config import get_config
from src.model import SLiMedNet

warnings.filterwarnings("ignore")

config = get_config()


def load_model_and_tokenizer(path, num_states):
    model = AutoModelForCausalLM.from_pretrained(config["base_model"], torch_dtype=torch.float16)
    model = SLiMedNet(state_embed_dim=num_states, model=model)
    checkpoint = torch.load(path, map_location=torch.device(config["device"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_length=40, state_tensor=None, num_generations=5):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(config["device"])
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    with torch.no_grad():
        output = model.generate(
            input_ids,
            state_tensor=state_tensor,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_length=max_length,
            max_new_tokens=100,
            num_return_sequences=num_generations,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.9,
            temperature=1,
            do_sample=True,
        )
    return [tokenizer.decode(g, skip_special_tokens=True) for g in output]


def get_state_tensor(state_mapping, state, device):
    state_mapping = config[state_mapping]
    if state in state_mapping:
        state_vector = state_mapping[state]
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(device)
        return state_tensor


# if __name__ == "__main__":
#     print("Loading checkpoint...\n")
#     checkpoint = "./resources/checkpoints/SLiMedGPT2_5.pth"
#     model, tokenizer = load_model_and_tokenizer(checkpoint, NUM_STATES)
#     device = torch.device(config["device"])
#     model = model.to(device)

#     print("Have at it!!")
#     while True:
#         try:
#             prompt = input("\nPrompt: ")
#             state = input("State (joy/sad/Enter): ")
#             state_tensor = get_state_tensor(STATE_MAPPING, state, device)
#             generated_text = generate_text(
#                 model=model,
#                 tokenizer=tokenizer,
#                 prompt=prompt,
#                 state_tensor=state_tensor,
#             )
#             print("=========")
#             print(generated_text)
#             print("=========")
#         except Exception as e:
#             print("Oops!! ", e)
