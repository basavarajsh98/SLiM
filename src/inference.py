import torch
from transformers import AutoTokenizer
from src.model import SLiMedNet
from config.config import get_config
from torch.amp import autocast

config = get_config()
                  
def load_model_and_tokenizer(path):
    model = SLiMedNet(state_embed_dim=config['num_states'], model_name=config['model_name'])
    checkpoint = torch.load(path, map_location=torch.device(config['device']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length, state_tensor=None):
    model.eval()
    inputs = tokenizer(prompt, return_tensors='pt').to(config['device'])
    input_ids = inputs['input_ids']
    attention_mask = inputs["attention_mask"]
    with torch.no_grad():
        output = model.generate(
            input_ids, 
            state_tensor=state_tensor, 
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_length=max_length,  
            num_return_sequences=1,  
            no_repeat_ngram_size=2,  
            top_k=config['top_k'],
            top_p=config['top_p'],
            temperature=config['temperature'],
            do_sample=True
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def get_state_tensor(state, device):
    state_mapping = config['state_mapping']
    if state in state_mapping:
        state_vector = state_mapping[state]
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(device)
        return state_tensor

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer(config['save_model_path'])
    device = torch.device(config['device'])
    model = model.to(device)

    prompt = input("Enter the prompt: ")
    emotion = input("Enter emotion (joy/sad): ")
    state_tensor = get_state_tensor(state, device)
    generated_text = generate_text(model=model, 
                                   tokenizer=tokenizer, 
                                   prompt=prompt, 
                                   state_tensor=state_tensor)
    print("Response: ", generated_text)       