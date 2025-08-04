import gradio as gr
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForCausalLM
from config.config import get_config
from src.model import SLiMedNet

# Load configuration and model
config = get_config()

def load_model_and_tokenizer(path):
    model = SLiMedNet(state_embed_dim=3, model=AutoModelForCausalLM.from_pretrained(config['model_name']))
    checkpoint = torch.load(path, map_location=torch.device(config['device']))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

checkpoint = "/home/qequ54zi/thesis/submission/SLiM/resources/checkpoints/SLiM_multi_state_wo_500_49.pth"
model, tokenizer = load_model_and_tokenizer(checkpoint)
device = torch.device(config['device'])
model = model.to(device)

# Define dropdown options
language_options = ["En", "De"]
sentiment_options = ["1", "2", "3", "4", "5"]
topic_options = ["Book", "Electronics", "Clothing", "Beauty"]

# Map user input to state representation
def get_state_tensor(topic, language, sentiment, device):
    state_name = f"{topic.lower()}_{language.lower()}_{sentiment.lower()}"
    state_mapping = config['multi_state_mapping']
    if state_name in state_mapping:
        state_vector = state_mapping[state_name]
        return torch.FloatTensor(state_vector).unsqueeze(0).to(device), state_vector
    return None

# Define the generation function
def generate(prompt, topic, language, sentiment):
    state_tensor,_  = get_state_tensor(topic, language, sentiment, device)
    inputs = tokenizer(prompt, return_tensors='pt').to(config['device'])
    input_ids = inputs['input_ids']
    attention_mask = inputs["attention_mask"]
    with torch.no_grad():
        output = model.generate(
            input_ids, 
            state_tensor=state_tensor, 
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=3, 
            max_new_tokens=50,
no_repeat_ngram_size = 2,
            temperature=1,
            top_k=50,
            top_p=0.9,
            do_sample=True,
        )
    return [tokenizer.decode(g, skip_special_tokens=True) for g in output]

# Define the interface function
def inference(prompt, topic, language, sentiment):
    state_tensor, state_vector = get_state_tensor(topic, language, sentiment, device)
    generated_texts = generate(prompt, topic, language, sentiment)
    return "\n\n".join([f"Generation {i+1}:\n{text}" for i, text in enumerate(generated_texts)]), state_vector

# Gradio Interface
with gr.Blocks(theme=gr.themes.Ocean()) as interface:
    gr.Markdown("# Steering GPT-2 with State Engineering")
    with gr.Row():
        with gr.Column(scale=1):  # Column for dropdowns and state vector
            prompt = gr.Textbox(label="Enter Prompt", placeholder="Type your prompt here...")
            # Task selection dropdown
            steering_task = gr.Dropdown(
                choices=["Multi_State"],
                label="Select Task",
                interactive=True
            )
            topic = gr.Dropdown(topic_options, label="Select Topic State", interactive=True    )
            language = gr.Dropdown(language_options, label="Select Language State",interactive=True    )
            sentiment = gr.Dropdown(sentiment_options, label="Select Sentiment(Topic Rating) State",interactive=True    )
            state_vector_output = gr.Textbox(label="State Vector", lines=1, interactive=False)
            generate_button = gr.Button("Generate")

        with gr.Column(scale=2):  # Column for generated output
            generated_text_output = gr.Textbox(label="Text Completions", lines=10)

    # Button to trigger inference
    generate_button.click(
        inference,
        inputs=[prompt, topic, language, sentiment],
        outputs=[generated_text_output, state_vector_output],
    )

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch()
