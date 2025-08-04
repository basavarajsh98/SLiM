import gradio as gr
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForCausalLM
from config.config import get_config
from src.model import SLiMedNet

# Load configuration and initialize the model and tokenizer
config = get_config()

def load_model_and_tokenizer(path, num_states):
    print("loading the model")
    model = AutoModelForCausalLM.from_pretrained(config['model_name'])
    model = SLiMedNet(state_embed_dim=num_states, model=model)
    checkpoint = torch.load(path, map_location=torch.device(config['device']))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def get_state_tensor(task, state, device):
    state_mapping = config[task]
    if state in state_mapping:
        state_vector = state_mapping[state]
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(device)
        return state_tensor
    else:
        return None

def generate_text(
    model, tokenizer, prompt, state_tensor=None, num_generations=5, max_new_tokens=50, temperature=1.0, top_k=50, top_p=0.9, no_repeat_ngram_size=2
):
    print(state_tensor)
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
            num_return_sequences=num_generations,  
            max_new_tokens = max_new_tokens,
            no_repeat_ngram_size=no_repeat_ngram_size,  
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            do_sample=True
        )
    return [tokenizer.decode(g, skip_special_tokens=True) for g in output]


# Load the model and tokenizer
CHECKPOINTS = {
    "emotion": "/home/qequ54zi/thesis/submission/SLiM/resources/checkpoints/SLiM_emotions_wo_500_24.pth",
    "sentiment": "/home/qequ54zi/thesis/submission/SLiM/resources/checkpoints/SLiM_sentiment_wo_500_24.pth",
    "language": "/home/qequ54zi/thesis/submission/SLiM/resources/checkpoints/SLiM_langauge_wo_24.pth",
    "toxicity": "/home/qequ54zi/thesis/submission/SLiM/resources/checkpoints/SLiM_detoxification_24.pth",
    "topic": "/home/qequ54zi/thesis/submission/SLiM/resources/checkpoints/SLiM_topic_wo_500_24.pth",
    "multi_state": "/home/qequ54zi/thesis/submission/SLiM/resources/checkpoints/SLiM_multi_state_wo_500_49.pth",

}
NUM_STATES = {
    "emotion": 5,
    "sentiment": 2,
    "language": 2,
    "toxicity": 1,
    "topic": 10,
    "multi_state": 3
}

device = torch.device(config['device'])

def load_model_for_task(task):
    task = task.lower()
    checkpoint_path = CHECKPOINTS.get(task, None)
    num_states = NUM_STATES.get(task, None)
    if checkpoint_path:
        model, tokenizer = load_model_and_tokenizer(checkpoint_path, num_states)
        model.to(device)
        return model, tokenizer
    else:
        return None, None

def update_model_and_emotion_choices(task):
    global model, tokenizer
    model, tokenizer = load_model_for_task(task)
    if "toxicity" in task.lower():
        # For "Toxicity," return a text input for user-entered state
        return gr.Textbox(label="Enter Toxicity Level", interactive=True), gr.Textbox(value="Enter a float value")
    else:
        # For other tasks, retain the dropdown behavior
        task = task.lower() + "_mapping"
        task_mapping = config.get(task, {})
        if task_mapping:
            choices = ["None"] + [item.capitalize() for item in task_mapping.keys()]
        else:
            choices = ["None"]
        return gr.Dropdown(choices=choices, value="None"), gr.Textbox(value="No state selected")

# Update the state vector display based on the selected emotion and task
def get_state_vector_with_task(state, task):
    task = task.lower() + "_mapping"
    if state != "None" and not ("toxicity" in task.lower()):
        state_vector = config.get(task, {}).get(state.lower(), "State vector not found")
    elif ("toxicity" in task.lower()):
        state_vector = [float(state)]
    else:
        state_vector = "No state selected"
    return f"{state_vector}"

def generate_texts_with_task(
    prompt, task, state, num_generations, max_new_tokens, temperature, top_k, top_p
):
    if "toxicity" in task.lower():
        try:
            # Convert user-entered value to a tensor
            state_tensor = torch.FloatTensor([float(state)]).unsqueeze(0).to(device)
        except ValueError:
            return "Error: Please enter a valid float value for the state."
    elif state != "None":
        # Use the mapped state tensor for other tasks
        task = task.lower() + "_mapping"
        state_tensor = get_state_tensor(task, state.lower(), device)
    else:
        state_tensor = None

    # Generate text with additional parameters
    generated_texts = generate_text(
        model,
        tokenizer,
        prompt,
        state_tensor=state_tensor,
        num_generations=int(num_generations),
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p)    )
    return "\n\n".join([f"Generation {i+1}:\n{text}" for i, text in enumerate(generated_texts)])

# Toggle state for advanced options
advanced_options_visible = False

# Function to toggle the visibility of advanced options
def toggle_advanced_options():
    global advanced_options_visible
    advanced_options_visible = not advanced_options_visible  # Flip the state
    visibility = advanced_options_visible
    return {
        num_generations_slider: gr.update(visible=visibility),
        max_new_tokens_slider: gr.update(visible=visibility),
        temperature_slider: gr.update(visible=visibility),
        top_k_slider: gr.update(visible=visibility),
        top_p_slider: gr.update(visible=visibility),
        advanced_options_button: gr.update(value="Hide Options" if visibility else "Advanced Options"),
    }

with gr.Blocks(theme=gr.themes.Ocean()) as app:
    gr.Markdown("# Steering GPT-2 with State Engineering")

    with gr.Row():
        with gr.Column():
            # Prompt input
            prompt_input = gr.Textbox(label="Enter Prompt", placeholder="Type your prompt here...")

            # Task selection dropdown
            steering_task = gr.Dropdown(
                choices=["None", "Emotion", "Sentiment", "Language", "Topic", "Toxicity"],
                label="Select Task",
                interactive=True
            )

            # State selection (updated dynamically based on task)
            # state = gr.Dropdown(
            #     choices=["None"], 
            #     label="Select State", 
            #     interactive=True           
            #      )
            state = gr.Textbox(label="Select State", interactive=True)

            # State vector display (non-editable)
            state_vector_display = gr.Textbox(label="State Vector", lines=1, interactive=False)

            # Advanced options toggle button
            advanced_options_button = gr.Button("Advanced Options")

            num_generations_slider = gr.Slider(1, 10, value=3, step=1, label="Number of Completions", visible=False)
            # Generation configuration sliders (initially hidden)
            max_new_tokens_slider = gr.Slider(10, 200, value=50, step=1, label="Max New Tokens", visible=False)
            temperature_slider = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature", visible=False)
            top_k_slider = gr.Slider(1, 100, value=50, step=1, label="Top-k", visible=False)
            top_p_slider = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="Top-p", visible=False)

            # Generate button
            generate_button = gr.Button("Generate")

        with gr.Column():
            # Display generated texts
            generated_texts_display = gr.Textbox(label="Text Completions", lines=10)

    # Update task dropdown logic
    steering_task.change(
        fn=update_model_and_emotion_choices,
        inputs=steering_task,
        outputs=[state, state_vector_display]
    )

    # Update state vector display
    state.change(
        fn=get_state_vector_with_task,
        inputs=[state, steering_task],
        outputs=state_vector_display
    )

    # Toggle visibility of sliders when advanced options are clicked
    advanced_options_button.click(
        fn=toggle_advanced_options,
        inputs=[],
        outputs=[
            num_generations_slider,
            max_new_tokens_slider,
            temperature_slider,
            top_k_slider,
            top_p_slider,
            advanced_options_button,
        ]
    )
    # Button click to generate text
    generate_button.click(
        fn=generate_texts_with_task,
        inputs=[
            prompt_input,
            steering_task,
            state,
            num_generations_slider,
            max_new_tokens_slider,
            temperature_slider,
            top_k_slider,
            top_p_slider        ],
        outputs=generated_texts_display
    )

# Launch the app
if __name__ == "__main__":
    app.launch()