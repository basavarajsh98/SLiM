import gradio as gr
import torch

from config.config import get_config
from src.inference import generate_text, get_state_tensor, load_model_and_tokenizer

# Load configuration and initialize the model and tokenizer
config = get_config()

CHECKPOINTS = {
    "emotion": "./SLiM/resources/checkpoints/SLiM_emotions_wo_500_24.pth",
    "sentiment": "./SLiM/resources/checkpoints/SLiM_sentiment_wo_500_24.pth",
    "language": "./SLiM/resources/checkpoints/SLiM_langauge_wo_24.pth",
    "toxicity": "./SLiM/resources/checkpoints/SLiM_detoxification_24.pth",
    "topic": "./SLiM/resources/checkpoints/SLiM_topic_wo_500_24.pth",
    "multi_state": "./SLiM/resources/checkpoints/SLiM_multi_state_wo_500_49.pth",
}
NUM_STATES = {
    "emotion": 5,
    "sentiment": 2,
    "language": 2,
    "toxicity": 1,
    "topic": 10,
    "multi_state": 3,
}

device = torch.device(config["device"])

# Multi-state dropdown options (from old GUI.py)
language_options = ["En", "De"]
sentiment_options = ["1", "2", "3", "4", "5"]
topic_options = ["Book", "Electronics", "Clothing", "Beauty"]

model = None
tokenizer = None


# Load model/tokenizer for a given task
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


# Update state selection UI and state vector display based on task
def update_model_and_state_choices(task):
    global model, tokenizer
    model, tokenizer = load_model_for_task(task)
    if task.lower() == "multi_state":
        # Use dropdowns for multi-state
        return (
            gr.Dropdown(
                choices=topic_options, label="Select Topic State", interactive=True
            ),
            gr.Dropdown(
                choices=language_options,
                label="Select Language State",
                interactive=True,
            ),
            gr.Dropdown(
                choices=sentiment_options,
                label="Select Sentiment(Topic Rating) State",
                interactive=True,
            ),
            gr.Textbox(
                value="No state selected", label="State Vector", interactive=False
            ),
        )
    elif "toxicity" in task.lower():
        return (
            gr.Textbox(label="Enter Toxicity Level", interactive=True),
            None,
            None,
            gr.Textbox(
                value="Enter a float value", label="State Vector", interactive=False
            ),
        )
    else:
        # For other tasks, use dropdown for state
        task_mapping = config.get(task.lower() + "_mapping", {})
        if task_mapping:
            choices = ["None"] + [item.capitalize() for item in task_mapping.keys()]
        else:
            choices = ["None"]
        return (
            gr.Dropdown(
                choices=choices, value="None", label="Select State", interactive=True
            ),
            None,
            None,
            gr.Textbox(
                value="No state selected", label="State Vector", interactive=False
            ),
        )


def get_state_vector_with_task(state, task, language=None, sentiment=None):
    if task.lower() == "multi_state" and state and language and sentiment:
        # Compose state key for multi-state
        state_key = f"{state.lower()}_{language.lower()}_{sentiment.lower()}"
        state_tensor, state_vector = get_state_tensor(
            config.get("multi_state_mapping", {}), state_key, device
        )
        return str(state_vector)
    task_mapping = task.lower() + "_mapping"
    if state != "None" and "toxicity" not in task.lower():
        state_vector = config.get(task_mapping, {}).get(
            state.lower(), "State vector not found"
        )
    elif "toxicity" in task.lower():
        try:
            state_vector = [float(state)]
        except Exception:
            state_vector = "Invalid float value"
    else:
        state_vector = "No state selected"
    return f"{state_vector}"


def generate_texts_with_task(
    prompt,
    task,
    state,
    num_generations,
    max_new_tokens,
    temperature,
    top_k,
    top_p,
    language=None,
    sentiment=None,
):
    global model, tokenizer
    if task.lower() == "multi_state" and state and language and sentiment:
        state_key = f"{state.lower()}_{language.lower()}_{sentiment.lower()}"
        state_tensor, _ = get_state_tensor(
            config.get("multi_state_mapping", {}), state_key, device
        )
    elif "toxicity" in task.lower():
        try:
            state_tensor = torch.FloatTensor([float(state)]).unsqueeze(0).to(device)
        except ValueError:
            return "Error: Please enter a valid float value for the state."
    elif state != "None":
        task_mapping = task.lower() + "_mapping"
        state_tensor = get_state_tensor(
            config.get(task_mapping, {}), state.lower(), device
        )
    else:
        state_tensor = None
    generated_texts = generate_text(
        model,
        tokenizer,
        prompt,
        state_tensor=state_tensor,
        num_generations=int(num_generations),
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
    )
    return "\n\n".join(
        [f"Generation {i + 1}:\n{text}" for i, text in enumerate(generated_texts)]
    )


# Advanced options toggle
advanced_options_visible = False


def toggle_advanced_options():
    global advanced_options_visible
    advanced_options_visible = not advanced_options_visible
    visibility = advanced_options_visible
    return {
        num_generations_slider: gr.update(visible=visibility),
        max_new_tokens_slider: gr.update(visible=visibility),
        temperature_slider: gr.update(visible=visibility),
        top_k_slider: gr.update(visible=visibility),
        top_p_slider: gr.update(visible=visibility),
        advanced_options_button: gr.update(
            value="Hide Options" if visibility else "Advanced Options"
        ),
    }


with gr.Blocks(theme=gr.themes.Ocean()) as app:
    gr.Markdown("# Steering an LLM(GPT-2) with State Engineering")
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Enter Prompt", placeholder="Type your prompt here..."
            )
            steering_task = gr.Dropdown(
                choices=[
                    "None",
                    "Emotion",
                    "Sentiment",
                    "Language",
                    "Topic",
                    "Toxicity",
                    "Multi_State",
                ],
                label="Select Task",
                interactive=True,
            )
            # Placeholders for dynamic state selection
            state = gr.Dropdown(
                choices=["None"], label="Select State", interactive=True
            )
            language_dropdown = gr.Dropdown(
                choices=language_options,
                label="Select Language State",
                interactive=True,
                visible=False,
            )
            sentiment_dropdown = gr.Dropdown(
                choices=sentiment_options,
                label="Select Sentiment(Topic Rating) State",
                interactive=True,
                visible=False,
            )
            state_vector_display = gr.Textbox(
                label="State Vector", lines=1, interactive=False
            )
            advanced_options_button = gr.Button("Advanced Options")
            num_generations_slider = gr.Slider(
                1, 10, value=3, step=1, label="Number of Completions", visible=False
            )
            max_new_tokens_slider = gr.Slider(
                10, 200, value=50, step=1, label="Max New Tokens", visible=False
            )
            temperature_slider = gr.Slider(
                0.1, 2.0, value=1.0, step=0.1, label="Temperature", visible=False
            )
            top_k_slider = gr.Slider(
                1, 100, value=50, step=1, label="Top-k", visible=False
            )
            top_p_slider = gr.Slider(
                0.5, 1.0, value=0.9, step=0.05, label="Top-p", visible=False
            )
            generate_button = gr.Button("Generate")
        with gr.Column():
            generated_texts_display = gr.Textbox(label="Text Completions", lines=10)

    # Update task dropdown logic
    def on_task_change(task):
        state_elem, lang_elem, sent_elem, state_vec_elem = (
            update_model_and_state_choices(task)
        )
        # Show/hide language/sentiment dropdowns for multi_state
        if task.lower() == "multi_state":
            return {
                state: gr.update(
                    visible=True, label="Select Topic State", choices=topic_options
                ),
                language_dropdown: gr.update(visible=True),
                sentiment_dropdown: gr.update(visible=True),
                state_vector_display: gr.update(value="No state selected"),
            }
        elif "toxicity" in task.lower():
            return {
                state: gr.update(
                    visible=True, label="Enter Toxicity Level", choices=None
                ),
                language_dropdown: gr.update(visible=False),
                sentiment_dropdown: gr.update(visible=False),
                state_vector_display: gr.update(value="Enter a float value"),
            }
        else:
            return {
                state: gr.update(visible=True, label="Select State"),
                language_dropdown: gr.update(visible=False),
                sentiment_dropdown: gr.update(visible=False),
                state_vector_display: gr.update(value="No state selected"),
            }

    steering_task.change(
        fn=on_task_change,
        inputs=steering_task,
        outputs=[state, language_dropdown, sentiment_dropdown, state_vector_display],
    )

    # Update state vector display
    def on_state_change(state_val, task_val, lang_val, sent_val):
        return get_state_vector_with_task(state_val, task_val, lang_val, sent_val)

    state.change(
        fn=on_state_change,
        inputs=[state, steering_task, language_dropdown, sentiment_dropdown],
        outputs=state_vector_display,
    )
    language_dropdown.change(
        fn=on_state_change,
        inputs=[state, steering_task, language_dropdown, sentiment_dropdown],
        outputs=state_vector_display,
    )
    sentiment_dropdown.change(
        fn=on_state_change,
        inputs=[state, steering_task, language_dropdown, sentiment_dropdown],
        outputs=state_vector_display,
    )
    # Toggle advanced options
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
        ],
    )

    # Generate button click
    def on_generate(
        prompt,
        task,
        state_val,
        num_gen,
        max_tokens,
        temp,
        topk,
        topp,
        lang_val,
        sent_val,
    ):
        return generate_texts_with_task(
            prompt,
            task,
            state_val,
            num_gen,
            max_tokens,
            temp,
            topk,
            topp,
            lang_val,
            sent_val,
        )

    generate_button.click(
        fn=on_generate,
        inputs=[
            prompt_input,
            steering_task,
            state,
            num_generations_slider,
            max_new_tokens_slider,
            temperature_slider,
            top_k_slider,
            top_p_slider,
            language_dropdown,
            sentiment_dropdown,
        ],
        outputs=generated_texts_display,
    )
# Launch the app
if __name__ == "__main__":
    app.launch()
