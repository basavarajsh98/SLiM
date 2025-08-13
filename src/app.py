import gradio as gr
import torch

from config.config import get_config
from src.inference import generate_text, get_state_tensor, load_model_and_tokenizer

# Load configuration
config = get_config()

CHECKPOINTS = {
    "emotion": "./resources/checkpoints/emotion_steering/emotions.pth",
    "sentiment": "./resources/checkpoints/sentiment_steering/sentiments.pth",
    "language": "./resources/checkpoints/language_steering/languages.pth",
    "toxicity": "./resources/checkpoints/detoxification/detoxification.pth",
    "topic": "./resources/checkpoints/topic_steering/topics.pth",
    "multi_state": "./resources/checkpoints/multi_state_steering/multi_states.pth",
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

# Multi-state dropdown options
language_options = ["En", "De"]
sentiment_options = ["1", "2", "3", "4", "5"]
topic_options = ["Book", "Electronics", "Clothing", "Beauty"]

# ===== Preload all models at startup =====
loaded_models = {}
print("\n[Startup] Preloading models for all tasks...")
for task_name, ckpt_path in CHECKPOINTS.items():
    num_states = NUM_STATES[task_name]
    print(f"[Startup] Loading model for task '{task_name}' from {ckpt_path}")
    try:
        model_instance, tokenizer_instance = load_model_and_tokenizer(
            ckpt_path, num_states
        )
        model_instance.to(device)
        loaded_models[task_name.lower()] = (model_instance, tokenizer_instance)
    except Exception as e:
        print(f"Error loading model for {task_name}: {e}")
        loaded_models[task_name.lower()] = (None, None)
print("[Startup] All models loaded successfully.\n")

model = None
tokenizer = None


# Load model/tokenizer for a given task from cache
def load_model_for_task(task):
    print(f"[Action] Loading model for task: {task}")
    return loaded_models.get(task.lower(), (None, None))


# Update state selection UI and state vector display based on task
def update_model_and_state_choices(task):
    global model, tokenizer
    model, tokenizer = load_model_for_task(task)
    if task.lower() == "multi_state":
        print(
            "[UI] Multi-State task selected — showing topic/language/sentiment dropdowns"
        )
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
        print("[UI] Toxicity task selected — showing float input for state")
        return (
            gr.Textbox(label="Enter Toxicity Level", interactive=True),
            None,
            None,
            gr.Textbox(
                value="Enter a float value", label="State Vector", interactive=False
            ),
        )
    else:
        print(f"[UI] {task} task selected — showing state dropdown")
        task_mapping = config.get(task.lower() + "_mapping", {})
        choices = (
            ["None"] + [item.capitalize() for item in task_mapping.keys()]
            if task_mapping
            else ["None"]
        )
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
    print(
        f"[Action] Getting state vector for Task: {task}, State: {state}, Lang: {language}, Sent: {sentiment}"
    )
    if task.lower() == "multi_state" and state and language and sentiment:
        state_key = f"{state.lower()}_{language.lower()}_{sentiment.lower()}"
        _, state_vector = get_state_tensor(
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
            print("[Error] Invalid float value for toxicity state")
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
    print(f"[Generate] Task: {task}, Prompt: '{prompt[:50]}...'")
    print(
        f"[Generate] Params — Num Gen: {num_generations}, Max Tokens: {max_new_tokens}, Temp: {temperature}, Top-k: {top_k}, Top-p: {top_p}"
    )
    if task.lower() == "multi_state" and state and language and sentiment:
        state_key = f"{state.lower()}_{language.lower()}_{sentiment.lower()}"
        state_tensor, _ = get_state_tensor(
            config.get("multi_state_mapping", {}), state_key, device
        )
    elif "toxicity" in task.lower():
        try:
            state_tensor = torch.FloatTensor([float(state)]).unsqueeze(0).to(device)
        except ValueError:
            print("[Error] Invalid float value entered for toxicity")
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
    print("[Generate] Text generation completed.")
    return "\n\n".join(
        [f"Generation {i + 1}:\n{text}" for i, text in enumerate(generated_texts)]
    )


# Advanced options toggle
advanced_options_visible = False


def toggle_advanced_options():
    global advanced_options_visible
    advanced_options_visible = not advanced_options_visible
    visibility = advanced_options_visible
    print(f"[UI] Advanced options {'shown' if visibility else 'hidden'}")
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

    def on_task_change(task):
        print(f"[UI] Task changed to: {task}")
        state_elem, lang_elem, sent_elem, state_vec_elem = (
            update_model_and_state_choices(task)
        )
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

    def on_state_change(state_val, task_val, lang_val, sent_val):
        print(f"[UI] State changed to: {state_val}, Task: {task_val}")
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
        print(f"[UI] Generate button clicked for Task: {task}")
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

if __name__ == "__main__":
    app.launch()
