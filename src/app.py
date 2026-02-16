import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config.config import get_config
from src.model import SLiMedNet

# -------------------------
# Load configuration
# -------------------------
config = get_config()

CHECKPOINTS = {
    "emotion": "./resources/checkpoints/emotion_steering/emotions.pth",
    "sentiment": "./resources/checkpoints/sentiment_steering/sentiments.pth",
    "language": "./resources/checkpoints/language_steering/languages.pth",
    "toxicity": "./resources/checkpoints/detoxification/detox.pth",
    "topic": "./resources/checkpoints/topic_steering/topics.pth",
    "multi_state": "./resources/checkpoints/multi_state_steering/multi_states.pth",
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

# -------------------------
# Model loading functions
# -------------------------
@st.cache_resource
def load_model_and_tokenizer(path, num_states):
    model = AutoModelForCausalLM.from_pretrained(config['base_model'])
    model = SLiMedNet(state_embed_dim=num_states, model=model)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
    tokenizer.pad_token = tokenizer.eos_token
    return model.to(device), tokenizer

def load_model_for_task(task):
    task = task.lower()
    checkpoint_path = CHECKPOINTS.get(task)
    num_states = NUM_STATES.get(task)
    if checkpoint_path and num_states is not None:
        return load_model_and_tokenizer(checkpoint_path, num_states)
    else:
        return None, None

def get_state_tensor(task, state):
    state_mapping = config[task]
    if state in state_mapping:
        return torch.FloatTensor(state_mapping[state]).unsqueeze(0).to(device)
    return None

# -------------------------
# Text generation function
# -------------------------
def generate_text(
    model, tokenizer, prompt, state_tensor=None, num_generations=5, max_new_tokens=50, 
    temperature=1.0, top_k=50, top_p=0.9, no_repeat_ngram_size=2
):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            state_tensor=state_tensor,
            num_return_sequences=num_generations,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=no_repeat_ngram_size,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return [tokenizer.decode(g, skip_special_tokens=True) for g in output]

# -------------------------
# Streamlit UI Layout
# -------------------------
st.set_page_config(page_title="SLiMedNet Text Steering", layout="wide")
st.title("🧭 Steering GPT-2 with State Engineering")

# Create two columns
left_col, right_col = st.columns([1, 2])  # Left: controls, Right: output

# ===== LEFT COLUMN: Controls =====
with left_col:
    prompt = st.text_area("Enter your prompt", "")

    task = st.selectbox("Select Task", ["None", "Emotion", "Sentiment", "Language", "Topic", "Toxicity"])

    if task != "None":
        model, tokenizer = load_model_for_task(task)
    else:
        model, tokenizer = None, None

    state_value = None
    if task.lower() == "toxicity":
        state_value = st.text_input("Enter Toxicity Level (float)")
    elif task.lower() != "none":
        mapping = config.get(task.lower() + "_mapping", {})
        state_choice = st.selectbox("Select State", ["None"] + [k.capitalize() for k in mapping.keys()])
        if state_choice != "None":
            state_value = state_choice.lower()

    # Show state vector
    if task.lower() == "toxicity" and state_value:
        try:
            state_vector = [float(state_value)]
        except ValueError:
            state_vector = "Invalid float value"
    elif task.lower() != "none" and state_value:
        state_vector = config.get(task.lower() + "_mapping", {}).get(state_value, "State vector not found")
    else:
        state_vector = "No state selected"

    st.markdown(f"**State Vector:** `{state_vector}`")

    with st.expander("Advanced Options"):
        num_generations = st.slider("Number of completions", 1, 10, 3)
        max_new_tokens = st.slider("Max new tokens", 10, 200, 50)
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0)
        top_k = st.slider("Top-k", 1, 100, 50)
        top_p = st.slider("Top-p", 0.5, 1.0, 0.9)

    generate_clicked = st.button("Generate")

# ===== RIGHT COLUMN: Output =====
with right_col:
    if generate_clicked:
        if model is None or tokenizer is None:
            st.error("Please select a valid task first.")
        else:
            if task.lower() == "toxicity" and state_value:
                try:
                    state_tensor = torch.FloatTensor([float(state_value)]).unsqueeze(0).to(device)
                except ValueError:
                    st.error("Invalid float value for toxicity level.")
                    st.stop()
            elif state_value and state_value != "None":
                state_tensor = get_state_tensor(task.lower() + "_mapping", state_value)
            else:
                state_tensor = None

            outputs = generate_text(
                model, tokenizer, prompt,
                state_tensor=state_tensor,
                num_generations=num_generations,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            st.markdown("###### Generated Completions")
            for i, text in enumerate(outputs, 1):
                st.markdown(
                    f"""
                    <div style="
                        padding: 1rem;
                        margin-bottom: 1rem;
                        border-radius: 10px;
                        background-color: #2c3e50; /* Dark slate background */
                        color: white; /* White text */
                        box-shadow: 0 1px 6px rgba(0,0,0,0.3);
                    ">
                        <strong>Generation {i}:</strong><br>
                        {text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

