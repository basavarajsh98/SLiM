
## State-wise Linear Modulation (SLiM): A Novel Approach for Steering Large Language Models

State-wise Linear Modulation (SLiM) is a novel, efficient, and generalizable approach for steering large language models. SLiM fine-tunes an LLM(GPT-2) by injecting state information through SLiM layers, enabling precise and context-aware control over generated text. This method allows the model to adapt to a wide range of steering tasks with minimal data and retraining.

## Features


- **Multi-Task & Multi-State Support**: Seamlessly switch between and combine steering tasks, including emotion, sentiment, language, topic, toxicity, and multi-state (combinatorial) control.
- **Efficient & Lightweight**: SLiM learns effective modulation parameters with minimal data and without extensive retraining, making it suitable for resource-constrained or rapid-prototyping scenarios.
- **Generalizable**: Demonstrated strong performance across categorical, ordinal, and continuous state values, and across diverse datasets.
- **Evaluation & Visualization**: Includes scripts for quantitative and qualitative evaluation (toxicity, perplexity, KL-divergence) and for visualizing SLiM parameters and model behavior.
- **Interactive Gradio App**: Explore SLiM's capabilities interactively via a web UI.

## Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/basavarajsh98/SLiM
   cd SLiM
   ```
2. **Create and activate a virtual environment**
   - On Unix/macOS:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Directory Structure
- `src/` - Source code
  - `app.py` - Gradio web app for interactive demo
  - `train.py` - Main training script for SLiM
  - `inference.py` - Inference utilities for SLiM-based text generation
  - `model.py` - SLiMedNet model with SLiM-based state-wise modulation
  - `dataset.py` - Dataset and dataloader utilities
  - `run_experiment.py` - (Optional) Experiment runner
  - `experiments/` - Task-specific training/evaluation scripts (all SLiM-based)
  - `evaluation/` - Evaluation and analysis scripts
  - `visualizations/` - Visualization scripts (SLiM, plots, etc.)
- `config/` - Configuration files
  - `config.yaml` - Main config (model, states, mappings, etc.)
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation

## Supported Steering Tasks
SLiM is designed to support a wide range of steering tasks, including:
- **Emotion Steering**: anger, fear, joy, love, sadness
- **Sentiment Steering**: positive, negative
- **Language Steering**: English, German
- **Topic Steering**: 10 Amazon product topics
- **Detoxification**: continuous toxicity control
- **Multi-State Steering**: combinations of topic, language, and sentiment/rating

## Usage
### 1. Training
Train a SLiM-enhanced model using the main script. You can specify the experiment type in `config/config.yaml` or via code:
```bash
python -m src.train
```
Or for a specific SLiM steering task (e.g., emotion):
```python
from src.experiments.emotion_steering import run_emotion_steering_experiment
run_emotion_steering_experiment()
```

### 2. Inference (CLI)
Generate text with a SLiM-trained checkpoint:
```bash
python -m src.inference
```

### 3. Gradio Web App
Launch the interactive demo:
```bash
python src/app.py
```
This provides a web UI to select tasks, states, and generate text interactively using SLiM.

### 4. Evaluation & Visualization
- **Toxicity Evaluation**: `src/evaluation/detoxification.py` (SLiM-based)
- **SLiM Parameter Visualization**: `src/visualizations/SLiM.py`
- **Perplexity & KL-Divergence Plots**: `src/visualizations/plots.py`

## Configuration
Edit `config/config.yaml` to adjust model, training, and state mappings for SLiM. Example:
```yaml
model_name: openai-community/gpt2
num_states: 5
# ...
emotion_mapping:
  anger: [1, 0, 0, 0, 0]
  fear: [0, 1, 0, 0, 0]
  joy: [0, 0, 1, 0, 0]
  love: [0, 0, 0, 1, 0]
  sadness: [0, 0, 0, 0, 1]
```

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies (transformers, torch, datasets, gradio, etc.)

## Citation
If you use State-wise Linear Modulation (SLiM) in your research, please cite this repository.

---
For more details, see the code and comments in each script. Contributions and issues are welcome!
