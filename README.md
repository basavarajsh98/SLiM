
## State-wise Linear Modulation (SLiM): A Novel Approach for Steering Large Language Models

State-wise Linear Modulation (SLiM) is a novel, efficient, and generalizable approach for steering large language models. SLiM fine-tunes a language model (GPT-2) by injecting state information through SLiM layers, enabling precise and context-aware control over generated text. This method allows the model to adapt to a wide range of steering tasks with minimal data and retraining.

## Features


- **Multi-Task & Multi-State Support**: Seamlessly switch between and combine steering tasks, including emotion, sentiment, language, topic, toxicity, and multi-state (combinatorial) control.
- **Efficient & Lightweight**: SLiM learns effective modulation parameters with minimal data and without extensive retraining, making it suitable for resource-constrained or rapid-prototyping scenarios.
- **Generalizable**: Demonstrated strong performance across categorical, ordinal, and continuous state values, and across diverse datasets.
- **Evaluation & Visualization**: Includes scripts for quantitative and qualitative evaluation (toxicity, perplexity, KL-divergence) and for visualizing SLiM parameters and model behavior.

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

## Supported Steering Tasks
SLiM is designed to support a wide range of steering tasks, including:
- **Emotion Steering**: anger, fear, joy, love, sadness
- **Sentiment Steering**: positive, negative
- **Language Steering**: English, German
- **Topic Steering**: 10 Amazon product topics
- **Detoxification**: continuous toxicity control
- **Multi-State Steering**: combinations of topic, language, and sentiment/rating

## Usage

## 0. Proof-of-Concept (PoC) Notebooks
The `src/poc/` directory contains Jupyter notebooks representing the first approaches we explored before developing the SLiM method. These preliminary implementations were conducted as proof of concept (PoC)—early, naive attempts to explore whether alternate formulations could be viable. These scripts were not followed by extensive experimentation or analysis like the SLiM method

These PoC files are useful for understanding, testing, and visualizing core ideas before integrating them into the main codebase.

### 1. Training
Train a SLiM-enhanced model using the main script. You can specify the experiment type in `config/config.yaml` or via code:
```bash
python -m src.train
```
Or for a specific SLiM steering task (e.g., emotion):
```python
PYTHONPATH=. python ./src/experiments/emotion_steering.py
```

### 2. Inference (CLI)
Generate text with a SLiM-trained checkpoint:
```bash
python -m src.inference
```

### 3. Streamlit Web App
Launch the interactive demo:
```bash
streamlit run src/app.py
```
This provides a web UI to select tasks, states, and generate text interactively using SLiM.

### 4. Evaluation & Visualization

**Evaluation**: All scripts and modules for evaluating SLiM models and steering tasks are located in `src/evaluation`. This includes quantitative and qualitative assessments.

**Visualization**: Tools and scripts for visualizing SLiM parameters, model behavior, and experiment results are found in `src/visualizations`.

## Configuration
Edit `config/config.yaml` to adjust model, training, and state mappings for SLiM. Example:

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies

## Citation
If you use State-wise Linear Modulation (SLiM) in your research, please cite this repository.

---
For more details, see the code and comments in each script. Contributions and issues are welcome!
