# SLiM GPT-2 Project

This project fine-tunes a GPT-2 model with state conditioning using FiLM layers to modulate the model based on different emotions (e.g., joy, sadness).

## Requirements
Install the dependencies listed in `requirements.txt`.

## Directory Structure
- `src/`: Contains all the source code files.
- `config/`: Configuration files for the project.

## Running the Project
1. **Train the model:**
   ```bash
    python -m src.train
3. **Generate text:**
   ```bash
    python src/inference.py --prompt "I feel" --state "joy"
