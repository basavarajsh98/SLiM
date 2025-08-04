import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Example text
text = "happy Birthday"

# Tokenize input
inputs = tokenizer(text, return_tensors="pt")

# Ensure no gradient calculation
with torch.no_grad(): #opens a context in which gradient calculations are disabled. This ensures that only the forward pass is computed.
    outputs = model(**inputs, labels=inputs["input_ids"]) #forward pass- computes the model’s outputs and loss.
    loss = outputs.loss #extracts the loss from the model's outputs.
    perplexity = torch.exp(loss) #calculates the perplexity by exponentiating the loss.

print(f"Perplexity: {perplexity.item()}")

Perplexity: 23.123456789