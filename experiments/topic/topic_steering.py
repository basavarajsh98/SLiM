import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.amp as amp
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F
from peft import prepare_model_for_kbit_training
import os
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter
from random import shuffle
from transformers import get_linear_schedule_with_warmup
import random
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

NUM_STATES=10
MAX_SEQUENCE_LENGTH=64

BATCH_SIZE=2
ACCUMULATION_STEPS=4
LEARNING_RATE=2e-5

EPOCHS=25
MAX_STEPS=None
SAVE_MODEL_PATH="resources/checkpoints/SLiM_topic"

MAX_SAMPLES_PER_TOPIC = 25000

import numpy as np
from collections import Counter
from random import shuffle, uniform
import torch
import os
from datasets import load_dataset
from transformers import AutoTokenizer

# Topics to index mapping
TOPIC_TO_INDEX = {
    'gourmet-food': 0,
    'video-game': 1,
    'clothing': 2,
    'beauty': 3,
    'arts':4,
    'book':5,
    'jewelry':6,
    'shoe':7,
    'musical-instrument':8,
    'electronics':9
}

def balance_dataset(texts):
    # Count the number of samples in each emotion category
    counts = Counter([tuple(label) for _, label in texts])
    min_count = 25000  # Find the minimum count across emotions
    print("min_count: ", min_count)
    balanced_texts = []
    emotion_buckets = {emotion: [] for emotion in counts.keys()}
    
    # Organize samples by emotion
    for text, label in texts:
        emotion_buckets[tuple(label)].append((text, tuple(label)))
    
    # Downsample to the minimum count in each category
    for emotion, samples in emotion_buckets.items():
        shuffle(samples)  # Shuffle to avoid any ordering bias
        balanced_texts.extend(samples[:min_count])

    balanced_texts = [(text, list(map(int, label))) for text, label in balanced_texts]
    shuffle(balanced_texts)  # Shuffle again to mix emotions in the final dataset
    return balanced_texts

def one_hot_encode(label, num_labels=10):
    if label < 0 or label >= num_labels:
        raise ValueError("Label must be between 0 and 9.")
    one_hot_vector = np.zeros(num_labels)
    one_hot_vector[label] = 1
    return one_hot_vector

def prepare_dataset(max_sequence_length, path='resources/datasets/topic.pth'):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    if not os.path.exists(path):
        datasets = [
            ('gourmet-food', load_dataset('contemmcm/amazon_reviews_2013', 'gourmet-food')),
            ('video-game', load_dataset('contemmcm/amazon_reviews_2013', 'video-game')),
            ('clothing', load_dataset('contemmcm/amazon_reviews_2013', 'clothing')),
            ('beauty', load_dataset('contemmcm/amazon_reviews_2013', 'beauty')),
            ('arts', load_dataset('contemmcm/amazon_reviews_2013', 'arts')),
            ('book', load_dataset('contemmcm/amazon_reviews_2013', 'book')),
            ('jewelry',load_dataset('contemmcm/amazon_reviews_2013', 'jewelry')),
            ('shoe',load_dataset('contemmcm/amazon_reviews_2013', 'shoe')),
            ('musical-instrument',load_dataset('contemmcm/amazon_reviews_2013', 'musical-instrument')),
            ('electronics',load_dataset('contemmcm/amazon_reviews_2013', 'electronics')),
        ]
        
        texts = []
        topic_counts = defaultdict(int)
        print("Started Preparing...")

        for topic, dataset in datasets:
            topic_index = TOPIC_TO_INDEX[topic]
            for item in dataset['complete']:
                if topic_counts[topic] < MAX_SAMPLES_PER_TOPIC:
                    rating = item.get('review/score', random.uniform(1, 5))  # Default to random rating if not available
                    texts.append((item['review/text'], one_hot_encode(topic_index)))
                    topic_counts[topic] += 1
                if topic_counts[topic] >= MAX_SAMPLES_PER_TOPIC:
                    break  # Stop adding more samples for this topic if limit is reached

        print("Preparation Complete. Sample counts per topic:")
        for topic, count in topic_counts.items():
            print(f"{topic}: {count} samples")
        
        # Balance the dataset
        print("Balancing dataset...")
        # texts = balance_dataset(texts, min_samples_per_bin=10000)
        print("Sample: ",texts[0])
        print("Total Size: ",len(texts))
        # Tokenize balanced texts
        print("Tokenizing started...")
        encoded_texts = tokenizer(
            [text for text, _ in texts], 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=max_sequence_length + 1
        )['input_ids']
        print("Done.")
        
        # Create final samples
        samples = [(tokens, state) for tokens, (_, state) in zip(encoded_texts, texts)]
        print("Total Size: ", len(samples))
        torch.save(samples, path)
    else:
        print(f"Found tokenized dataset at {path}!\nImporting...")
        samples = torch.load(path)
        print("Done.")    
        print("Total Size: ", len(samples))
    return samples, tokenizer

class SLiMed_Dataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample, state = self.samples[idx]
        input_seq = sample[:-1]
        target_seq = sample[1:]
        return input_seq, target_seq, torch.FloatTensor(state)

def collate_fn(batch):
    input_seqs, target_seqs, states = zip(*batch)
    input_seqs = torch.stack(input_seqs)
    target_seqs = torch.stack(target_seqs)
    states = torch.stack(states)
    return input_seqs, target_seqs, states

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"\nTotal trainable params: {trainable_params} || All params: {all_param} || Trainable %: {100 * trainable_params / all_param:.2f}%")

def save_model(model, tokenizer, epoch, loss, perplexity, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'loss': loss,
        'perplexity': perplexity,
    }, f"{path}_{epoch}.pth")
    print(f"Model saved to {path}_{epoch}.pth.")

class StateBlock(nn.Module):
    def __init__(self, state_dim, target_dim, activation=nn.ReLU):
        super(StateBlock, self).__init__()
        self.projection  = nn.Sequential(
            nn.Linear(state_dim, target_dim // 4), 
            activation(),
            nn.Linear(target_dim // 4, target_dim // 2),
            activation(),
            nn.Linear(target_dim // 2, target_dim),
            nn.LayerNorm(target_dim)
        )

    def forward(self, state_vector):
        return self.projection(state_vector)

class SLiMedNet(nn.Module):
    def __init__(self, state_embed_dim, apply_film_at_layers=None):
        super(SLiMedNet, self).__init__()
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        for param in model.parameters():
            param.requires_grad= False
        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False, 
            r=16, 
            lora_alpha=32, 
            lora_dropout=0.05,
            fan_in_fan_out=True        
            )
        self.gpt2 = get_peft_model(model, peft_config)
        self.state_proj = StateBlock(state_embed_dim, self.gpt2.config.n_embd)       
        self.apply_film_at_layers = apply_film_at_layers if apply_film_at_layers else list(range(len(self.gpt2.transformer.h)))
        self.gate = nn.ModuleList([nn.Linear(state_embed_dim, 1) for _ in range(len(self.apply_film_at_layers))])
        self.film_scale = nn.ModuleList([nn.Linear(self.gpt2.config.n_embd, self.gpt2.config.n_embd) for _ in range(len(self.apply_film_at_layers))])
        self.film_shift = nn.ModuleList([nn.Linear(self.gpt2.config.n_embd, self.gpt2.config.n_embd) for _ in range(len(self.apply_film_at_layers))])

        self.hooks = self.register_gpt_film_hooks()

    def register_gpt_film_hooks(self):
        """
        Register hooks for each transformer layer to apply FiLM modulation.
        """
        hooks = []
        for i, idx in enumerate(self.apply_film_at_layers):
            layer = self.gpt2.transformer.h[idx]
            hooks.append(layer.register_forward_hook(self.create_film_hook(i)))
        return hooks

    def create_film_hook(self, layer_idx):
        def hook(module, input, output):
            if self.current_state_embed is not None:
                projected_state = self.state_proj(self.current_state_embed)
                gate_value = torch.sigmoid(self.gate[layer_idx](self.current_state_embed))
                scale = torch.tanh(self.film_scale[layer_idx](projected_state))
                shift = torch.tanh(self.film_shift[layer_idx](projected_state))
                steered_output = (output[0] * scale + shift) * gate_value
                steered_output = F.layer_norm(steered_output, steered_output.shape[-1:])
                steered_output = steered_output + output[0]
                output = (steered_output,) + output[1:]
            return output
        return hook

    def forward(self, input_ids, state_tensor=None, attention_mask=None):
        if state_tensor is None:
            self.current_state_embed = None
        else:
            self.current_state_embed = state_tensor.unsqueeze(1)
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def generate(self, input_ids, state_tensor=None, attention_mask=None, **generate_kwargs):
        if state_tensor is None:
            self.current_state_embed = None
        else:
            self.current_state_embed = state_tensor.unsqueeze(1)
        return self.gpt2.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def generate_text(model, tokenizer, prompt, max_length, state_tensor=None):
    model.eval()
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs["attention_mask"]
    with torch.no_grad(), amp.autocast('cuda'):
        output = model.generate(
            input_ids, 
            state_tensor=state_tensor, 
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_length=max_length,  
            num_return_sequences=1,  
            no_repeat_ngram_size=2,  
            top_k=50,  
            top_p=0.93,  
            temperature=0.9,  
            do_sample=True
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def train(model, dataloader, criterion, optimizer, accumulation_steps=4, epochs=2, max_steps=None, save_model_path="results/SLiM_gpt"):
    model.train()
    step = 0
    for epoch in range(epochs):
        running_loss = 0
        optimizer.zero_grad()

        for i, (input_seq, target_seq, state) in enumerate(dataloader):
            input_seq, target_seq, state = input_seq.to(device), target_seq.to(device), state.to(device)
            
            with amp.autocast('cuda'):
                logits = model(input_seq, state)
                loss = criterion(logits.view(-1, logits.size(-1)), target_seq.view(-1))
            
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()  # Step the scheduler after each optimizer step
                step += 1
                # if max_steps:
                if step % 1000 == 0 or step == max_steps:
                    model.eval()
                    print(f"Step {step} loss: {(loss.item()):.3f}")
                    print("\n=============")                    
                    print("Baseline: ", generate_text(model, tokenizer, "I feel", max_length=40, state_tensor=None), '\n')
                    print("=============")  
                    print("Food: ", generate_text(model, tokenizer, "I feel", max_length=40, state_tensor=torch.FloatTensor(one_hot_encode(0)).unsqueeze(0).to(device)).strip(), '\n')
                    print("=============")                       
                    print("Gaming:", generate_text(model, tokenizer, "I feel", max_length=40, state_tensor=torch.FloatTensor(one_hot_encode(1)).unsqueeze(0).to(device)).strip(), '\n')
                    print("=============")                                      
                    print("Clothing: ", generate_text(model, tokenizer, "I feel", max_length=40, state_tensor=torch.FloatTensor(one_hot_encode(2)).unsqueeze(0).to(device)).strip(), '\n')                   
                    print("=============")   
                    print("Beauty: ", generate_text(model, tokenizer, "I feel", max_length=40, state_tensor=torch.FloatTensor(one_hot_encode(3)).unsqueeze(0).to(device)).strip(), '\n')
                    print("=============")   
                    print("arts: ", generate_text(model, tokenizer, "I feel", max_length=40, state_tensor=torch.FloatTensor(one_hot_encode(4)).unsqueeze(0).to(device)).strip(), '\n')
                    print("=============")          
                    print("book: ", generate_text(model, tokenizer, "I feel", max_length=40, state_tensor=torch.FloatTensor(one_hot_encode(5)).unsqueeze(0).to(device)).strip(), '\n')
                    print("=============")                              
                    print("jewelry: ", generate_text(model, tokenizer, "I feel", max_length=40, state_tensor=torch.FloatTensor(one_hot_encode(6)).unsqueeze(0).to(device)).strip(), '\n')
                    print("=============")                    
                    print("shoe: ", generate_text(model, tokenizer, "I feel", max_length=40, state_tensor=torch.FloatTensor(one_hot_encode(7)).unsqueeze(0).to(device)).strip(), '\n')
                    print("=============")                    
                    print("musical-instrument: ", generate_text(model, tokenizer, "I feel", max_length=40, state_tensor=torch.FloatTensor(one_hot_encode(8)).unsqueeze(0).to(device)).strip(), '\n')
                    print("=============")                    
                    print("electronics: ", generate_text(model, tokenizer, "I feel", max_length=40, state_tensor=torch.FloatTensor(one_hot_encode(9)).unsqueeze(0).to(device)).strip(), '\n')
                    print("=============\n")  
                    model.train()                  
            running_loss += loss.item()
            if max_steps and step >= max_steps:
                print(f"Reached max steps: {max_steps}. Stopping training.")
                step_loss = running_loss / (max_steps)
                step_ppx = torch.exp(torch.tensor(step_loss)).item()
                print(f"Step {max_steps} loss: {step_loss:.3f} perplexity: {step_ppx:.3f}")
                save_model(model, tokenizer, max_steps, step_loss, step_ppx, save_model_path)
                return

        epoch_loss = running_loss / len(dataloader)
        epoh_ppx = torch.exp(torch.Tensor([epoch_loss])).item()
        print(f"Epoch {epoch+1} loss: {epoch_loss:.3f} perplexity: {epoh_ppx:.3f}")
        save_model(model, tokenizer, epoch, epoch_loss, epoh_ppx, save_model_path)

if __name__== "__main__":
    samples, tokenizer = prepare_dataset(max_sequence_length=MAX_SEQUENCE_LENGTH)
    dataset = SLiMed_Dataset(samples)
    dataloader = DataLoader(dataset, 
                            batch_size=BATCH_SIZE, 
                            num_workers=4,
                            shuffle=True, 
                            pin_memory=True,
                            collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SLiMedNet(state_embed_dim=NUM_STATES).to(device)
    print_trainable_parameters(model)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = amp.GradScaler()
    # Define the number of total steps based on the dataset size and batch size
    total_steps = len(dataloader) // ACCUMULATION_STEPS * EPOCHS

    # Setting up the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Scheduler for learning rate, with warm-up steps for smoother convergence
    warmup_steps = int(0.1 * total_steps)  # Using 10% of total steps as warm-up
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )

    start = time.time()
    train(model, 
          dataloader=dataloader, 
          criterion=criterion, 
          optimizer=optimizer, 
          accumulation_steps=ACCUMULATION_STEPS, 
          epochs=EPOCHS,
          max_steps=MAX_STEPS,
          save_model_path=SAVE_MODEL_PATH
          )
    print(f"\nTraining completed in {(time.time() - start)/60} min.")
    print_trainable_parameters(model)
