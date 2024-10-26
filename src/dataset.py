import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import os
import numpy as np
from config.config import get_config
from torch.utils.data import Dataset
import pickle

config = get_config()

def one_hot_encode_emotions(emotion, num_classes=config['num_states']):
    one_hot_vector = np.zeros(num_classes, dtype=int)
    one_hot_vector[[2, 4].index(emotion)] = 1
    return one_hot_vector

def prepare_dataset(max_sequence_length=config['max_sequence_length'], path=config['dataset_path']):
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    if not os.path.exists(path):
        print("Loading dataset...")
        dataset = load_dataset('ma2za/many_emotions', 'raw')

        texts=[]
        for item in dataset['en']:
            label = item['label']
            if label in [2, 4]:
                one_hot_vector = one_hot_encode_emotions(label)
                texts.append((item['text'], one_hot_vector))

        base_emotions = [ 'joy', 'sad']
        emotion_counts = {emotion: 0 for emotion in base_emotions}

        for example,label in texts:
            one_hot_vector = label
            for i, count in enumerate(one_hot_vector):
                if count == 1:
                    emotion_counts[base_emotions[i]] += 1

        print("\nTotal Count: " ,emotion_counts)
        print("\nSample: ", texts[:1])   

        print("\nStarting tokenization...")
        encoded_texts = tokenizer(
            [text for text, _ in texts],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_sequence_length + 1
        )['input_ids']
        print("Done.")
        
        samples = [(tokens, state) for tokens, (_, state) in zip(encoded_texts, texts)]
        torch.save(samples, path, _use_new_zipfile_serialization=True)
        print(f"Saved to {path}")
    else:
        print(f"Found tokenized dataset at {path}!\nImporting...")
        samples = torch.load(path)
        print("Done.")
    return samples, tokenizer

def collate_fn(batch):
    input_seqs, target_seqs, states = zip(*batch)
    input_seqs = torch.stack(input_seqs)
    target_seqs = torch.stack(target_seqs)
    states = torch.stack(states)
    return input_seqs, target_seqs, states

class CustomDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample, state = self.samples[idx]
        input_seq = sample[:-1]
        target_seq = sample[1:]
        return input_seq, target_seq, torch.FloatTensor(state)