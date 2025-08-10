import numpy as np
import torch
from torch.utils.data import Dataset

from config.config import get_config

config = get_config()


def one_hot_encode_emotions(emotion, num_classes=config["num_states"]):
    one_hot_vector = np.zeros(num_classes, dtype=int)
    one_hot_vector[[2, 4].index(emotion)] = 1
    return one_hot_vector


def collate_fn(batch):
    input_seqs, target_seqs, states = zip(*batch)
    input_seqs = torch.stack(input_seqs)
    target_seqs = torch.stack(target_seqs)
    states = torch.stack(states)
    return input_seqs, target_seqs, states


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
