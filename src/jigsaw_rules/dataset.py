import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding

class RedditDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

class ExtendedRedditDataset(Dataset):
    def __init__(self, encodings, labels, extra_feats):
        self.encodings = encodings          # dict of lists/tensors from tokenizer
        self.labels = labels                # list/array of ints (0/1)
        self.extra_feats = extra_feats      # list/array of shape [N, 2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        # store as list; collator will tensorize & stack
        item["extra_feats"] = torch.tensor(self.extra_feats[idx], dtype=torch.float)
        return item

class FusionCollator:
    """Pads text fields and stacks the 2-prob feature as `extra_feats`."""
    def __init__(self, tokenizer):
        self.base = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        # pull out and remove extras before padding
        extras = torch.stack([f.pop("extra_feats") for f in features], dim=0)
        batch = self.base(features)
        batch["extra_feats"] = extras  # [B, 2]
        return batch
