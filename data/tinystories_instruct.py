import os
import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from datasets import Dataset as HFDataset
from typing import Optional
from utils.config import config
import pickle
from tqdm import tqdm
import json

class ChunkedDataset(Dataset):
    def __init__(self, tokenized_data):
        self.inputs = tokenized_data

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]

class TinyStoriesInstructDataModule(L.LightningDataModule):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def setup(self, stage: Optional[str] = None):
        cache_path_train = os.path.join(config.cache_dir, "tinystories_instruct_train_dataset.json")
        cache_path_val = os.path.join(config.cache_dir, "tinystories_instruct_val_dataset.json")

        if os.path.exists(cache_path_train) and os.path.exists(cache_path_val):
            with open(cache_path_train, 'r') as f_train:
                self.train_dataset = ChunkedDataset(json.load(f_train))
            with open(cache_path_val, 'r') as f_val:
                self.val_dataset = ChunkedDataset(json.load(f_val))
        else:
            dataset = load_dataset("roneneldan/TinyStoriesInstruct")

            # tokenized_dataset = dataset.map(
            #     lambda x: self.tokenizer([item + self.tokenizer.eos_token for item in x["text"]], truncation=False, padding=False),
            #     batched=True,
            #     num_proc=config.data.num_workers,
            # )

            merged_train_dataset = self._make_story(dataset["train"])
            merged_val_dataset = self._make_story(dataset["validation"])

            tokenized_train_dataset = merged_train_dataset.map(
                lambda x: self.tokenizer([item + self.tokenizer.eos_token for item in x["text"]], truncation=False, padding=False),
                batched=True,
                num_proc=config.data.num_workers,
            )

            tokenized_val_dataset = merged_val_dataset.map(
                lambda x: self.tokenizer([item + self.tokenizer.eos_token for item in x["text"]], truncation=False, padding=False),
                batched=True,
                num_proc=config.data.num_workers,
            )

            train_chunks = self._chunk_data(tokenized_train_dataset)
            val_chunks = self._chunk_data(tokenized_val_dataset)

            self.train_dataset = ChunkedDataset(train_chunks)
            self.val_dataset = ChunkedDataset(val_chunks)

            os.makedirs(config.cache_dir, exist_ok=True)
            with open(cache_path_train, 'w') as f_train:
                json.dump(train_chunks, f_train)
            with open(cache_path_val, 'w') as f_val:
                json.dump(val_chunks, f_val)
    
    def _make_story(self, x):
        storyies = []
        story = ""
        for item in tqdm(x):
            if item["text"].strip() == "<|endoftext|>":
                storyies.append(story)
                story = ""
                continue

            story += item["text"] + "\n"
        
        dataset = HFDataset.from_dict({"text": storyies})
        return dataset

    def _chunk_data(self, tokenized_data):
        chunks = []
        for item in tqdm(tokenized_data):
            input_ids = item["input_ids"]

            for i in range(0, len(input_ids), config.data.max_length):
                chunk = input_ids[i:i + config.data.max_length]    
                chunks.append(chunk)
        return chunks
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        padded_batch = pad_sequence(
            [torch.LongTensor(chunk) for chunk in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = (padded_batch != self.tokenizer.pad_token_id).int()

        labels = padded_batch.clone()
        # labels[padded_batch == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": padded_batch,
            "labels": labels,
            "attention_mask": attention_mask
        }

if __name__ == '__main__':
    from utils.tokenizer import gpt2_tokenizer

    tokenizer = gpt2_tokenizer()
    dm = TinyStoriesInstructDataModule(tokenizer)

    dm.setup()

    val_dataloader = dm.val_dataloader()

    for batch in val_dataloader:
        print(batch)
        print(tokenizer.decode(batch["input_ids"][0]))
        break

    train_dataloader = dm.train_dataloader()

    for batch in train_dataloader:
        print(batch)
        print(tokenizer.decode(batch["input_ids"][0]))
        break
