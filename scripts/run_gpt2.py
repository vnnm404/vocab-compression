import torch
from transformers import AutoTokenizer

from models.gpt2 import GPT2
from data.tinystories import TinyStoriesDataModule, ChunkedDataset
from training.trainer import train
from utils.tokenizer import gpt2_tokenizer
from utils.config import config

torch.set_float32_matmul_precision('medium')

def main():
    tokenizer = gpt2_tokenizer()

    model = GPT2(tokenizer)
    data_module = TinyStoriesDataModule(tokenizer)

    trainer = train(model, data_module)

if __name__ == "__main__":
    main()