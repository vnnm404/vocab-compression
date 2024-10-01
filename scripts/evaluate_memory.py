import torch
from transformers import AutoTokenizer

from models.simple_optim import Simple
from data.tinystories import TinyStoriesDataModule, ChunkedDataset
from training.trainer import train
from utils.tokenizer import gpt2_tokenizer
from utils.config import config

torch.set_float32_matmul_precision('medium')

def main():
    tokenizer = gpt2_tokenizer()
    print(len(tokenizer))
    model = Simple(tokenizer, group_size=config.model.compression['group_size'])

    # Sample batch
    input_ids = torch.randint(0, len(tokenizer), (4, 128))
    attention_mask = torch.ones_like(input_ids)
    

if __name__ == "__main__":
    main()
