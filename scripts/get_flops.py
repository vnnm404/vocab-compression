from models.gpt2 import GPT2
from models.gpt_neo import GPT_NEO
from models.simple import Simple

from utils.tokenizer import gpt2_tokenizer
from utils.config import config

import torch
from fvcore.nn import FlopCountAnalysis


tokenizer = gpt2_tokenizer()

# model = GPT2(tokenizer)
# model = GPT_NEO(tokenizer)
model = Simple(tokenizer)

input_ids, labels = torch.randint(0, len(tokenizer), (1, 512)), torch.randint(0, len(tokenizer), (1, 512))
attention_mask = torch.ones_like(input_ids)

flops = FlopCountAnalysis(model, inputs=(input_ids, attention_mask))
total_flops = flops.total()

# gigaflops
print(total_flops / 1e9)
