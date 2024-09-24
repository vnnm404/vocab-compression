import torch
import math
import pytorch_lightning as L
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from utils.config import config

import time
from fvcore.nn import FlopCountAnalysis

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.register_buffer('pe', self._pe(max_len, d_model))

    def _pe(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

class DecoderOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=config.model.gpt2["hidden_size"],
            nhead=config.model.gpt2["heads"],
            dim_feedforward=4 * config.model.gpt2["hidden_size"],
            activation='gelu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            self.layer,
            num_layers=config.model.gpt2["layers"],
        )
    
    def forward(self, input_ids, padding_mask):
        mask = nn.Transformer.generate_square_subsequent_mask(input_ids.size(1)).to(input_ids.device)
        output = self.transformer(input_ids, mask=mask, src_key_padding_mask=padding_mask)
        return output

class Baseline(L.LightningModule):
    def __init__(self, tokenizer, learning_rate=0.0005):
        super().__init__()
        self.save_hyperparameters()

        hidden_size = config.model.gpt2["hidden_size"]
        
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.learning_rate = learning_rate

        self.embedding = nn.Embedding(len(tokenizer), hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        
        self.model = DecoderOnly()
        
        self.lm_head = nn.Linear(hidden_size, len(tokenizer))

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.val_sample_inputs = []
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        embeddings = self.embedding(input_ids)
        embeddings = self.positional_encoding(embeddings)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        padding_mask = (~attention_mask.bool()).float()

        h = self.model(embeddings, padding_mask)

        logits = self.lm_head(h)  # [batch_size, seq_length, vocab_size]

        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Flatten the tokens
            loss = self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            return loss

        else:
            return logits
    
    def training_step(self, batch, batch_idx):
        start_time = time.time()

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        loss = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        end_time = time.time()

        batch_size, seq_length = input_ids.size()
        num_tokens = batch_size * seq_length

        time_taken = end_time - start_time
        tokens_per_sec = num_tokens / time_taken
        self.log('train_tokens_per_sec', tokens_per_sec, prog_bar=True)

        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)  # in GB
            self.log('train_memory_MB', mem_allocated, prog_bar=True)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        start_time = time.time()

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        loss = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        end_time = time.time()

        batch_size, seq_length = input_ids.size()
        num_tokens = batch_size * seq_length

        time_taken = end_time - start_time
        tokens_per_sec = num_tokens / time_taken
        self.log('val_tokens_per_sec', tokens_per_sec, prog_bar=True)

        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)  # in GB
            self.log('val_memory_MB', mem_allocated, prog_bar=True)

        self.log('val_loss', loss)

        return loss
    
    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        logits = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
        )

        # Get predictions
        predictions = torch.argmax(logits, dim=-1)

        return predictions
    
    def generate(self, text, max_length=256):
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)

        for _ in range(max_length):
            attention_mask = torch.ones_like(input_ids).to(self.device)
            logits = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
            )

            next_token_logits = logits[:, -1, :]

            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


if __name__ == '__main__':
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    model = Baseline(tokenizer)

    # Sample batch
    input_ids = torch.randint(0, len(tokenizer), (4, 128))
    attention_mask = torch.ones_like(input_ids)

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    loss = model.training_step(batch, 0)
    print('Training loss:', loss.item())

    val_loss = model.validation_step(batch, 0)
    print('Validation loss:', val_loss.item())

    # Test inference
    test_outputs = model.test_step(batch, 0)
    print('Test outputs:', test_outputs)

    # Test generation
    generated_text = model.generate("Once upon a time", max_length=50)
    print('Generated text:', generated_text)
