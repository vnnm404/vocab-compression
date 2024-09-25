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


class Simple(L.LightningModule):
    def __init__(self, tokenizer, group_size=None, learning_rate=0.0005):
        super().__init__()
        self.save_hyperparameters()
        hidden_size = config.model.gpt2["hidden_size"]
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.pad_token_id = tokenizer.pad_token_id
        self.learning_rate = learning_rate
        if group_size is None:
            self.group_size = math.ceil(math.sqrt(self.vocab_size))
        else:
            self.group_size = group_size
        self.num_groups = math.ceil(self.vocab_size / self.group_size)
        self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        self.model = DecoderOnly()
        self.grouper = nn.Linear(hidden_size, self.num_groups)

        # Shared linear layer
        self.shared_linear = nn.Linear(hidden_size, self.group_size)

        # Per-group modulation parameters
        self.scale = nn.Embedding(self.num_groups, self.group_size)
        self.shift = nn.Embedding(self.num_groups, self.group_size)

        # Initialize modulation parameters
        nn.init.ones_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)

        self.ignore_index = -100
        self.group_loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.token_loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.val_sample_inputs = []
    
    def mapper(self, input_ids):
        """Maps input tokens to group IDs and new token IDs within the group."""
        new_tokens = input_ids % self.group_size  # Token IDs within the group
        groups = input_ids // self.group_size     # Group IDs
        return new_tokens, groups

    def apply_linear(self, h, groups):
        shared_output = self.shared_linear(h)  # [batch_size, seq_length, group_size]

        groups_flat = groups.view(-1)  # [N]
        shared_output_flat = shared_output.view(-1, self.group_size)  # [N, group_size]

        scale = self.scale(groups_flat)  # [N, group_size]
        shift = self.shift(groups_flat)  # [N, group_size]

        modulated_output_flat = shared_output_flat * scale + shift  # [N, group_size]
        modulated_output = modulated_output_flat.view_as(shared_output)  # [batch_size, seq_length, group_size]

        return modulated_output

    def forward(self, input_ids, attention_mask=None, labels=None):
        embeddings = self.embedding(input_ids)
        embeddings = self.positional_encoding(embeddings)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        padding_mask = (~attention_mask.bool()).float()
        h = self.model(embeddings, padding_mask)

        group_logits = self.grouper(h)  # [batch_size, seq_length, num_groups]

        if labels is not None:
            new_tokens, groups = self.mapper(labels)
            
            token_logits = self.apply_linear(h, groups)

            padding_mask = (labels == self.pad_token_id)
            groups = groups.masked_fill(padding_mask, self.ignore_index)
            new_tokens = new_tokens.masked_fill(padding_mask, self.ignore_index)

            # Compute group loss
            group_loss = self.group_loss_fn(
                group_logits.view(-1, self.num_groups),
                groups.view(-1),
            )

            # Compute token loss
            token_loss = self.token_loss_fn(
                token_logits.view(-1, self.group_size),
                new_tokens.view(-1),
            )

            total_loss = group_loss + token_loss

            return total_loss, group_loss, token_loss

        else:
            group_probs = F.softmax(group_logits, dim=-1)
            predicted_groups = torch.argmax(group_probs, dim=-1)  # [batch_size, seq_length]

            token_logits = self.apply_linear(h, predicted_groups)

            predicted_tokens = torch.argmax(token_logits, dim=-1)
            output_tokens = predicted_groups * self.group_size + predicted_tokens

            return output_tokens

    def training_step(self, batch, batch_idx):
        start_time = time.time()

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        outputs = self(
            input_ids=input_ids[:, :-1],
            attention_mask=attention_mask[:, :-1],
            labels=input_ids[:, 1:],
        )

        loss, group_loss, token_loss = outputs

        end_time = time.time()

        batch_size, seq_length = input_ids.size()
        num_tokens = batch_size * seq_length

        time_taken = end_time - start_time
        tokens_per_sec = num_tokens / time_taken
        self.log('train_tokens_per_sec', tokens_per_sec, prog_bar=True)

        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)  # in GB
            self.log('train_memory_MB', mem_allocated, prog_bar=True)

        self.log('train_loss', loss, prog_bar=True)
        self.log('group_loss', group_loss)
        self.log('token_loss', token_loss)

        # print(loss, group_loss, token_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        start_time = time.time()

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        outputs = self(
            input_ids=input_ids[:, :-1],
            attention_mask=attention_mask[:, :-1],
            labels=input_ids[:, 1:],
        )
        
        loss, group_loss, token_loss = outputs

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
        self.log('val_group_loss', group_loss)
        self.log('val_token_loss', token_loss)

        return loss
    
    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']

        predicted_tokens = self(
            input_ids=input_ids,
            attention_mask=None,
            labels=None,
        )

        return predicted_tokens
    
    def generate(self, text, max_length=256):
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        
        for _ in range(max_length):
            outputs = self.test_step({"input_ids": input_ids}, 0)

            if outputs[0, -1].item() == self.tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, outputs[:, -1].unsqueeze(0)], dim=-1)

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


if __name__ == '__main__':
    from utils.tokenizer import gpt2_tokenizer

    tokenizer = gpt2_tokenizer()
    model = Simple(tokenizer, len(tokenizer))

    # Sample batch
    input_ids = torch.randint(0, len(tokenizer), (4, 128))
    attention_mask = torch.ones_like(input_ids)

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    loss = model.training_step(batch, 0)
    print(loss)

    val_loss = model.validation_step(batch, 0)
    print(val_loss)

    # Test inference
    test_outputs = model.test_step(batch, 0)
    # print(test_outputs)
