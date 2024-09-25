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
            # Inference mode
            # Only process the last token to save memory
            h_last = h[:, -1, :]  # [batch_size, hidden_size]
            group_logits_last = self.grouper(h_last)  # [batch_size, num_groups]

            # Compute group probabilities
            group_probs = F.softmax(group_logits_last, dim=-1)  # [batch_size, num_groups]

            # Compute token logits within each group
            shared_output = self.shared_linear(h_last)  # [batch_size, group_size]

            # Modulate shared output for each group
            scale = self.scale.weight  # [num_groups, group_size]
            shift = self.shift.weight  # [num_groups, group_size]

            # Expand shared_output to [batch_size, 1, group_size]
            shared_output_expanded = shared_output.unsqueeze(1)  # [batch_size, 1, group_size]

            # Apply modulation for each group
            modulated_output = shared_output_expanded * scale.unsqueeze(0) + shift.unsqueeze(0)  # [batch_size, num_groups, group_size]

            # Compute token probabilities within each group
            token_probs = F.softmax(modulated_output, dim=-1)  # [batch_size, num_groups, group_size]

            # Combine group and token probabilities
            # P(t) = P(group) * P(t | group)
            probs = group_probs.unsqueeze(-1) * token_probs  # [batch_size, num_groups, group_size]

            # Map to vocabulary indices
            group_offset = torch.arange(self.num_groups, device=h.device) * self.group_size  # [num_groups]
            token_indices = torch.arange(self.group_size, device=h.device)  # [group_size]
            vocab_indices = (group_offset.unsqueeze(1) + token_indices.unsqueeze(0)).flatten()  # [num_groups * group_size]

            # Handle tokens beyond vocab size
            valid_indices = vocab_indices < self.vocab_size
            vocab_indices = vocab_indices[valid_indices]

            # Flatten probs and select valid indices
            probs_flat = probs.view(h.size(0), -1)  # [batch_size, num_groups * group_size]
            probs_flat = probs_flat[:, valid_indices]

            # Initialize probabilities over the full vocabulary
            full_probs = torch.zeros((h.size(0), self.vocab_size), device=h.device)
            full_probs[:, vocab_indices] = probs_flat

            # Return the full probability distribution over the vocabulary
            return full_probs

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
    
    def generate(self, text, max_length=256, temperature=0.4, top_k=None, top_p=None):
        self.eval()  # Set model to evaluation mode
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)

        for _ in range(max_length):
            # Get the probability distribution over the vocabulary
            full_probs = self(
                input_ids=input_ids,
                attention_mask=None,
                labels=None
            )  # [batch_size, vocab_size]

            # Apply temperature scaling
            if temperature != 1.0:
                full_probs = full_probs.pow(1.0 / temperature)

            # Apply top-k sampling
            if top_k is not None:
                top_k = min(max(top_k, 1), full_probs.size(-1))  # Safety check
                indices_to_remove = full_probs < torch.topk(full_probs, top_k)[0][:, -1, None]
                full_probs[indices_to_remove] = 0

            # Apply nucleus (top-p) sampling
            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(full_probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                indices_to_remove = cumulative_probs > top_p
                sorted_probs[indices_to_remove] = 0
                # Scatter back to original ordering
                full_probs = torch.zeros_like(full_probs).scatter(-1, sorted_indices, sorted_probs)

            # Normalize probabilities
            full_probs = full_probs / full_probs.sum(dim=-1, keepdim=True)

            # Sample the next token
            next_token = torch.multinomial(full_probs, num_samples=1)  # [batch_size, 1]

            # Append the predicted next token
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Stop if EOS token is generated
            if next_token.item() == self.tokenizer.eos_token_id:
                break

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
