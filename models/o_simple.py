import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from utils.config import config


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


class Simple(pl.LightningModule):
    def __init__(self, tokenizer):
        super().__init__()
        self.save_hyperparameters()

        self.vocab_size = len(tokenizer)
        self.hidden_size = config.model.gpt2["hidden_size"]
        self.num_groups = math.ceil(math.sqrt(self.vocab_size))
        self.group_size = math.ceil(self.vocab_size / self.num_groups)

        # Precompute group assignments and group token IDs
        token_ids = torch.arange(self.vocab_size)
        self.register_buffer('group_assignments', token_ids // self.group_size)
        self.register_buffer('group_token_ids', token_ids % self.group_size)

        # Embedding layers
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embedding = nn.Embedding(config.data.max_length, self.hidden_size)

        # Transformer decoder layers
        self.transformer = DecoderOnly()

        # Output layers
        self.grouper_linear = nn.Linear(self.hidden_size, self.num_groups)
        self.group_linears = nn.ModuleList([
            nn.Linear(self.hidden_size, self.group_size) for _ in range(self.num_groups)
        ])

        # Loss functions
        self.group_loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.token_loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    def forward(self, input_ids, labels=None):
        batch_size, seq_length = input_ids.size()

        # Create position ids
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = token_embeddings + position_embeddings

        # Transformer decoder
        # print(embeddings.shape)
        hidden_states = self.transformer(
            input_ids=embeddings,
            padding_mask=None,
        )  # [batch_size, seq_length, hidden_size]

        # Output computations
        group_logits = self.grouper_linear(hidden_states)  # [batch_size, seq_length, num_groups]

        if labels is not None:
            # Training mode
            # Get group IDs and group token IDs for labels
            group_ids = self.group_assignments[labels]  # [batch_size, seq_length]
            group_token_ids = self.group_token_ids[labels]  # [batch_size, seq_length]

            # Compute group loss
            group_loss = self.group_loss_fn(
                group_logits.view(-1, self.num_groups),
                group_ids.view(-1),
            )

            # Initialize token loss
            token_loss = torch.tensor(0.0, device=input_ids.device)

            # Process each group separately
            for group_id in range(self.num_groups):
                # Get positions where group_ids == group_id
                mask = group_ids == group_id  # [batch_size, seq_length]

                if mask.any():
                    positions = mask.nonzero(as_tuple=True)
                    hidden_states_group = hidden_states[positions]  # [num_positions, hidden_size]
                    token_logits_group = self.group_linears[group_id](hidden_states_group)  # [num_positions, group_size]
                    group_token_ids_group = group_token_ids[positions]  # [num_positions]

                    # Compute token loss for this group
                    token_loss_group = self.token_loss_fn(
                        token_logits_group,
                        group_token_ids_group,
                    )
                    token_loss += token_loss_group * (len(group_token_ids_group) / (batch_size * seq_length))

            total_loss = group_loss + token_loss
            return total_loss, group_loss, token_loss

        else:
            # Inference mode
            # First, predict group probabilities
            group_probs = F.softmax(group_logits, dim=-1)  # [batch_size, seq_length, num_groups]

            # For simplicity, pick the most probable group
            _, predicted_group_ids = torch.max(group_probs, dim=-1)  # [batch_size, seq_length]

            # Initialize token logits
            token_logits = torch.zeros(
                batch_size, seq_length, self.vocab_size, device=input_ids.device
            )

            # Process each group separately
            for group_id in range(self.num_groups):
                # Get positions where predicted_group_ids == group_id
                mask = predicted_group_ids == group_id  # [batch_size, seq_length]

                if mask.any():
                    positions = mask.nonzero(as_tuple=True)
                    hidden_states_group = hidden_states[positions]  # [num_positions, hidden_size]
                    token_logits_group = self.group_linears[group_id](hidden_states_group)  # [num_positions, group_size]

                    # Map group token logits back to vocab indices
                    vocab_indices = torch.arange(self.vocab_size, device=input_ids.device)
                    group_mask = self.group_assignments == group_id  # [vocab_size]
                    group_vocab_indices = vocab_indices[group_mask]  # [group_size]

                    # Place token_logits_group into token_logits at the correct vocab positions
                    for idx, (batch_idx, seq_idx) in enumerate(zip(*positions)):
                        token_logits[batch_idx, seq_idx, group_vocab_indices] = token_logits_group[idx]

            return token_logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        loss, group_loss, token_loss = self.forward(input_ids[:, :-1], labels[:, 1:])
        
        self.log('train_loss', loss)
        self.log('group_loss', group_loss)
        self.log('token_loss', token_loss)

        print(loss)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        loss, group_loss, token_loss = self.forward(input_ids[:, :-1], labels[:, 1:])
        
        self.log('val_loss', loss)
        self.log('group_loss', group_loss)
        self.log('token_loss', token_loss)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


if __name__ == '__main__':
    from utils.tokenizer import gpt2_tokenizer

    tokenizer = gpt2_tokenizer()
    model = Simple(tokenizer)

    # sample batch
    input_ids, labels = torch.randint(0, len(tokenizer), (4, 128)), torch.randint(0, len(tokenizer), (4, 128))
    attention_mask = torch.ones_like(input_ids)

    loss = model.training_step({
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }, 0)

    print(loss)

    val_loss = model.validation_step({
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }, 0)

    print(val_loss)
