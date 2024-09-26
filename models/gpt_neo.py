import torch
import pytorch_lightning as L
import torch.nn.functional as F
from transformers import GPTNeoConfig, GPTNeoForCausalLM
from utils.config import config

import time


class GPT_NEO(L.LightningModule):
    def __init__(self, tokenizer, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()

        attention_types = [[["global", "local"], config.model.gpt2["layers"] // 2]]
        gpt_neo_config = GPTNeoConfig(
            vocab_size=len(tokenizer),
            max_position_embeddings=config.data.max_length,
            hidden_size=config.model.gpt2["hidden_size"],
            num_layers=config.model.gpt2["layers"],
            num_heads=config.model.gpt2["heads"],
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            attention_types=attention_types,
            window_size=config.model.gptneo["window_size"],
        )
        self.model = GPTNeoForCausalLM(gpt_neo_config)
        self.config = gpt_neo_config

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.learning_rate = learning_rate

        self.val_predictions = []
        self.val_targets = []

        self.test_generations = []

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        start_time = time.time()

        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss

        end_time = time.time()

        batch_size, seq_length = input_ids.size()
        num_tokens = batch_size * seq_length

        time_taken = end_time - start_time
        tokens_per_sec = num_tokens / time_taken
        self.log('train_tokens_per_sec', tokens_per_sec, prog_bar=True)

        print(tokens_per_sec)

        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_reserved() // (1024 * 1024 * 1024)  # in GB
            self.log('train_memory_MB', mem_allocated, prog_bar=True)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        start_time = time.time()

        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        end_time = time.time()

        batch_size, seq_length = input_ids.size()
        num_tokens = batch_size * seq_length

        time_taken = end_time - start_time
        tokens_per_sec = num_tokens / time_taken
        self.log('val_tokens_per_sec', tokens_per_sec, prog_bar=True)

        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_reserved() // (1024 * 1024 * 1024)  # in GB
            self.log('val_memory_MB', mem_allocated, prog_bar=True)

        loss = outputs.loss
        self.log("val_loss", loss)

        return loss

    def on_validation_epoch_end(self):
        self.val_predictions = []
        self.val_targets = []

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]

        output_ids = self.model.generate(
            input_ids=input_ids,
            max_length=config.data.max_length,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.pad_token_id,
            num_return_sequences=1,
        )

        decoded_outputs = [
            self.tokenizer.decode(output.cpu().numpy(), skip_special_tokens=True) for output in output_ids
        ]
        self.test_generations.extend(decoded_outputs)

        return decoded_outputs

    def on_test_epoch_end(self):
        self.test_predictions = []
        self.test_targets = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


if __name__ == "__main__":
    from utils.tokenizer import gpt2_tokenizer

    tokenizer = gpt2_tokenizer()
    model = GPT_NEO(tokenizer)

    # sample batch
    input_ids, labels = torch.randint(0, len(tokenizer), (4, 128)), torch.randint(
        0, len(tokenizer), (4, 128)
    )
    attention_mask = torch.ones_like(input_ids)

    loss = model.training_step(
        {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}, 0
    )

    print(loss)

    val_loss = model.validation_step(
        {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}, 0
    )

    print(val_loss)

    test_generations = model.test_step({"input_ids": input_ids}, 0)

    print(test_generations)
