import torch
import torch.nn as nn
import pytorch_lightning as pl
import math

class ImageClassificationModel(pl.LightningModule):
    def __init__(self, vocab_size=184320, learning_rate=1e-3, group_size=None):
        super().__init__()
        self.save_hyperparameters()
        hidden_size = 512

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),  # Output: 32 x 100 x 100
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32 x 50 x 50
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # Output: 64 x 50 x 50
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 x 25 x 25
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # Output: 128 x 25 x 25
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 128 x 12 x 12
        )

        self.flatten_size = 128 * 12 * 12  # 18,432 features

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, hidden_size),
            nn.ReLU(),
        )

        if group_size is None:
            self.group_size = math.ceil(math.sqrt(vocab_size))
        else:
            self.group_size = group_size
        self.num_groups = math.ceil(vocab_size / self.group_size)

        self.group_linear = nn.Linear(hidden_size, self.num_groups)

        self.shared_linear = nn.Linear(hidden_size, self.group_size)

        self.scale = nn.Embedding(self.num_groups, self.group_size)
        self.shift = nn.Embedding(self.num_groups, self.group_size)

        nn.init.ones_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)

        self.group_loss_fn = nn.CrossEntropyLoss()
        self.token_loss_fn = nn.CrossEntropyLoss()

        self.learning_rate = learning_rate

    def mapper(self, labels):
        """Maps labels to group IDs and token IDs within the group."""
        token_in_group = labels % self.group_size  # Token IDs within the group
        group_ids = labels // self.group_size      # Group IDs
        return token_in_group, group_ids

    def apply_linear(self, h, group_ids):
        # h: [batch_size, hidden_size]
        shared_output = self.shared_linear(h)

        scale = self.scale(group_ids)
        shift = self.shift(group_ids)

        # Modulate the shared_output
        modulated_output = shared_output * scale + shift

        return modulated_output

    def forward(self, x, labels=None):

        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        h = self.fc_layers(x)

        group_logits = self.group_linear(h)

        if labels is not None:
            token_in_group, group_ids = self.mapper(labels)

            token_logits = self.apply_linear(h, group_ids)

            return group_logits, token_logits, group_ids, token_in_group
        else:
            group_probs = torch.softmax(group_logits, dim=-1)
            predicted_group_ids = torch.argmax(group_probs, dim=-1)

            token_logits = self.apply_linear(h, predicted_group_ids)

            token_probs = torch.softmax(token_logits, dim=-1)
            predicted_token_in_group = torch.argmax(token_probs, dim=-1)

            predicted_labels = predicted_group_ids * self.group_size + predicted_token_in_group

            return predicted_labels

    def training_step(self, batch, batch_idx):
        images, labels = batch  
        labels = labels.long().to(self.device)

        group_logits, token_logits, group_ids, token_in_group = self(images, labels)

        group_loss = self.group_loss_fn(group_logits, group_ids)
        token_loss = self.token_loss_fn(token_logits, token_in_group)

        loss = group_loss + token_loss

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('group_loss', group_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('token_loss', token_loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.long().to(self.device)

        group_logits, token_logits, group_ids, token_in_group = self(images, labels)

        group_loss = self.group_loss_fn(group_logits, group_ids)
        token_loss = self.token_loss_fn(token_logits, token_in_group)

        loss = group_loss + token_loss

        with torch.no_grad():
            predicted_group_ids = torch.argmax(group_logits, dim=-1)
            group_acc = (predicted_group_ids == group_ids).float().mean()

            x = self.conv_layers(images)
            x = x.view(x.size(0), -1)
            h = self.fc_layers(x)
            token_logits_pred = self.apply_linear(h, predicted_group_ids)
            predicted_token_in_group = torch.argmax(token_logits_pred, dim=-1)

            predicted_labels = predicted_group_ids * self.group_size + predicted_token_in_group
            acc = (predicted_labels == labels).float().mean()

        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_group_loss', group_loss, on_epoch=True, prog_bar=True)
        self.log('val_token_loss', token_loss, on_epoch=True, prog_bar=True)
        self.log('val_group_acc', group_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


def train():
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger
    import torch
    from data.datamodule import SyntheticImageDataModule

    model = ImageClassificationModel(vocab_size=184320, learning_rate=1e-3)

    data_dir = "enhanced_synthetic_dataset/"
    datamodule = SyntheticImageDataModule(data_dir, batch_size=64, num_workers=4)

    wandb_logger = WandbLogger(project='CNN_compressed', name='first_run')

    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',
        filename='CNN-comp-{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss',
        mode='min',
        save_last=True,
    )

    trainer = Trainer(
        max_epochs=40,
        devices=[1],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, datamodule)

def main():
    pass
if __name__ == "__main__":
    main()


