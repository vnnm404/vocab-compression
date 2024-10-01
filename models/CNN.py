import torch
import torch.nn as nn
import pytorch_lightning as pl
from data.datamodule import SyntheticImageDataModule

class ImageClassificationModel(pl.LightningModule):
    def __init__(self, vocab_size=184320, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # Define CNN layers
        self.conv_layers = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),  # Output: 32 x 100 x 100
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32 x 50 x 50
            
            # Second convolutional layer
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # Output: 64 x 50 x 50
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 x 25 x 25
            
            # Third convolutional layer
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # Output: 128 x 25 x 25
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 128 x 12 x 12
        )

        # Calculate the size after flattening
        self.flatten_size = 128 * 12 * 12  # 18,432 features

        # Define fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Linear(512, vocab_size)  # Output size: vocab_size (92,160)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        # Pass through convolutional layers
        x = self.conv_layers(x)
        # Flatten the output
        x = x.view(x.size(0), -1)
        # Pass through fully connected layers
        x = self.fc_layers(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch  # images: (batch_size, 3, 100, 100), labels: (batch_size)
        labels = labels.long().to(self.device)  # Ensure labels are LongTensor and on the correct device
        logits = self(images)  # logits: (batch_size, vocab_size)
        loss = self.criterion(logits, labels)
        # Log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.long().to(self.device)
        logits = self(images)
        loss = self.criterion(logits, labels)
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        # Log loss and accuracy
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
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
        filename='CNN-{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss',
        mode='min',
        save_last=True,
    )

    trainer = Trainer(
        max_epochs=40,
        # devices=[1],
        accelerator='cpu' if torch.cuda.is_available() else 'cpu',
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, datamodule)

def main():
    pass

if __name__ == "__main__":
    main()
