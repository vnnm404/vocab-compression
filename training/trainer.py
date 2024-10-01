import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.loggers import WandbLogger

from utils.tokenizer import gpt2_tokenizer
from data.tinystories import TinyStoriesDataModule, ChunkedDataset
from models.gpt2 import GPT2
from utils.config import config

L.seed_everything(0, workers=True)

from pytorch_lightning.callbacks import Callback
class StopHalfEpochCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx >= trainer.num_training_batches // 2:
            # trainer.validate(pl_module)
            val_loss = trainer.callback_metrics.get("val_loss")
            with open("val_changes.txt", "a") as f:
                f.write(f"{config.model.name}:{val_loss.item()}\n")
            trainer.should_stop = True

def train(model, data_module):
    wandb_logger = WandbLogger(name=config.training.run_name, project='vocab-compression-2')

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename=config.model.name + '-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )
    print("initializing trainer")
    trainer = L.Trainer(
        logger=wandb_logger,
        devices=[1],
        accelerator="auto",
        max_epochs=config.training.epochs,
        callbacks=[checkpoint_callback], #
        log_every_n_steps=20,
        gradient_clip_val=1.0,
        deterministic=True,
        val_check_interval=1000,
        accumulate_grad_batches=4,
        strategy='ddp_find_unused_parameters_true',
        # fast_dev_run=2400,
    )

    trainer.fit(model, data_module)
    return trainer
