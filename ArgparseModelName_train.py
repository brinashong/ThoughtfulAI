import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import os
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from torchmetrics import Accuracy
from torchvision import transforms, datasets
from torchvision import models
import wandb
from accelerate import Accelerator

accelerator = Accelerator(log_with="wandb")

torch.set_float32_matmul_precision("medium")
data_dir = '/home/users/nus/e0008091/scratch/ThoughtfulAI/data/'
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'test']}
class_names = image_datasets['train'].classes
class ImageClassifierModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = data_transforms
        self.image_datasets = image_datasets
        self.class_names = image_datasets['train'].classes
        self.num_classes = len(class_names)
        set_train_full = self.image_datasets['train']
        self.set_train, self.set_val = random_split(set_train_full, [0.9, 0.1])
        self.set_test = self.image_datasets['test']
    def train_dataloader(self):
        return DataLoader(self.set_train, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    def val_dataloader(self):
        return DataLoader(self.set_val, batch_size=self.batch_size, num_workers=4)
    def test_dataloader(self):
        return DataLoader(self.set_test, batch_size=self.batch_size, num_workers=4)
class ImagePredictionLogger(pl.callbacks.Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples
    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        trainer.logger.experiment.log({
            "examples":[wandb.Image(x, caption=f"Pred:{class_names[pred] if class_names is not None else pred}, Label:{class_names[y] if class_names is not None else y}")
                           for x, pred, y in zip(val_imgs[:self.num_samples],
                                                 preds[:self.num_samples],
                                                 val_labels[:self.num_samples])]
            })
class LitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=2e-4, transfer=True, model_name=None):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.dim = input_shape
        self.num_classes = num_classes
        self.feature_extractor = models.get_model(model_name, pretrained=transfer)
        if transfer:
            self.feature_extractor.eval()
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        n_sizes = self._get_conv_output(input_shape)
        self.classifier = nn.Linear(n_sizes, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy('multiclass', num_classes=num_classes)
    def _get_conv_output(self, shape):
        batch_size = 1
        tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._forward_features(tmp_input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
    def _forward_features(self, x):
        x = self.feature_extractor(x)
        return x
    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        acc = self.accuracy(outputs, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        acc = self.accuracy(outputs, targets)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
        return loss
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        acc = self.accuracy(outputs, targets)
        self.log("test/loss", loss)
        self.log("test/acc", acc)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


import argparse

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_name', type=str, help='name of model to get from torchvision')
    args = parser.parse_args()

    dm = ImageClassifierModule(batch_size=32, data_dir=data_dir)
    val_samples = next(iter(dm.val_dataloader()))
    model = LitModel((3, 224, 224), dm.num_classes, learning_rate=2e-4, transfer=True, model_name=args.model_name)
    accelerator.init_trackers("Retail Image Classification 80 Percent Crop", config={}, init_kwargs={"wandb":{"name":args.model_name}})
    wandb_logger = WandbLogger(project='Retail Image Classification 80 Percent Crop', job_type='train', log_model="all")
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val/loss")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="./checkpoints/", save_top_k=1, monitor="val/loss")
    trainer = pl.Trainer(max_epochs=100, accelerator="auto", logger=wandb_logger, callbacks=[early_stop_callback, ImagePredictionLogger(val_samples), checkpoint_callback])
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)
    wandb.finish()

if __name__ == "__main__":
    main()