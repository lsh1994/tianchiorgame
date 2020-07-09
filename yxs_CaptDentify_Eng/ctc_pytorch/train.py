from collections import OrderedDict
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader
import pandas as pd
from dataloader import CaptDataset
import numpy as np
import config
from model import TransformGeneral
from holocron import optim as hoptim
from Utilss import calc_acc


class LitPlants(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.net = TransformGeneral.Model(config.num_classes, input_shape=(3, config.H, config.W))

    def forward(self, x):
        y = self.net(x)
        return y

    def configure_optimizers(self):
        optimizer = hoptim.RAdam(filter(lambda p: p.requires_grad, self.parameters()), lr=config.lr, weight_decay=config.weight_decay)
        optimizer = hoptim.wrapper.Lookahead(optimizer)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
        return [optimizer], [scheduler]

    # def prepare_data(self):
    #     data = pd.read_csv("../data/train.csv")
    #     split = int(len(data) * 0.9)
    #     data = data.sample(frac=1, random_state=2020)  # 打乱
    #     data[:split].to_csv("../data/train_train.csv", index=False)
    #     data[split:].to_csv("../data/train_val.csv", index=False)

        ###############
    def train_dataloader(self):
        train_dataset = CaptDataset(
            "../data/train_train.csv",
            transforms=config.transforms_train, dir="../data/train/")
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=8
        )
        return train_loader

    def training_step(self, batch, batch_idx):  # mixup + labelsmooth+ circle_loss
        data, target, input_lengths, target_lengths = batch

        output = self(data)

        output_log_softmax = F.log_softmax(output, dim=-1)
        loss_val = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)

        acc = calc_acc(target, output, config.key_words)

        tqdm_dict = {'train_loss': loss_val, 'train_acc': acc}
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    ##########
    def val_dataloader(self):
        val_dataset = CaptDataset(
            "../data/train_val.csv",
            transforms=config.transforms_val, dir="../data/train/")
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=config.batch_size * 2,
            shuffle=False,
            num_workers=8
        )
        return val_loader

    def validation_step(self, batch, batch_idx):
        data, target, input_lengths, target_lengths = batch

        output = self(data)

        output_log_softmax = F.log_softmax(output, dim=-1)
        loss_val = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)

        acc = calc_acc(target, output, config.key_words)

        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc': acc,
        })

        return output

    def validation_epoch_end(self, outputs):
        tqdm_dict = {}
        for metric_name in ["val_loss", "val_acc"]:
            metric_total = 0
            for output in outputs:
                metric_value = output[metric_name]
                if self.trainer.use_dp or self.trainer.use_ddp2:
                    metric_value = torch.mean(metric_value)
                metric_total += metric_value
            tqdm_dict[metric_name] = metric_total / len(outputs)

        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': tqdm_dict["val_loss"]}
        return result


if __name__ == '__main__':

    if config.pre_weight:

        trainer = pl.Trainer(
            resume_from_checkpoint=config.pre_weight,
            default_save_path=f'output/{config.prefix}',
            gpus=1,accumulate_grad_batches=256//config.batch_size,
            amp_level="o1",  # GPU半精度
            max_epochs=config.epochs,
            checkpoint_callback=pl.callbacks.model_checkpoint.ModelCheckpoint(
                filepath=f'output/{config.prefix}.' + '{epoch}-{val_acc:.5f}',
                monitor='val_acc', mode='max', verbose=True, save_top_k=1, save_weights_only=True),
            # early_stop_callback=pl.callbacks.early_stopping.EarlyStopping(monitor='val_acc', patience=25, verbose=True),
            callbacks=[pl.callbacks.LearningRateLogger()]
        )
        net = LitPlants()
        trainer.fit(net)
    else:
        trainer = pl.Trainer(
            default_save_path=f'output/{config.prefix}',
            gpus=1,accumulate_grad_batches=256//config.batch_size,
            amp_level="o1",  # GPU半精度
            max_epochs=config.epochs,
            checkpoint_callback=pl.callbacks.model_checkpoint.ModelCheckpoint(
                filepath=f'output/{config.prefix}.' + '{epoch}-{val_acc:.5f}',
                monitor='val_acc', mode='max', verbose=True, save_top_k=1, save_weights_only=True),
            # early_stop_callback=pl.callbacks.early_stopping.EarlyStopping(monitor='val_acc', patience=25,verbose=True),
            callbacks=[pl.callbacks.LearningRateLogger()]
        )
        net = LitPlants()
        trainer.fit(net)
