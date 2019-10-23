# 常用库
from tqdm import tqdm
import pandas as pd
import os,warnings
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from tensorboardX import SummaryWriter
import torchvision

# 额外库
import torchsummary
from torchtoolbox.optimizer import Lookahead
from apex import amp

# 自定义库
from utils import auto_augment, m_transform, m_dataloader,radam
import config
import m_model

warnings.filterwarnings("ignore")

def ricap(images, targets, beta=0.3):

    # size of image
    I_x, I_y = images.size()[2:]

    # generate boundary position (w, h)
    w = int(np.round(I_x * np.random.beta(beta, beta)))
    h = int(np.round(I_y * np.random.beta(beta, beta)))
    w_ = [w, I_x - w, w, I_x - w]
    h_ = [h, h, I_y - h, I_y - h]

    # select four images
    cropped_images = {}
    c_ = {}
    W_ = {}
    for k in range(4):
        index = torch.randperm(images.size(0))
        x_k = np.random.randint(0, I_x - w_[k] + 1)
        y_k = np.random.randint(0, I_y - h_[k] + 1)
        cropped_images[k] = images[index][:, :,
                                          x_k:x_k + w_[k], y_k:y_k + h_[k]]
        c_[k] = targets[index]
        W_[k] = (w_[k] * h_[k]) / (I_x * I_y)

    # patch cropped images
    patched_images = torch.cat(
        (torch.cat((cropped_images[0], cropped_images[1]), 2),
         torch.cat((cropped_images[2], cropped_images[3]), 2)),
        3)

    targets = (c_, W_)
    return patched_images, targets


def ricap_criterion(outputs, c_, W_):
    loss = sum([W_[k] * F.cross_entropy(outputs, c_[k]) for k in range(4)])
    return loss


def train_one_epoch(model, train_loader, optimizer):

    model.train()

    running_loss = 0.0
    true_ans_list = []
    preds_cat = []

    for step, (inputs, labels) in enumerate(tqdm(train_loader)):
        inputs, labels = inputs.cuda(), labels.cuda()

        inputs, (c_, W_) = ricap(inputs, labels)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = ricap_criterion(outputs, c_, W_)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        true_ans_list.append(labels)
        preds_cat.append(preds)

    all_true_ans = torch.cat(true_ans_list)
    all_preds = torch.cat(preds_cat)

    epoch_loss = running_loss / all_true_ans.shape[0]
    epoch_acc = accuracy_score(all_true_ans.cpu(), all_preds.cpu())
    epoch_f1 = f1_score(all_true_ans.cpu(), all_preds.cpu(), average="macro")
    return epoch_loss, epoch_acc, epoch_f1


def validate(model, valid_loader):
    model.eval()

    running_loss = 0.0
    true_ans_list = []
    preds_cat = []

    with torch.no_grad():
        for step, (inputs, labels) in enumerate(tqdm(valid_loader)):
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = F.cross_entropy(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            true_ans_list.append(labels)
            preds_cat.append(preds)

        all_true_ans = torch.cat(true_ans_list)
        all_preds = torch.cat(preds_cat)

        epoch_loss = running_loss / all_true_ans.shape[0]
        epoch_acc = accuracy_score(all_true_ans.cpu(), all_preds.cpu())
        epoch_f1 = f1_score(
            all_true_ans.cpu(),
            all_preds.cpu(),
            average="macro")
        return epoch_loss, epoch_acc, epoch_f1


if __name__ == '__main__':

    train_df_all = pd.read_csv("data/eda_trainlabel.csv")

    train_df, val_df = train_test_split(train_df_all[["FileName", "type"]],
                                        test_size=config.test_size,
                                        shuffle=True
                                        )
    train_df.to_csv(f"output/{config.prefix}_train.csv", index=None)
    val_df.to_csv(f"output/{config.prefix}_val.csv", index=None)
    print(f"train length: {len(train_df)}, val length: {len(val_df)}")

    writer = SummaryWriter(
        log_dir="output/run/",filename_suffix=config.prefix)

    # 数据加载
    train_augmentation = transforms.Compose([
        transforms.Resize(config.IMG_SIZE+32),
        transforms.RandomResizedCrop(config.IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        auto_augment.AutoAugment(),
        m_transform.RandomErasing(0.5, [0.02, 0.4], 0.3, 20),
        transforms.ToTensor(),
        transforms.Normalize(config.mean, config.std)
    ])
    val_augmentation = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.mean, config.std)
    ])
    train_dataset = m_dataloader.IMetDataset(
        train_df,
        os.path.join(config.base, "Train"),
        n_classes=config.NUM_CLASSES,
        transforms=train_augmentation)
    val_dataset = m_dataloader.IMetDataset(
        val_df,
        os.path.join(config.base, "Train"),
        n_classes=config.NUM_CLASSES,
        transforms=val_augmentation)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size*2,
        shuffle=False,
        num_workers=6)

    writer.add_image(
        'Train Image',
        torchvision.utils.make_grid(
            iter(train_loader).__next__()[0],
            normalize=True,
            scale_each=True),
        0)
    writer.add_image('Valid Image',
                     torchvision.utils.make_grid(iter(val_loader).__next__()[0], normalize=True, scale_each=True), 0)

    # 模型

    model = m_model.get_model(config.name)

    optimizer = radam.RAdam(model.parameters(), lr=config.lr)
    optimizer = Lookahead(optimizer)

    sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10, min_lr=1e-7)

    model.cuda()
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    print(model)
    torchsummary.summary(model, (3, config.IMG_SIZE, config.IMG_SIZE))

    # 训练
    best_model_f1 = 0.0
    best_model = None
    best_model_ep = 0
    for epoch in range(1, config.N_EPOCHS + 1):

        print(f"Starting {epoch}/{config.N_EPOCHS} epoch...")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            optimizer)
        val_loss, val_acc, val_f1 = validate(
            model, val_loader)

        sheduler.step(train_loss)

        if val_f1 >= best_model_f1:
            print(
                f'better model with f1: {round(best_model_f1, 5)} -> f2: {round(val_f1,5)}')
            best_model = model
            best_model_f1 = val_f1
            best_model_ep = epoch
            torch.save(best_model, config.save_bestmodel_path)

        torch.save(model, config.save_lastmodel_path)

        print(
            f'train_loss:{train_loss}, train_acc:{train_acc}, train_f1:{train_f1}')
        print(
            f'val_loss:{val_loss}, val_acc:{val_acc}, val_f1:{val_f1}')
        print(f"lr: {optimizer.param_groups[0]['lr']}")

        writer.add_scalar("train/train_loss", train_loss, epoch)
        writer.add_scalar("train/train_acc", train_acc, epoch)
        writer.add_scalar("train/train_f1", train_f1, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

        writer.add_scalar("valid/val_loss", val_loss, epoch)
        writer.add_scalar("valid/val_acc", val_acc, epoch )
        writer.add_scalar("valid/val_f1", val_f1, epoch )

        print()

    print(
        f'Best model with f1: {round(best_model_f1, 5)} on epoch {best_model_ep}')
