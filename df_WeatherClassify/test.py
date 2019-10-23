import os
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.m_dataloader import IMetDataset
import torch
import torch.nn.functional as F
import config
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    # 加载模型
    model = torch.load(config.save_bestmodel_path)
    model.cuda()

    # 加载数据
    SAMPLE_SUBMISSION_DF = pd.read_csv("data/submit_example.csv")

    test_augmentation = transforms.Compose([
        transforms.Resize(config.IMG_SIZE+32),
        transforms.RandomResizedCrop(config.IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config.mean, config.std)
    ])
    subm_dataset = IMetDataset(
        SAMPLE_SUBMISSION_DF,
        os.path.join(config.base, "Test"),
        n_classes=config.NUM_CLASSES,
        transforms=test_augmentation, is_test=True)
    subm_dataloader = DataLoader(subm_dataset,
                                 batch_size=32,
                                 shuffle=False, num_workers=2)
    # 预测

    def get_subm_answers(model, subm_dataloader, tta=11):

        model.eval()
        all_preds = np.zeros(
            shape=(
                SAMPLE_SUBMISSION_DF.shape[0],
                config.NUM_CLASSES))

        with torch.no_grad():
            for i in range(tta):
                print(f"tta {i+1}/{tta} epoch:")
                preds_cat = []
                for step, (inputs, labels) in enumerate(tqdm(subm_dataloader)):
                    inputs = inputs.cuda()
                    outputs = model(inputs)
                    preds_cat.append(outputs.data)

                preds_cat = F.softmax(
                    torch.cat(preds_cat), dim=1).cpu().numpy()
                all_preds = np.add(all_preds, preds_cat)
        return all_preds / tta, SAMPLE_SUBMISSION_DF.FileName.tolist()

    subm_preds, submids = get_subm_answers(model, subm_dataloader)

    # 预测的N类值
    pd.DataFrame(np.concatenate([np.array(submids).reshape(-1, 1), subm_preds], axis=1)).to_csv(
        "output/output__all_last.csv", index=None)

    # 预测标签
    subm_preds = np.argmax(subm_preds, axis=1)
    df_to_process = pd.DataFrame({"FileName": submids, "type": subm_preds})
    df_to_process["type"] = df_to_process["type"].apply(lambda x: x + 1)
    df_to_process.to_csv(f"output/submit_last.csv", index=None)

    print(df_to_process["type"].value_counts())
