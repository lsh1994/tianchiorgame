from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloader import CaptDataset
from train import LitPlants
from Utilss import decode_target, decode
import pandas as pd
import time
import numpy as np
import config
import os, glob
import torch
from scipy import stats
import multiprocessing
os.environ["PYTHONOPTIMIZE"]="1"

##################
def predict(weight_file, TTA, trans):
    print(f"filename:{weight_file},tta:{TTA}")

    model = LitPlants.load_from_checkpoint(weight_file).cuda()
    model.eval()

    # 开启TTA时候，注意修改数据增强
    test_dataset = CaptDataset("../data/test.csv", transforms=trans, is_test=True, dir="../data/test/")
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=128,
        shuffle=False, num_workers=4
    )

    ttar = []
    with torch.no_grad():
        for i in range(TTA):
            result = []
            for image, filename in tqdm(test_loader):
                output = model(image.cuda())
                output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
                for i in range(len(output_argmax)):
                    pred = decode(output_argmax[i], config.key_words)
                    result.append([filename[i].rsplit(".")[0], pred])
            result = pd.DataFrame(np.array(result), columns=["filename", "label"])
            ttar.append(result)

    r = ttar[0]
    for i in range(1, len(ttar)):
        r = pd.merge(r, ttar[i], on="filename")
    r = r.to_numpy()
    r = np.concatenate((r[:, 0].reshape(-1, 1), stats.mode(r[:, 1:], axis=1)[0]), axis=1)
    r = pd.DataFrame(r, columns=["filename", "label"])

    tm = str(time.strftime('%m%d%H%M'))
    wc = os.path.split(weight_file)[-1].rsplit(".", maxsplit=1)[0]
    r.to_csv(f"output/submit_{wc}_{tm}.csv", index=False, header=False)

# 多线程
def processing_predict(args):
    return predict(args[0],args[1],args[2])

def testimg2csv():
    fn = sorted(list(os.listdir("../data/test")), key=lambda x: int(x[:-4]))
    label = ["-" for i in range(len(fn))]
    r = pd.DataFrame(data={"filename": fn, "label": label})
    r.to_csv("../data/test.csv", index=False)



if __name__ == '__main__':
    # testimg2csv()

    # predict("output/densenet121.epoch=71-val_acc=0.70801.ckpt", 7, config.transforms_train)
    # predict("output/densenet161.epoch=65-val_acc=0.73809.ckpt", 7, config.transforms_train)
    # predict("output/dpn68b.epoch=57-val_acc=0.72721.ckpt", 7, config.transforms_train)

    # 同一个模型，不同权重
    pool = multiprocessing.Pool(5)
    print("多线程：",multiprocessing.cpu_count())
    pool.map(processing_predict,[(i,7,config.transforms_train) for i in glob.glob("output/*.ckpt")])

