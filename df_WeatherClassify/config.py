
IMG_SIZE = 224
NUM_CLASSES = 9

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

name_dict = {
    i: j for i, j in zip(
        range(
            1, 10), [
                "雨凇", "雾凇", "雾霾", "霜", "露", "结冰", "降雨", "降雪", "冰雹"])}

base = r"D:\Game_lsh\weather"

# 训练参数
batch_size = 64
N_EPOCHS = 200
test_size = 0.1
lr = 0.001


name = "efficientb3"  # efficientb3,densenet121,xception

prefix = f"{name}_1022"
save_bestmodel_path = f"output/{prefix}_best_model.pt"
save_lastmodel_path = f"output/{prefix}_last_model.pt"


