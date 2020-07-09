import torch
import torch.nn as nn
import config
import torchvision
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, n_classes, input_shape=(3, 64, 192)):
        super(Model, self).__init__()
        self.input_shape = input_shape

        model_ft = torchvision.models.densenet121(pretrained=True)
        model_ft.features.transition2.pool.kernel_size = (2,1)
        model_ft.features.transition2.pool.stride = (2,1)
        model_ft.features.transition3.pool.kernel_size = (2, 1)
        model_ft.features.transition3.pool.stride = (2, 1)

        self.cnn = model_ft.features
        self.lstm = nn.LSTM(input_size=self.infer_features(), hidden_size=128, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(in_features=256, out_features=n_classes)

    def infer_features(self):
        x = torch.zeros((1,) + self.input_shape)
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x.shape[1]

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = F.dropout(x, p=0.5)
        x = self.fc(x)
        return x
