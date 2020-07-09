import torch
import torch.nn as nn
import config
import torchvision
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, n_classes, input_shape=(3, 64, 192)):
        super(Model, self).__init__()
        self.input_shape = input_shape

        model_ft = torchvision.models.resnet50(pretrained=True)
        # print(model_ft)
        model_ft.layer3[0].downsample[0].stride = (2, 1)
        model_ft.layer3[0].conv2.stride = (2, 1)
        model_ft.layer4[0].downsample[0].stride = (2, 1)
        model_ft.layer4[0].conv2.stride = (2, 1)

        # for i, p in enumerate(list(model_ft.state_dict())):
        #     print(i, p)
        # for i, p in enumerate(model_ft.parameters()):
        #     if i < 66:
        #         p.requires_grad = False

        self.cnn = nn.Sequential(
            model_ft.conv1,
            model_ft.bn1,
            model_ft.relu,
            model_ft.maxpool,
            model_ft.layer1,
            model_ft.layer2,

            model_ft.layer3,
            # nn.Dropout2d(0.25),

            model_ft.layer4,
            # nn.Dropout2d(0.25),
        )

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