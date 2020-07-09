import torch
import torch.nn as nn
import config
import torchvision
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import pretrainedmodels
import torchvision

class Model(nn.Module):
    def __init__(self, n_classes, input_shape, pretrained='imagenet'):
        super(Model, self).__init__()
        self.input_shape = input_shape

        # model_ft = pretrainedmodels.vgg16_bn(pretrained=pretrained)._features
        # model_ft = pretrainedmodels.resnet50(pretrained=pretrained)
        # model_ft = pretrainedmodels.densenet121(pretrained=pretrained)
        # model_ft = pretrainedmodels.se_resnet50(pretrained=pretrained)
        # model_ft = pretrainedmodels.se_resnext50_32x4d()
        # model_ft = EfficientNet.from_pretrained("efficientnet-b2")
        # model_ft = pretrainedmodels.xception(pretrained=pretrained)
        # model_ft = pretrainedmodels.dpn68b()

        # model_ft = pretrainedmodels.vgg19_bn()._features
        # model_ft = pretrainedmodels.resnet152(pretrained=pretrained)  #.features
        # model_ft = pretrainedmodels.dpn107()
        # model_ft = pretrainedmodels.senet154()
        # model_ft = pretrainedmodels.se_resnet152()
        # model_ft = pretrainedmodels.se_resnext101_32x4d()
        # model_ft = pretrainedmodels.resnext101_64x4d()
        model_ft = pretrainedmodels.densenet161()
        # model_ft = EfficientNet.from_pretrained("efficientnet-b7")  # .extract_features



        self.cnn = model_ft
        self.lstm = nn.LSTM(input_size=self.infer_features(), hidden_size=128, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(in_features=256, out_features=n_classes)

    def infer_features(self):
        x = torch.zeros((1,) + self.input_shape)
        x = self.cnn.features(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x.shape[1]

    def forward(self, x):
        x = self.cnn.features(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = F.dropout(x, p=0.5)
        x = self.fc(x)
        return x