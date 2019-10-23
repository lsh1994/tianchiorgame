from efficientnet_pytorch import EfficientNet
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import config
import pretrainedmodels
import torchsummary


def get_efficientb3():
    model = EfficientNet.from_pretrained('efficientnet-b3')
    model._fc = torch.nn.Sequential(
        torch.nn.Dropout(),
        torch.nn.Linear(
            in_features=model._fc.in_features,
            out_features=config.NUM_CLASSES)
    )
    return model

def get_efficientb1():
    model = EfficientNet.from_pretrained('efficientnet-b1')
    model._fc = torch.nn.Sequential(
        torch.nn.Dropout(),
        torch.nn.Linear(
            in_features=model._fc.in_features,
            out_features=config.NUM_CLASSES)
    )
    return model


def get_densenet121():
    model = torchvision.models.densenet121(pretrained=True)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(),
        nn.Linear(
            model.classifier.in_features,
            config.NUM_CLASSES)
    )
    return model


def get_xception():
    model = pretrainedmodels.xception(num_classes=1000, pretrained="imagenet")
    model.last_linear = torch.nn.Sequential(
        torch.nn.Dropout(),
        nn.Linear(
            in_features=model.last_linear.in_features,
            out_features=config.NUM_CLASSES)
    )
    return model

def get_model(name):
    model = eval(f"get_{name}")()
    return model


if __name__ == '__main__':

    m = get_model("xception")
    print(m)
    m.cuda()
    torchsummary.summary(m, (3, 224, 224))

    pass
