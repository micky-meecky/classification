import torch.nn as nn
import pretrainedmodels
import torch
import timm
from train import getModelSize

def pretrainedmodelstest():
    # Load the model
    torch.hub.set_dir("downloaded_models")
    model_name = 'resnet34'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    model.eval()

    # Print the model architecture
    print(model)

    # Create a new model with a subset of layers from the original model
    new_model = nn.Sequential(*list(model.children())[:5], nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1), *list(model.children())[6:])

    print('new model:\n')
    # Print the new model architecture
    print(new_model)

def timmtest():

    # 更改下载路径
    torch.hub.set_dir("downloaded_models")
    # model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, in_chans=1, num_classes=2)
    print(model.get_classifier())

    print(getModelSize(model))




if __name__ == '__main__':
    # pretrainedmodelstest()
    timmtest()
