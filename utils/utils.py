
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import pretrainedmodels
import pretrainedmodels.utils as utils
from mymodels.models import Net
from mymodels.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from mymodels.unetr import UNETR
from mymodels.Unet import UNet

def Device(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("\n Using GPU \n")
        model = DataParallel(model)
        model.to(device)
    else:
        print("Using CPU")
        # model = DataParallel(model)
        model.to(device)
    return device


def LossExport(cls_running_loss, seg_running_loss, running_loss, datas, writer, epoch, _have_segtask):
    # 计算平均epoch_cls_loss
    epoch_cls_loss = cls_running_loss / len(datas)  # len(train_loader)是batch的个数---------------
    writer.add_scalars('Loss', {'epoch_cls_loss': epoch_cls_loss}, epoch)
    print('epoch_cls_loss = ', epoch_cls_loss)
    if _have_segtask:
        epoch_seg_loss = seg_running_loss / len(datas)  # len(train_loader)是batch的个数-----------------
        writer.add_scalars('Loss', {'epoch_seg_loss': epoch_seg_loss}, epoch)
        print('epoch_seg_loss = ', epoch_seg_loss)
    epoch_loss = running_loss / len(datas)  # len(train_loader)是batch的个数-----------------
    writer.add_scalars('Loss', {'epoch_loss': epoch_loss}, epoch)
    print('epoch_loss = ', epoch_loss, '\n')

    return epoch_cls_loss


def SaveModel(model, epoch, epoch_cls_loss, save_model_dir):
    temploss = 100.0
    if epoch % 10 == 0:  # 每10个epoch保存一次模型
        torch.save(model.state_dict(), save_model_dir + '/model' + str(epoch) + '.pth')
        print('save model')
    if temploss > epoch_cls_loss:
        temploss = epoch_cls_loss
        torch.save(model.state_dict(), save_model_dir + '/miniclsloss' + '.pth')
        print('save model，and epoch_cls_loss = ', temploss, '\n')


def WriteIntoTxt(txtcontent, txtdir):
    with open(txtdir, 'w') as f:
        for i in txtcontent:
            f.write(i + '\n')


def InitModel(modelname, use_pretrained: bool = False):
    model = None
    if use_pretrained:
        torch.hub.set_dir("./mymodels/downloaded_models")
        model = pretrainedmodels.__dict__[modelname](num_classes=1000, pretrained='imagenet')
        # modelname如果是以resnet开头的，则在他最后一层添加一个softmax激活
        if modelname.startswith('resnet'):
            # 替换输出层
            num_classes = 3
            model.last_linear = nn.Linear(model.last_linear.in_features, num_classes)
            model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            pass
    else:
        if modelname == 'resnet18':
            model = resnet18()
        elif modelname == 'unetr':
            model = UNETR()
        elif modelname == 'unet':
            model = UNet(1, 1)
        elif modelname == 'Net':
            model = Net()
        elif modelname == 'resnet34':
            model = resnet34()
        elif modelname == 'resnet50':
            model = resnet50()
        elif modelname == 'resnet101':
            model = resnet101()
    return model

