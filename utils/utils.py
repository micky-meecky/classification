import shutil

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import pretrainedmodels
import pretrainedmodels.utils as utils
from mymodels.models import Net
from mymodels.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from mymodels.unetr import UNETR
from mymodels.Unet import UNet
from mymodels.ViT import ViT_model
import os
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

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
    if epoch % 20 == 0:  # 每20个epoch保存一次模型
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

def Device(model):
    if torch.cuda.is_available():
        device_ids = [i for i in range(torch.cuda.device_count())]
        device = f"cuda:{device_ids[0]}"
        print("\n Using GPU \n")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model.to(device)
    model = DataParallel(model, device_ids=device_ids) if torch.cuda.is_available() else model
    return model, device





def InitModel(modelname, use_pretrained: bool = False, class_num=3):
    model = None
    if use_pretrained:
        torch.hub.set_dir("./mymodels/downloaded_models")
        model = pretrainedmodels.__dict__[modelname](num_classes=1000, pretrained='imagenet')
        # modelname如果是以resnet开头的，则在他最后一层添加一个softmax激活
        if modelname.startswith('resnet'):
            # 替换输出层
            num_classes = 2
            model.last_linear = nn.Linear(model.last_linear.in_features, num_classes)
            model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if modelname.startswith('densenet'):
            # 替换输出层
            num_classes = 2
            model.last_linear = nn.Linear(4096, num_classes)
            model.features.conv0 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # 打印模型
            # print(model)
    else:
        if modelname == 'resnet18':
            model = resnet18(class_num)
        elif modelname == 'unetr':
            model = UNETR()
        elif modelname == 'unet':
            model = UNet(3, 1)
        elif modelname == 'Net':
            model = Net()
        elif modelname == 'resnet34':
            model = resnet34(class_num)
        elif modelname == 'resnet50':
            model = resnet50(class_num)
        elif modelname == 'resnet101':
            model = resnet101(class_num)
        elif modelname == 'resnet152':
            model = resnet152(class_num)
        elif modelname == 'ViT':
            model = ViT_model(256, 32, 10)   # 256是输入图片的大小，32是patch的大小，3是类别数
    return model


def update_lr(lr, optimizer):
    """Update the learning rate."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def LrDecay(lr_warm_epoch, lr_cos_epoch, lr, lr_low, optimizer):
    lr_sch = None
    if lr_warm_epoch != 0 and lr_cos_epoch == 0:
        update_lr(lr_low, optimizer)  # 使用warmup需要吧lr初始化为最小lr
        lr_sch = GradualWarmupScheduler(optimizer,
                                        multiplier=lr / lr_low,
                                        total_epoch=lr_warm_epoch,
                                        after_scheduler=None)
        print('use warmup lr sch')
    elif lr_warm_epoch == 0 and lr_cos_epoch != 0:
        lr_sch = lr_scheduler.CosineAnnealingLR(optimizer,
                                                lr_cos_epoch,
                                                eta_min=lr_low)
        print('use cos lr sch')
    elif lr_warm_epoch != 0 and lr_cos_epoch != 0:
        update_lr(lr_low, optimizer)  # 使用warmup需要吧lr初始化为最小lr
        scheduler_cos = lr_scheduler.CosineAnnealingLR(optimizer,
                                                       lr_cos_epoch,
                                                       eta_min=lr_low)
        lr_sch = GradualWarmupScheduler(optimizer,
                                        multiplier=lr / lr_low,
                                        total_epoch=lr_warm_epoch,
                                        after_scheduler=scheduler_cos)
        print('use warmup and cos lr sch')
    else:
        if lr_sch is None:
            print('use linear decay')

    return lr_sch


def AdjustLr(lr_sch, optimizer, epoch, lr_cos_epoch, lr_warm_epoch, num_epochs_decay, current_lr, lr_low, decay_step, decay_ratio):
    # 学习率策略部分 =========================
    # lr scha way 1:
    if lr_sch is not None:
        if (epoch + 1) <= (lr_cos_epoch + lr_warm_epoch):
            lr_sch.step()

    # lr scha way 2: Decay learning rate(如果使用方式1,则不使用此方式)
    if lr_sch is None:
        if ((epoch + 1) >= num_epochs_decay) and (
                (epoch + 1 - num_epochs_decay) % decay_step == 0):  # 根据设置衰减速率来更新lr
            if current_lr >= lr_low:
                lr = current_lr * decay_ratio
                # self.lr /= 100.0
                update_lr(lr, optimizer)
                print('Decay learning rate to lr: {}.'.format(lr))

    return lr_sch, optimizer


def GetCurrentLr(optimizer):
    current_lr = optimizer.param_groups[0]['lr']  # 获取当前lr
    return current_lr


def Mkdir(Dir):
    if not os.path.exists(Dir):
        os.makedirs(Dir)
    else:
        # 删掉原来的model文件
        shutil.rmtree(Dir)
        os.makedirs(Dir)


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)




