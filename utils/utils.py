import shutil

import timm
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import pretrainedmodels
from mymodels.models import Net
from mymodels.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from mymodels.unetr import UNETR, UNETRcls, UNETRseg, UNETRclsz12, UNETRclstoken
from mymodels.Unet import UNet, UNetcls, UNetseg, Res101UNet, AgUNet, AgUNetseg, ResUNet, InDilatedUNet, SideUNet, \
    CasDilatedUNet, M_UNet_seg
from mymodels.testsmp import UNet as ResUnet
from mymodels.ViT import ViT_model, ViTseg, ViTcls
from mymodels.swinunet import SwinUnet
from mymodels.UnetPP import UNetPlusPlusSeg, DSUNetPlusPlusSeg
from mymodels.MTunet import MTUNet
from mymodels.CBAMUnet import AgCBAMUNet, AgCBAMPixViTUNet, CBAMUNet, SideCBAMUNet, SideAgCBAMUNet, PixViTUNet, \
    CBAMPixViTUNet, SideCBAMPixViTUNet, SideAgCBAMPixViTUNet
from mymodels.Transunet.Transunet import TransUNet
from mymodels.swinViT import swin_base_patch4_window7_224, Swinseg, Swincls
import os
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from torchvision.models.googlenet import GoogLeNet
import torch.nn.init as init
from collections import OrderedDict


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

    return epoch_cls_loss, epoch_loss


def SaveModel(model, epoch, epoch_loss, save_model_dir):
    temploss = 100.0
    # if epoch % 1007 == 0:  # 每20个epoch保存一次模型
    #     torch.save(model.state_dict(), save_model_dir + '/model' + str(epoch) + '.pth')
    #     print('save model，and epoch_loss = ', epoch_loss, '\n')
    # if temploss > epoch_loss:
    #     temploss = epoch_loss
    #     torch.save(model.state_dict(), save_model_dir + '/miniloss' + '.pth')
    #     print('save miniloss model，and epoch_loss = ', temploss, '\n')
    pass


def WriteIntoTxt(txtcontent, txtdir):
    with open(txtdir, 'w') as f:
        for i in txtcontent:
            f.write(i + '\n')


def Device(model):
    if torch.cuda.is_available():
        # device_ids = [i for i in range(torch.cuda.device_count())]
        if torch.cuda.device_count() > 1:
            # 设置为使用1,2,3号GPU
            device_ids = [3, 0]  # 使用的是3个GPU，1,2,3号
            print("\n Using GPU device: {}".format(device_ids))
        else:
            device_ids = [0]  # 使用的是1个GPU，0号
            print("\n Using GPU device: {}".format(device_ids[0]))
        device = f"cuda:{device_ids[0]}"
        model.to(device)
        model = DataParallel(model, device_ids=device_ids) if torch.cuda.is_available() else model
    else:
        device = torch.device("cpu")
        print("Using CPU")
        model.to(device)
    return model, device


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, model, device):
        super(MultiTaskLossWrapper, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(2))
        # 初始化为0.2326621264219284, -1.3984906673431396
        # self.log_vars = nn.Parameter(torch.tensor([0., 0.]))
        self.model = model
        self.device = device

    def forward(self,
                cls_out,
                SR_flat,
                targets4v,
                GT_flat,
                criterion_seg,
                criterion_cls):
        seg_loss = criterion_seg(SR_flat, GT_flat, self.device, self.log_vars[0])
        cls_loss = criterion_cls(cls_out, targets4v, self.log_vars[1])
        loss = seg_loss + cls_loss
        # 求平均
        num = GT_flat.size(0)
        seg_loss = seg_loss / num
        cls_loss = cls_loss / num
        loss = loss / num / 2  # 除以2是因为有两个loss
        return seg_loss, cls_loss, loss, self.log_vars


class CustomGoogLeNet(GoogLeNet):
    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False, init_weights=True,
                 pretrained=False):
        super().__init__(num_classes=num_classes, aux_logits=aux_logits, transform_input=transform_input,
                         init_weights=init_weights)
        if pretrained:
            pretrained_model = models.googlenet(pretrained=True)
            state_dict = pretrained_model.state_dict()

            # Remove keys that do not match
            state_dict.pop("conv1.conv.weight")
            state_dict.pop("fc.weight")
            state_dict.pop("fc.bias")

            self.load_state_dict(state_dict, strict=False)

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform_input:
            x = x.clone()
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x = x_ch0
        return x


def InitModel(modelname, use_pretrained: bool = False, class_num=3, _have_segtask=False, _only_segtask=False,
              channel=3, img_size=256):
    model = None
    if use_pretrained:
        if modelname == 'res101UNetsmp':
            model = ResUnet(encoder_name='resnet101', in_channels=channel, oseg=True)
        if modelname == 'res101UNet':
            model = Res101UNet(channel, 1)
        if modelname == 'preswin_vit_segc':
            # 创建新模型实例
            model = Swinseg(oseg=True, channel=channel)

            # 加载预训练权重
            model_path = os.path.join(os.path.expanduser("~"),
                                      ".cache/torch/hub/checkpoints/swin_base_patch4_window7_224.pth")
            pretrained_weights = torch.load(model_path, map_location=torch.device('cpu'))
            pretrained_weights = pretrained_weights['model']

            # 创建有序字典以保存成功加载的参数
            new_state_dict = OrderedDict()

            # 记录成功加载的键
            successfully_loaded_keys = []

            # 迭代预训练权重，并检查是否与新模型的编码器匹配
            for k, v in pretrained_weights.items():
                if k in model.encoder.state_dict() and model.encoder.state_dict()[k].shape == v.shape:
                    new_state_dict[k] = v
                    successfully_loaded_keys.append(k)

            # 加载成功匹配的参数
            model.encoder.load_state_dict(new_state_dict, strict=False)

            # 冻结成功加载的参数
            # for name, param in model.encoder.named_parameters():
            #     if name in successfully_loaded_keys:
            #         param.requires_grad = False
        if modelname == 'preswin_vit_cls':
            # 创建新模型实例
            model = Swincls(oseg=False, task='cls', channel=channel)

            # 加载预训练权重
            model_path = os.path.join(os.path.expanduser("~"),
                                      ".cache/torch/hub/checkpoints/swin_base_patch4_window7_224.pth")
            pretrained_weights = torch.load(model_path, map_location=torch.device('cpu'))
            pretrained_weights = pretrained_weights['model']

            # 创建有序字典以保存成功加载的参数
            new_state_dict = OrderedDict()

            # 记录成功加载的键
            successfully_loaded_keys = []

            # 迭代预训练权重，并检查是否与新模型的编码器匹配
            for k, v in pretrained_weights.items():
                if k in model.encoder.state_dict() and model.encoder.state_dict()[k].shape == v.shape:
                    new_state_dict[k] = v
                    successfully_loaded_keys.append(k)

            # 加载成功匹配的参数
            model.encoder.load_state_dict(new_state_dict, strict=False)

            # 冻结成功加载的参数
            # for name, param in model.encoder.named_parameters():
            #     if name in successfully_loaded_keys:
            #         param.requires_grad = False
        if modelname.startswith('resnet101'):
            model = models.resnet101(pretrained=True)
            # 替换输出层
            num_classes = class_num
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            # 修改输入通道数
            model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if modelname.startswith('densenet121'):
            # 替换输出层
            model = models.densenet121(pretrained=True)
            num_classes = class_num
            # 修改输出类别
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            # 修改输入通道数
            model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # 打印模型
            # print(model)
        if modelname.startswith('xception'):
            model = pretrainedmodels.xception(num_classes=1000, pretrained='imagenet')
            # 更改最后一层的输出分类数
            model.last_linear = nn.Linear(model.last_linear.in_features, class_num)
            # 更改输入通道数为1
            model.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False)
        if modelname.startswith('efficientnet'):
            torch.hub.set_dir("./mymodels/downloaded_models")
            model = timm.create_model('efficientnet_b7', pretrained=True, in_chans=1, num_classes=1)
        if modelname.startswith('googlenet'):  # 有很多问题
            model = CustomGoogLeNet(pretrained=True)
            # 替换输出层
            num_classes = class_num
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            # 修改输入通道数
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if modelname.startswith('swin_transformer'):
            torch.hub.set_dir("./mymodels/downloaded_models")
            model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, in_chans=1, num_classes=2)
        if modelname.startswith('resnet18'):
            model = models.resnet18(pretrained=True)
            # 替换输出层
            num_classes = class_num
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            # 修改输入通道数
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if modelname.startswith('mobilenetv3'):
            torch.hub.set_dir("./mymodels/downloaded_models")
            model = timm.create_model('mobilenetv3_large_100', pretrained=True, in_chans=1, num_classes=1)
        if modelname.startswith('resnest14d'):
            torch.hub.set_dir("./mymodels/downloaded_models")
            model = timm.create_model('resnest14d', pretrained=True, in_chans=1, num_classes=1)
        if modelname.startswith('skresnet18'):
            torch.hub.set_dir("./mymodels/downloaded_models")
            model = timm.create_model('skresnet18', pretrained=True, in_chans=1, num_classes=1)
        if modelname.startswith('vgg16_bn'):
            torch.hub.set_dir("./mymodels/downloaded_models")
            model = timm.create_model('vgg16_bn', pretrained=True, in_chans=1, num_classes=1)

    else:
        if modelname == 'resnet18':
            model = resnet18(class_num)
        elif modelname == 'unetr':
            if _only_segtask:
                model = UNETRseg()
            else:
                if _have_segtask:
                    model = UNETR()
                else:
                    model = UNETRcls()
        elif modelname == 'unetrclsz12':
            model = UNETRclsz12()
        elif modelname == 'unetrclstoken':
            model = UNETRclstoken()
        elif modelname == 'swin-vit':
            model = swin_base_patch4_window7_224(num_classes=1)
        elif modelname == 'swin_vit_segc':
            model = Swinseg()
        elif modelname == 'unet':
            if _only_segtask:
                model = UNetseg(channel, 1)
            else:
                if _have_segtask:
                    model = UNet(channel, 1)
                else:
                    model = UNetcls(channel, 1)
        elif modelname == 'convunet':
            if _only_segtask:
                model = UNetseg(channel, 1, 'convpool')
            else:
                if _have_segtask:
                    model = UNet(channel, 1, 'convpool')
                else:
                    model = UNetcls(channel, 1, 'convpool')

        elif modelname == 'agconvunet':
            model = AgUNet(channel, 1, 'convpool')
        elif modelname == 'agunet':
            model = AgUNet(channel, 1)
        elif modelname == 'agunetseg':
            model = AgUNetseg(channel, 1)
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
            model = ViT_model(512, 16, 1)  # 256是输入图片的大小，32是patch的大小，3是类别数
        elif modelname == 'ViTseg':
            model = ViTseg()
        elif modelname == 'ViTcls':
            model = ViTcls()
        elif modelname == 'swin_unet':
            model = SwinUnet()
        elif modelname == 'MTunet':
            model = MTUNet()
        elif modelname == 'Transunet':
            img_size = 256
            mlp_dim = img_size * 4
            model = TransUNet(
                img_dim=img_size,
                in_channels=channel,
                out_channels=img_size,
                head_num=4,
                mlp_dim=mlp_dim,
                block_num=8,
                patch_dim=16,
                class_num=1,
                task='segcls',
            )
        elif modelname == 'AgCBAMUNet':
            model = AgCBAMUNet(channel, 1)
        elif modelname == 'AgCBAMPixViTUNet':
            model = AgCBAMPixViTUNet(img_size, channel, 1)
        elif modelname == 'CBAMPixViTUNet':
            model = CBAMPixViTUNet(img_size, channel, 1)
        elif modelname == 'SideCBAMPixViTUNet':
            model = SideCBAMPixViTUNet(img_size, channel, 1)
        elif modelname == 'SideAgCBAMPixViTUNet':
            model = SideAgCBAMPixViTUNet(img_size, channel, 1)
        elif modelname == 'PixViTUNet':
            model = PixViTUNet(img_size, channel, 1)
        elif modelname == 'CBAMUNet':
            model = CBAMUNet(channel, 1)
        elif modelname == 'SideCBAMUNet':
            model = SideCBAMUNet(channel, 1)
        elif modelname == 'SideAgCBAMUNet':
            model = SideAgCBAMUNet(channel, 1)
        elif modelname == 'ResUNet':
            model = ResUNet(channel, 1, 'convpool')
        elif modelname == 'InDilatedUNet':
            model = InDilatedUNet(channel, 1, 'maxpool')
        elif modelname == 'CasDilatedUNet':
            model = CasDilatedUNet(channel, 1, 'maxpool')
        elif modelname == 'SideUNet':
            model = SideUNet(channel, 1, 'NoSE')
        elif modelname == 'SideconvUNet':
            model = SideUNet(channel, 1, 'NoSE', 'convpool')
        elif modelname == 'SideSEUNet':
            model = SideUNet(channel, 1, 'SE')
        elif modelname == 'M_UNet_seg':
            model = M_UNet_seg(channel, 1)
        elif modelname == 'UNetPlusPlusSeg':
            model = UNetPlusPlusSeg(channel, 1)
        elif modelname == 'DSUNetPlusPlusSeg':
            model = DSUNetPlusPlusSeg(channel, 1)
        else:
            assert False, 'model name error'
    return model


def check_grad(model):
    i = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            # print(f'Parameter {name} does not require grad')
            i += 1

    print(f'There are {i} / {len(list(model.parameters()))} layers that do not require grad')


def GetTPFP(predicted, targets4):
    predicted = predicted.squeeze().long()
    tp = torch.sum((predicted == 0) & (targets4 == 0)).item()
    fp = torch.sum((predicted == 0) & (targets4 == 1)).item()
    tn = torch.sum((predicted == 1) & (targets4 == 1)).item()
    fn = torch.sum((predicted == 1) & (targets4 == 0)).item()
    return tp, fp, tn, fn


def PrintTrainInfo(_only_segtask, epoch, epoch_num, epoch_tp, epoch_fp, epoch_tn, epoch_fn, num_zero, num_one,
                   tmp_pre, tmp_tar, writer, Iter):
    precision, recall, f1_score, acc = 0, 0, 0, 0
    if not _only_segtask:
        # 计算精确率（Precision）、召回率（Recall）和F1分数
        precision = epoch_tp / (epoch_tp + epoch_fp) if epoch_tp + epoch_fp > 0 else 0
        recall = epoch_tp / (epoch_tp + epoch_fn) if epoch_tp + epoch_fn > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        acc = (epoch_tp + epoch_tn) / (epoch_tp + epoch_tn + epoch_fp + epoch_fn)
        print(
            f'Epoch: {epoch + 1}/{epoch_num},'
            f' Precision: {precision:.4f},'
            f' Recall: {recall:.4f},'
            f' F1-score: {f1_score:.4f},'
            f' Accuracy: {acc:.4f}'
        )

        # 打印num_zero和num_one
        print('num_zero: ', num_zero)
        print('num_one: ', num_one)
        print('\npredicted = ', tmp_pre)
        print('targets4 = ', tmp_tar)
        print('epoch_tp = ', epoch_tp, 'epoch_fp = ', epoch_fp, 'epoch_tn = ', epoch_tn, 'epoch_fn = ', epoch_fn)

        if epoch % 3 == 0:
            writer.add_scalars('Accuracy', {'train acc': acc}, Iter)
            writer.add_scalars('precision', {'train precision': precision}, Iter)
            writer.add_scalars('recall', {'train recall': recall}, Iter)
            writer.add_scalars('f1_score', {'train f1_score': f1_score}, Iter)

    return precision, recall, f1_score, acc


def update_lr(lr, optimizer):
    """Update the learning rate."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def LrDecay(lr_warm_epoch, lr_cos_epoch, lr, lr_low, optimizer):
    lr_sch = None
    if lr_warm_epoch != 0 and lr_cos_epoch == 0:
        update_lr(lr_low, optimizer)
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
        update_lr(lr_low, optimizer)
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


def AdjustLr(lr_sch, optimizer, epoch, lr_cos_epoch, lr_warm_epoch, num_epochs_decay, current_lr, lr_low, decay_step,
             decay_ratio):
    # 学习率策略部分 =========================
    # lr scha way 1:
    if lr_sch is not None:
        # if current_lr > lr_low and (epoch + 1) > lr_cos_epoch:
        #     lr = current_lr * decay_ratio
        #     update_lr(lr, optimizer)
        if (epoch + 1) <= (lr_cos_epoch + lr_warm_epoch):
            lr_sch.step()

    # lr scha way 2: Decay learning rate(如果使用方式1,则不使用此方式)
    if lr_sch is None:
        if ((epoch + 1) >= num_epochs_decay) and (
                (epoch + 1 - num_epochs_decay) % decay_step == 0):  # 根据设置衰减速率来更新lr
            if current_lr >= lr_low:
                lr = current_lr * decay_ratio
                update_lr(lr, optimizer)
                print('Decay learning rate to lr: {}.'.format(lr))

    return lr_sch, optimizer


# 初始化网络权重
def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # 随机初始化
            init.uniform_(m.weight)
            # Xavier 初始化
            init.xavier_uniform_(m.weight)
            # He 初始化
            init.kaiming_uniform_(m.weight)
            # 偏置初始化
            if m.bias is not None:
                init.zeros_(m.bias)


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


if __name__ == '__main__':
    print('main')
    # 测试一下学习率衰减的方式，以及warmup的方式
    lr = 1e-5  # 初始学习率
    lr_low = 1e-15  # 最低学习率
    decay_step = 10  # 每decay_step个epoch衰减一次
    decay_ratio = 0.951  # 每decay_step个epoch衰减一次，衰减比例为decay_ratio
    num_epochs_decay = 40  # 从第几个epoch开始衰减
    lr_warm_epoch = 6  # warmup的epoch数
    lr_cos_epoch = 790  # cos衰减的epoch数
    optimizer = torch.optim.Adam([torch.randn(3, 3)], lr=lr)
    lr_sch = LrDecay(lr_warm_epoch, lr_cos_epoch, lr, lr_low, optimizer)
    # 初始化一个学习率列表
    lr_list = []
    for epoch in range(800):
        lr_sch, optimizer = AdjustLr(lr_sch, optimizer, epoch, lr_cos_epoch, lr_warm_epoch, num_epochs_decay,
                                     GetCurrentLr(optimizer), lr_low, decay_step, decay_ratio)
        # 将学习率记录在一个列表lr_list中
        lr_list.append(GetCurrentLr(optimizer))
        print('epoch: {}, lr: {}'.format(epoch, GetCurrentLr(optimizer)))
