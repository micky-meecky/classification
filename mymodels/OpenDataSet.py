import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


def Cifar_10(bs):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='../Data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='../Data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs,
                                             shuffle=False, num_workers=2)

    return trainloader, testloader


def Cifar_100(bs):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.CIFAR100(root='../Data', train=True,
                                                download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                                    shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='../Data', train=False,
                                                  download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs,
                                                    shuffle=False, num_workers=2)
    return trainloader, testloader

def Imagenet(bs):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.ImageNet(root='../Data', train=True,
                                                download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                                    shuffle=True, num_workers=2)
    testset = torchvision.datasets.ImageNet(root='../Data', train=False,
                                                    download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs,
                                                    shuffle=False, num_workers=2)
    return trainloader, testloader


def SelectDataSet(args: str = 'Cifar_10', bs: int = 4):
    if args == 'Cifar_10':
        return Cifar_10(bs)
    elif args == 'Cifar_100':
        return Cifar_100(bs)
    elif args == 'Imagenet':
        return Imagenet(bs)
    else:
        raise ValueError('No such dataset')




if __name__ == '__main__':
    trainloader, testloader = SelectDataSet('Cifar_10', 4)

    # 打印训练集的大小
    print(trainloader.dataset.data.shape)
    # 打印测试集的大小
    print(testloader.dataset.data.shape)
