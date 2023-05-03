import torch
import pretrainedmodels.utils as utils
import torch.nn as nn
from torch.nn.parallel import DataParallel
import pretrainedmodels
import shutil
from mymodels import OpenDataSet

if __name__ == '__main__':

    # Load the model
    torch.hub.set_dir("downloaded_models")
    model_name = 'resnet18'
    new_model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    # new_model = nn.Sequential(*list(model.children())[:5], nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1), *list(model.children())[6:])
    print('new model:\n')
    # Print the new model architecture
    # print(new_model)


    if torch.cuda.is_available():
        device_ids = [i for i in range(torch.cuda.device_count())]
        device = f"cuda:{device_ids[0]}"
        print("\n Using GPU \n")
    else:
        device = torch.device("cpu")
        print("Using CPU")


    new_model.to(device)
    new_model = DataParallel(new_model, device_ids=device_ids) if torch.cuda.is_available() else new_model

    train_loader, test_loader = OpenDataSet.SelectDataSet('Cifar_10', 100)

    criteria = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)


    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        outputs = new_model(images)

        loss = criteria(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if (i+1) % 100 == 0:
            print(f"Epoch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")


