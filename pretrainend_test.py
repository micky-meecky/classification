import torch
from torchvision import models
import pretrainedmodels.utils as utils
import torch.nn as nn
from torch.nn.parallel import DataParallel
import pretrainedmodels
import shutil
from mymodels import OpenDataSet

if __name__ == '__main__':
    # Load the model
    model_name = 'dense121'
    new_model = models.densenet121(pretrained=True)
    new_model.classifier = nn.Linear(1024, 10)

    if torch.cuda.is_available():
        device_ids = [i for i in range(torch.cuda.device_count())]
        device = f"cuda:{device_ids[0]}"
        print("\n Using GPU \n")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    new_model = DataParallel(new_model, device_ids=device_ids) if torch.cuda.is_available() else new_model
    new_model.to(device)

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


