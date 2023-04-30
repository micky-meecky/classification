import torch.nn as nn
import pretrainedmodels

# Load the model
model_name = 'resnet18'
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
model.eval()

# Print the model architecture
print(model)

# Create a new model with a subset of layers from the original model
new_model = nn.Sequential(*list(model.children())[:5], nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1), *list(model.children())[6:])

print('new model:\n')
# Print the new model architecture
print(new_model)
