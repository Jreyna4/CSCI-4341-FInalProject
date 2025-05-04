import torch.nn as nn
from torchvision import models

def get_model(model_name="chexnet", num_classes=2):
    if model_name == "chexnet":
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "efficientnet":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Invalid model name. Use 'chexnet' or 'efficientnet'.")
    return model
