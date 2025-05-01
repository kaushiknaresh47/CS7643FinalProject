import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=20, pretrained=True, freeze_base=False):
    model = models.efficientnet_b0(pretrained=pretrained)
    
    if freeze_base:
        for param in model.features.parameters():
            param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    
    return model