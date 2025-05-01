import torch
import torchvision

def get_model(num_classes=20, pretrained=True, freeze_base=True):
    if pretrained:
        model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
    else:
        model = torchvision.models.vit_b_16()

    if freeze_base:
        for name, param in model.named_parameters():
            if "head" not in name:
                # freeze all except the final layer
                param.requires_grad = False

    full_model = torch.nn.Sequential(
        model,
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.heads.head.out_features, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, num_classes)
    )
    return full_model