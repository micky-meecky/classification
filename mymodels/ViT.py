import torch
from vit_pytorch import ViT


def ViT_model(image_size=256, patch_size=16, num_classes=3):
    v = ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=1024,
        depth=12,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    return v


if __name__ == "__main__":

    img = torch.randn(1, 3, 256, 256)

    model = ViT_model()

    preds = model(img)  # (1, 1000)

    print(preds.shape)
    # 输出类别
    print(torch.argmax(preds))
