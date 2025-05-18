import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image

class ECALayer(nn.Module):
    def __init__(self, channel, k_size=None):
        super().__init__()
        if k_size is None:
            t = int(abs((torch.log2(torch.tensor(channel, dtype=torch.float32)) / 2 + 1)))
            k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.view(x.size(0), 1, x.size(1))
        y = self.conv(y)
        y = self.sigmoid(y).view(x.size(0), x.size(1), 1, 1)
        return x * y.expand_as(x)

class OptiSA(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        backbone = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.maxpool,
            backbone.stage2,
            backbone.stage3,
            backbone.stage4,
            backbone.conv5,
        )
        in_feats = backbone.fc.in_features
        self.eca = ECALayer(channel=in_feats)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.eca(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OptiSA(num_classes=5).to(device)
model.load_state_dict(torch.load('best_opti_sa.pth', map_location=device))
model.eval()

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# Example prediction code (commented out to avoid errors when importing)
# img_path = '/content/sun.jpeg'
# img = Image.open(img_path).convert('RGB')
# x = val_tf(img).unsqueeze(0).to(device)
# with torch.no_grad():
#     logits = model(x)
#     probs  = torch.softmax(logits, dim=1)
#     pred_idx = probs.argmax(dim=1).item()
# classes = ['daisy','dandelion','rose','sunflower','tulip']
# print(f"Predicted class: {classes[pred_idx]}  (p={probs[0,pred_idx]:.4f})")
