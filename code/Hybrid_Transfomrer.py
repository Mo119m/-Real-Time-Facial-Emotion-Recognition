import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Set seeds and backend optimizations
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = True

# Fix for ResNet Bottleneck inplace ReLU bug
from torchvision.models.resnet import Bottleneck
original_bottleneck_forward = Bottleneck.forward

def modified_bottleneck_forward(self, x):
    identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)
    out = self.conv3(out)
    out = self.bn3(out)
    if self.downsample is not None:
        identity = self.downsample(x)
    out = out + identity
    out = self.relu(out)
    return out

Bottleneck.forward = modified_bottleneck_forward

# Data transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Data directories
data_root = '/Users/aj/Desktop/ISYE604/Project/data'
data_root_agumented = '/Users/aj/Desktop/ISYE604/Project/604_augmented_data'
train_dir = os.path.join(data_root, 'train')
test_dir  = os.path.join(data_root_agumented, 'test')

# Create datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
test_dataset  = datasets.ImageFolder(test_dir, transform=test_transform)

num_classes = len(train_dataset.classes)
print(f"Detected {num_classes} classes: {train_dataset.classes}")

# CNN-ViT Hybrid Model
class CNN_ViT_Hybrid(nn.Module):
    def __init__(self, num_classes=num_classes, cnn_out_dim=2048):
        super().__init__()
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for m in base_model.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False
        self.cnn_backbone = nn.Sequential(*list(base_model.children())[:-2])
        self.pos_embedding = nn.Parameter(torch.zeros(1, 49, cnn_out_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_out_dim, nhead=8, dim_feedforward=4096,
            dropout=0.1, activation='relu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(cnn_out_dim),
            nn.Linear(cnn_out_dim, num_classes)
        )

    def forward(self, x):
        x = self.cnn_backbone(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.mlp_head(x)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True,
                              num_workers=os.cpu_count() // 2 or 4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False,
                             num_workers=os.cpu_count() // 2 or 4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = CNN_ViT_Hybrid().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_acc = 0.0
    epochs = 12
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            if torch.cuda.is_available():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        scheduler.step()
        train_loss = running_loss / len(train_loader)
        train_acc = correct / len(train_dataset)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
        val_acc = val_correct / len(test_dataset)
        val_accuracies.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Acc: {val_acc:.4f} | Best Acc: {best_acc:.4f}\n")

    # Load best model and evaluate
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    test_correct = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            test_correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = test_correct / len(test_dataset)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    # Plot curves
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_accuracies, label='Train Acc')
    plt.plot(epochs_range, val_accuracies, label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
    disp.plot(xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
