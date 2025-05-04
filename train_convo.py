import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm  # For progress bars

# Choose best available device
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# Load ConvNeXt encoder and remove classification head
convnext = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
convnext.classifier = nn.Identity()
convnext = convnext.to(device).eval()

# Freeze encoder
for param in convnext.parameters():
    param.requires_grad = False

# Define classifier
class ConvNeXtClassifier(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        features = features.view(features.size(0), -1)  # Flatten from (B, 1024, 1, 1) to (B, 1024)
        return self.classifier(features)

# Transform: resize, center crop, normalize
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def train():
    # Load datasets
    train_dataset = ImageFolder("data/train", transform=transform)
    val_dataset = ImageFolder("data/val", transform=transform)
    test_dataset = ImageFolder("data/test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0)

    # Initialize model
    model = ConvNeXtClassifier(convnext, num_classes=len(train_dataset.classes)).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

    best_val_acc = 0.0
    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100
        train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total * 100
        val_loss /= len(val_loader)

        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_convnext_classifier.pth")
            print("  ✅ Saved best model.\n")

if __name__ == "__main__":
    train()


# End of training loop
# Use the best model for testing in evaluate.py
# Note: In a real scenario, you would also implement early stopping and learning rate scheduling