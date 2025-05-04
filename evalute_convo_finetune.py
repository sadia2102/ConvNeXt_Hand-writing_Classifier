import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np
import os

# Device
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# Define ConvNeXt classifier (same as train script)
class ConvNeXtClassifier(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)

# Load ConvNeXt encoder
convnext = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
convnext.classifier = nn.Identity()
convnext = convnext.to(device).eval()

# Create full model and load weights
model = ConvNeXtClassifier(convnext, num_classes=2).to(device)
model.load_state_dict(torch.load("finetuned_convnext_classifier.pth", map_location=device))
model.eval()

# Define transform (must match training)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load field dataset
field_dataset = ImageFolder("data/field", transform=transform)
field_loader = DataLoader(field_dataset, batch_size=32, shuffle=False)

class_names = field_dataset.classes
print("Class Mapping:", field_dataset.class_to_idx)

# Evaluation
all_preds = []
all_labels = []
misclassified = []

with torch.no_grad():
    for images, labels in tqdm(field_loader, desc="Evaluating on field data"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        for i in range(len(preds)):
            if preds[i] != labels[i]:
                path, _ = field_dataset.samples[len(all_labels) - len(preds) + i]
                misclassified.append(path)

# Metrics
print("\n✅ Field Accuracy: {:.2f}%".format(
    100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)))

print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

print("\n🧱 Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# Save misclassified file paths
if misclassified:
    with open("misclassified_field.txt", "w") as f:
        for path in misclassified:
            f.write(f"{path}\n")
    print(f"\n❌ Misclassified {len(misclassified)} images. Saved to misclassified_field.txt")
else:
    print("\n✅ No misclassifications!")
    
    # Save misclassified images 