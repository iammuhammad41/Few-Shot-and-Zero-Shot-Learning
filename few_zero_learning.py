import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_few_shot_subset(dataset, shots: int, num_classes: int = 10):
    """
    Sample `shots` examples per class from `dataset`.
    Returns a Subset of the original dataset.
    """
    labels = np.array(dataset.targets)
    indices = []
    for c in range(num_classes):
        c_idxs = np.where(labels == c)[0]
        selected = np.random.choice(c_idxs, shots, replace=False)
        indices.extend(selected.tolist())
    return Subset(dataset, indices)

def train_linear_head(model, train_loader, test_loader, device, epochs=20, lr=1e-3):
    """
    Fine-tune only the final linear layer of `model` on `train_loader`.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        avg_loss = running_loss / len(train_loader.dataset)
        # optional: print(f"  Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")
    # return trained model

def evaluate(model, dataloader, device):
    """
    Compute classification accuracy of `model` on `dataloader`.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def evaluate_zero_shot(clip_model, processor, dataloader, device, class_names):
    """
    Zero-shot inference with CLIP: for each image, pick the class whose prompt
    has highest similarity.
    """
    # prepare text prompts once
    prompts = [f"a photo of a {c}" for c in class_names]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)

    clip_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            # preprocess images
            pixel_inputs = processor(images=imgs, return_tensors="pt").pixel_values.to(device)
            # forward
            outputs = clip_model(pixel_values=pixel_inputs,
                                 input_ids=text_inputs.input_ids,
                                 attention_mask=text_inputs.attention_mask)
            logits = outputs.logits_per_image  # shape [batch, num_classes]
            preds = logits.argmax(dim=1).cpu()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def main():
    # 1. Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("outputs", exist_ok=True)

    # Data transforms and loaders
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])

    train_full = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    test_set  = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)

    class_names = train_full.classes  # ['airplane', 'automobile', ... 'truck']

    # Few-shot experiments
    few_shot_results = {}
    for shots in [1, 5, 10]:
        print(f"\n=== {shots}-shot experiment ===")
        few_dataset = get_few_shot_subset(train_full, shots, num_classes=len(class_names))
        few_loader  = DataLoader(few_dataset, batch_size=shots*len(class_names), shuffle=True, num_workers=2)

        # load pretrained ResNet18, freeze all but final layer
        model = models.resnet18(pretrained=True)
        for p in model.parameters():
            p.requires_grad = False
        n_feats = model.fc.in_features
        model.fc = nn.Linear(n_feats, len(class_names))
        model = model.to(device)

        # train only fc
        train_linear_head(model, few_loader, test_loader, device, epochs=20, lr=1e-3)
        acc = evaluate(model, test_loader, device)
        print(f"{shots}-shot test accuracy: {acc*100:.2f}%")
        few_shot_results[f"{shots}-shot"] = acc

    # Zero-shot via CLIP
    print("\n=== zero-shot via CLIP ===")
    clip_model   = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor    = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    zero_acc     = evaluate_zero_shot(clip_model, processor, test_loader, device, class_names)
    print(f"zero-shot test accuracy: {zero_acc*100:.2f}%")
    few_shot_results["zero-shot"] = zero_acc

    # Plot comparison
    labels = list(few_shot_results.keys())
    accs   = [few_shot_results[k]*100 for k in labels]

    plt.figure(figsize=(6,4))
    bars = plt.bar(labels, accs, color=['C0','C0','C0','C1'])
    plt.ylabel("Accuracy (%)")
    plt.ylim(0,100)
    plt.title("Few‑Shot vs Zero‑Shot on CIFAR‑10")
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()-5, f"{acc:.1f}%", 
                 ha='center', color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/few_zero_shot_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
