import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from model import get_model

def train(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders("train_labels.csv", "test_labels.csv", "ChestXrayImages")

    model = get_model("chexnet").to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(10):
        loss = train(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, test_loader, device)
        losses.append(loss)
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={acc:.4f}")

    torch.save(model.state_dict(), "chexnet_model.pth")
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("loss_plot.png")
    plt.show()

if __name__ == "__main__":
    main()
