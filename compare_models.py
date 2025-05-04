import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import get_model
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd

def evaluate(model, loader, device):
    model.eval()
    all_labels = []
    all_probs = []
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images)
            if outputs.shape[-1] == 1 or len(outputs.shape) == 1:
                outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    all_preds = (np.array(all_probs) > 0.5).astype(int)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean()
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = float('nan')
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return acc, prec, rec, f1, auc

def train_and_eval(model_name, device, epochs=10):
    train_loader, val_loader, test_loader = get_dataloaders(
        "train.csv", "val.csv", "test.csv", "DATASET", batch_size=32)
    model = get_model(model_name, num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if outputs.shape[-1] == 1 or len(outputs.shape) == 1:
                outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # Early stopping on val loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().to(device)
                outputs = model(images)
                if outputs.shape[-1] == 1 or len(outputs.shape) == 1:
                    outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"best_{model_name}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    # Evaluate on test set
    acc, prec, rec, f1, auc = evaluate(model, test_loader, device)
    return {"model": model_name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for model_name in ["chexnet", "efficientnet"]:
        print(f"Training and evaluating {model_name}...")
        res = train_and_eval(model_name, device, epochs=10)
        results.append(res)
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("model_comparison_results.csv", index=False)

if __name__ == "__main__":
    main() 