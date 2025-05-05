import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import get_dataloaders, ChestXrayDataset
from model import get_model
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import os

def train(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(images)
        if outputs.shape[-1] == 1 or len(outputs.shape) == 1:
            outputs = outputs.squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate(model, loader, device, dataset_df=None, split_name="val"):
    model.eval()
    all_labels = []
    all_probs = []
    all_filenames = []
    running_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images)
            if outputs.shape[-1] == 1 or len(outputs.shape) == 1:
                outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            # Get filenames if dataset_df is provided
            if dataset_df is not None:
                batch_indices = range(i * loader.batch_size, i * loader.batch_size + len(images))
                all_filenames.extend(dataset_df.iloc[list(batch_indices), 0].tolist())
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
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix ({split_name})")
    plt.savefig(f"confusion_matrix_{split_name}.png")
    plt.close()
    # Save confusion matrix as CSV
    pd.DataFrame(cm).to_csv(f"confusion_matrix_{split_name}.csv", index=False)
    # Error analysis: save misclassified samples
    if dataset_df is not None and all_filenames:
        misclassified = []
        for fname, true, pred, prob in zip(all_filenames, all_labels, all_preds, all_probs):
            if true != pred:
                misclassified.append({
                    'filename': fname,
                    'true_label': int(true),
                    'pred_label': int(pred),
                    'probability': float(prob)
                })
        if misclassified:
            pd.DataFrame(misclassified).to_csv(f"misclassified_{split_name}.csv", index=False)
    return running_loss / len(loader), acc, prec, rec, f1, auc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders(
        "train.csv", "val.csv", "test.csv", "DATASET", batch_size=16)
    val_df = pd.read_csv("val.csv")
    test_df = pd.read_csv("test.csv")

    model = get_model("chexnet", num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    losses = []
    val_losses = []
    for epoch in range(1, 51):
        loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = evaluate(model, val_loader, device, val_df, split_name="val")
        test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = evaluate(model, test_loader, device, test_df, split_name="test")
        losses.append(loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}: Train Loss={loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f} | Val Prec={val_prec:.4f} | Val Rec={val_rec:.4f} | Val F1={val_f1:.4f} | Val AUC={val_auc:.4f}")
        print(f"           Test Loss={test_loss:.4f} | Test Acc={test_acc:.4f} | Test Prec={test_prec:.4f} | Test Rec={test_rec:.4f} | Test F1={test_f1:.4f} | Test AUC={test_auc:.4f}")
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    # Plot loss curves
    plt.plot(losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.show()

if __name__ == "__main__":
    main()
