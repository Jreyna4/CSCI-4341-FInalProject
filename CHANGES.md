# CHANGES.md

## Major Changes and Additions

### Data Preparation & Loading
- Added `prepare_pneumonia_csvs.py` to generate `train.csv`, `val.csv`, and `test.csv` for binary pneumonia classification, combining split and label information.
- Updated `dataset.py`:
  - Added data augmentation (random flip, rotation, color jitter) for training set.
  - Improved image path handling to search all subdirectories.
  - Added support for validation set and updated `get_dataloaders` to return train, val, and test loaders.

### Model Training & Evaluation
- Updated `main.py`:
  - Uses train, val, and test splits.
  - Implements early stopping based on validation loss.
  - Computes and prints accuracy, precision, recall, F1-score, and AUC-ROC for val and test sets after each epoch.
  - Computes and saves confusion matrices (as images and CSVs) for val and test sets after each epoch.
  - Saves misclassified samples (filename, true label, predicted label, probability) to CSV for error analysis.
  - Plots both training and validation loss curves.
  - Saves the best model checkpoint as `best_model.pth`.

### Model Comparison
- Added `compare_models.py`:
  - Trains and evaluates both CheXNet and EfficientNet on the same splits.
  - Outputs a summary table and CSV (`model_comparison_results.csv`) of their test metrics.

### Grad-CAM Visualization
- Added `gradcam_utils.py`:
  - Provides a Grad-CAM utility class and function to generate and save Grad-CAM heatmaps for any trained model (CheXNet or EfficientNet).

### Other Improvements
- Ensured all evaluation and error analysis features can be run automatically after training.
- Improved code modularity and documentation readiness for research and reproducibility.

---

**These changes bring the codebase in line with the promises in the README and prepare it for robust, research-grade experiments at the computing center.** 