import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths (adjust as needed)
data_entry_path = os.path.join('DATASET', 'Data_Entry_2017.csv')
train_val_list_path = os.path.join('DATASET', 'train_val_list.txt')
test_list_path = os.path.join('DATASET', 'test_list.txt')

# 1. Load label info
data_entry = pd.read_csv(data_entry_path)
# Map: filename -> 1 if Pneumonia in Finding Labels, else 0
def has_pneumonia(labels):
    return int('Pneumonia' in str(labels).split('|'))

data_entry['pneumonia_label'] = data_entry['Finding Labels'].apply(lambda x: 1 if 'Pneumonia' in str(x).split('|') else 0)
label_map = dict(zip(data_entry['Image Index'], data_entry['pneumonia_label']))

# 2. Load splits
def load_list(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip() in label_map]

train_val_files = load_list(train_val_list_path)
test_files = load_list(test_list_path)

# Limit the total number of images to 500 (350 train, 75 val, 75 test)
MAX_TOTAL_IMAGES = 500
MAX_TRAIN_VAL = int(MAX_TOTAL_IMAGES * 0.85)  # 425 images for train+val
MAX_TEST = MAX_TOTAL_IMAGES - MAX_TRAIN_VAL    # 75 images for test

# Take a subset of the files
train_val_files = train_val_files[:MAX_TRAIN_VAL]
test_files = test_files[:MAX_TEST]

# 3. Split train_val into train/val (70/15/15 overall)
train_files, val_files = train_test_split(train_val_files, test_size=0.1765, random_state=42)  # 0.1765*0.85 ≈ 0.15

# 4. Write CSVs
def write_csv(file_list, out_path):
    df = pd.DataFrame({'filename': file_list, 'label': [label_map[f] for f in file_list]})
    df.to_csv(out_path, index=False)

write_csv(train_files, 'train.csv')
write_csv(val_files, 'val.csv')
write_csv(test_files, 'test.csv')

print(f'CSV files generated with reduced dataset size:')
print(f'- Training set: {len(train_files)} images')
print(f'- Validation set: {len(val_files)} images')
print(f'- Test set: {len(test_files)} images') 