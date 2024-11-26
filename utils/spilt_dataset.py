import os
import random
import shutil

def split_dataset(dataset_dir, train_dir, val_dir, test_dir, train_size=0.7, val_size=0.15):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    all_files = [f for f in os.listdir(dataset_dir) if f.endswith('.png') or f.endswith('.jpg')]
    random.shuffle(all_files)

    total_files = len(all_files)
    train_end = int(train_size * total_files)
    val_end = train_end + int(val_size * total_files)

    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    for f in train_files:
        shutil.move(os.path.join(dataset_dir, f), os.path.join(train_dir, f))
    for f in val_files:
        shutil.move(os.path.join(dataset_dir, f), os.path.join(val_dir, f))
    for f in test_files:
        shutil.move(os.path.join(dataset_dir, f), os.path.join(test_dir, f))

if __name__ == "__main__":
    dataset_dir = r'C:\Users\Smruti Jagtap\OneDrive\Desktop\CV Project\DATASET'  # Path to your dataset
    split_dataset(dataset_dir, 'dataset/train', 'dataset/val', 'dataset/test')