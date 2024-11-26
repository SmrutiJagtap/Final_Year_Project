import os
from PIL import Image
from torch.utils.data import Dataset

class HandwrittenDataset(Dataset):
    def __init__(self, images_dir, labels_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.image_labels = self._load_labels(labels_file)

    def _load_labels(self, labels_file):
        image_labels = []
        with open(labels_file, 'r') as f:
            for line in f:
                # Assuming each line is formatted as "image_filename label"
                parts = line.strip().split()
                if len(parts) == 2:
                    image_labels.append((parts[0], int(parts[1])))  # Assuming the label is an integer
        return image_labels

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_name, label = self.image_labels[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")  # Convert to RGB

        if self.transform:
            image = self.transform(image)

        return image, label