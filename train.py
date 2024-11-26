import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataset import HandwrittenDataset  # Replace with the actual module where HandwrittenDataset is defined

# Define your model using ResNet-50
class YourModel(nn.Module):
    def __init__(self, num_classes):
        super(YourModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)  # Load pre-trained ResNet-50
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)  # Adjust the final layer

    def forward(self, x):
        return self.resnet(x)  # Forward pass through ResNet-50

# Training function
def train():
    # Define transformations for the training data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to match ResNet input size
        transforms.ToTensor(),  # Convert images to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])

    # Load your dataset
    images_dir = r'C:\Users\Smruti Jagtap\OneDrive\Desktop\CV Project\DATASET\train'  # Update this with the path to your images
    labels_file = r'C:\Users\Smruti Jagtap\OneDrive\Desktop\CV Project\DATASET\labels.txt'  # Update this with the path to your labels text file
    train_dataset = HandwrittenDataset(images_dir=images_dir, labels_file=labels_file, transform=transform)  # Ensure this returns (image, label)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    num_classes = 15  # Set this to the number of classes in your dataset
    model = YourModel(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(10):  # Number of epochs
        for data, target in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            output = model(data)  # Forward pass

            # Debugging: Print shapes of output and target
            print(f"Output shape: {output.shape}, Target shape: {target.shape}")  

            # Ensure target is a tensor and calculate loss
            loss = F.cross_entropy(output, target)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            print(f"Loss: {loss.item()}")  # Print loss for monitoring

if __name__ == "__main__":
    train()