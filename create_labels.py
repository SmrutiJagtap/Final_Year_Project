import os

# Specify the directory containing the images
images_dir = r'C:\Users\Smruti Jagtap\OneDrive\Desktop\CV Project\DATASET\train'  # Adjust this path if necessary

# List to hold the filename and labels
labels = []

# Loop through each file in the images directory
for filename in os.listdir(images_dir):
    if filename.endswith('.jpg'):  # Change this if your images have a different format (like .png)
        # Extract the label from the filename or assign a default label
        label = filename.split('.')[0]  # This will use the filename (without extension) as the label
        labels.append(f"{filename},{label}")

# Write the labels to labels.txt
with open('dataset/labels.txt', 'w', encoding='utf-8') as f:
    for label in labels:
        f.write(label + '\n')

print(f"Generated labels.txt with {len(labels)} entries.")