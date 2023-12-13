import os
import random
from shutil import copyfile

def split_images(source_folder, train_fp, val_fp, split_ratio=0.8, seed=None):
    # Set seed for reproducibility
    random.seed(seed)

    # Get a list of all image files in the source folder
    image_files = [f.split('.')[0] for f in os.listdir(source_folder) if f.lower().endswith(('.png'))]

    # Shuffle the list of image files
    random.shuffle(image_files)

    # Calculate the split index
    split_index = int(len(image_files) * split_ratio)

    # Split the images into training and validation sets
    train_images = image_files[:split_index]
    validation_images = image_files[split_index:]

    # Copy images to the respective folders
    with open(train_fp, 'w') as file:
        file.write('\n'.join(train_images))
    print('Train files listed in txt')

    with open(val_fp, 'w') as file:
        file.write('\n'.join(validation_images))
    print('Validation files listed in txt')


    return None

# Example usage:
source_folder = '/workspaces/ECE661GroupProject_TransferLearning/data/image_png'
train_fp = '/workspaces/ECE661GroupProject_TransferLearning/data/splits/train.txt'
val_fp = '/workspaces/ECE661GroupProject_TransferLearning/data/splits/val.txt'
split_images(source_folder, train_fp, val_fp, split_ratio=0.8, seed=42)
