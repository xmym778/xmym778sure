import re
import os
import torch
import pickle
import logging
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, ConcatDataset, Subset, DataLoader


# Implements dataset file saving and loading
def saveDatasetToFile(filename, dataset):
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)


def loadDatasetFromFile(filename):
    with open(filename, 'rb') as f:
        loaded_dataset = pickle.load(f)
    return loaded_dataset


# Custom dataset class for Breast Cancer Datasets
class BreastCancerDataset(Dataset):
    def __init__(self, data, image_folder, mask_folder, img_transform=None, mask_transform=None):
        self.data = data
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, mask_name = self.data[idx]
        image = Image.open(os.path.join(self.image_folder, img_name)).convert('L')
        mask = Image.open(os.path.join(self.mask_folder, mask_name)).convert('L')

        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return (image, mask)


# Implements creation of datasets from their respective folders
def createDataset(image_folder, mask_folder, img_transform=None, mask_transform=None):
    image_files = [file for file in os.listdir(image_folder) if file.endswith('.png')]
    data = process_data(image_files)
    return BreastCancerDataset(data, image_folder, mask_folder, img_transform, mask_transform)


# Now uses the exact same filename for mask (no "_mask" suffix)
def process_data(image_files):
    data = []
    for image_file in image_files:
        data.append((image_file, image_file))  # mask file has the same name
    return data


def make_data(ROOT_DIR):
    # Define image class folders
    benign_folder = os.path.join(ROOT_DIR, "benign")
    malignant_folder = os.path.join(ROOT_DIR, "malignant")
    mask_folder = os.path.join(ROOT_DIR, "mask")

    # Define parameters
    IMAGE_SIZE = 256

    TEST_SIZE = 0.01

    # Define transformations
    img_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    # Create dataset according to classes (only benign and malignant)
    benign_dataset = createDataset(benign_folder, mask_folder, img_transform=img_transform, mask_transform=mask_transform)
    malignant_dataset = createDataset(malignant_folder, mask_folder, img_transform=img_transform, mask_transform=mask_transform)

    logging.info(f'''Breast cancer dataset lengths:
                    Benign dataset length:      {len(benign_dataset)}
                    Malignant dataset length:   {len(malignant_dataset)}
                 ''')

    # Combine the two datasets into a single dataset (train + val)
    combined_train_val_dataset = ConcatDataset([benign_dataset, malignant_dataset])

    # Shuffle the combined dataset using a random seed
    torch.manual_seed(42)
    shuffled_indices = torch.randperm(len(combined_train_val_dataset))

    # Split the shuffled dataset into train_val and test datasets
    train_val_indices, test_indices = train_test_split(shuffled_indices, test_size=TEST_SIZE, random_state=42)

    # Prepare train_val dataset
    train_val_dataset = Subset(combined_train_val_dataset, train_val_indices)
    logging.info(f'Train + Validation dataset length: {len(train_val_dataset)}')

    # Prepare test dataset
    te_dataset = Subset(combined_train_val_dataset, test_indices)
    logging.info(f'Test dataset length: {len(te_dataset)}')

    return train_val_dataset, te_dataset
