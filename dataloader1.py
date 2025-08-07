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
    def __init__(self, data, folder_path, img_transform=None,mask_transform=None):
        self.data = data
        self.folder_path = folder_path
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, mask_name = self.data[idx]
        image = Image.open(os.path.join(self.folder_path, img_name)).convert('L')
        mask = Image.open(os.path.join(self.folder_path, mask_name)).convert('L')

        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # 直接返回图像和掩码，不再返回单独的标签
        return (image, mask)


# Implements creation of datasets from their respective folders
def createDataset(folder_path, img_transform=None,mask_transform=None):
    image_files = [file for file in os.listdir(folder_path) if file.endswith('.png')]
    data = process_data(image_files)
    return BreastCancerDataset(data, folder_path, img_transform,mask_transform)


# Splits the image and mask filename and then processes it
def process_data(image_files, ext='.png'):
    data = []
    for image_file in image_files:
        f_split = re.split('[._]', image_file)
        if "mask" not in f_split:
            img_name = f_split[0] + ext
            mask_name = f_split[0] + "_mask" + ext
            data.append((img_name, mask_name))
    return data

def make_data(ROOT_DIR, args):
    # Define image class folders
    benign_folder = os.path.join(ROOT_DIR, "benign")
    malignant_folder = os.path.join(ROOT_DIR, "malignant")

    # Define parameters
    IMAGE_SIZE = 256
    BATCH_SIZE = args.batch_size
    TEST_SIZE = args.test_size
    RANDOM_SEED = args.random_seed

    # Define transformations
    img_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        # transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    # Create dataset according to classes (only benign and malignant)
    benign_dataset = createDataset(benign_folder, img_transform=img_transform,mask_transform=mask_transform)
    malignant_dataset = createDataset(malignant_folder, img_transform=img_transform,mask_transform=mask_transform)

    logging.info(f'''Breast cancer dataset lengths:
                    Benign dataset length:      {len(benign_dataset)}
                    Malignant dataset length:   {len(malignant_dataset)}
                 ''')

    # Combine the two datasets into a single dataset (train + val)
    combined_train_val_dataset = ConcatDataset([benign_dataset, malignant_dataset])

    # Shuffle the combined dataset using a random seed
    torch.manual_seed(RANDOM_SEED)
    shuffled_indices = torch.randperm(len(combined_train_val_dataset))

    # Split the shuffled dataset into train_val and test datasets
    train_val_indices, test_indices = train_test_split(shuffled_indices, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    # Prepare train_val dataset
    train_val_dataset = Subset(combined_train_val_dataset, train_val_indices)
    logging.info(f'Train + Validation dataset length: {len(train_val_dataset)}')

    # Prepare test dataset
    test_dataset = Subset(combined_train_val_dataset, test_indices)
    logging.info(f'Test dataset length: {len(test_dataset)}')

    return train_val_dataset , test_dataset