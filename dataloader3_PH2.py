import os
import torch
import pickle
import logging
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset

# ================================
# Save and Load Dataset
# ================================
def saveDatasetToFile(filename, dataset):
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)

def loadDatasetFromFile(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# ================================
# Custom Dataset for Skin Lesion
# ================================
class SkinDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        # 只加载.bmp图片
        self.image_names = sorted([
            f for f in os.listdir(image_dir) if f.endswith('.bmp')
        ])
        self.mask_names = [name.replace('.bmp', '_lesion.bmp') for name in self.image_names]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

# ================================
# Dataset Creation Function
# ================================
def make_data(ROOT_DIR):
    image_folder = os.path.join(ROOT_DIR, "image")
    mask_folder = os.path.join(ROOT_DIR, "mask")
    IMAGE_SIZE = 256
    TEST_SIZE = 0.01

    img_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    full_dataset = SkinDataset(image_folder, mask_folder, img_transform=img_transform, mask_transform=mask_transform)

    logging.info(f"Total SkinDataset size: {len(full_dataset)}")

    # Shuffle and split into train/val and test
    torch.manual_seed(42)
    indices = torch.randperm(len(full_dataset)).tolist()
    train_val_indices, test_indices = train_test_split(indices, test_size=TEST_SIZE, random_state=42)

    train_val_dataset = Subset(full_dataset, train_val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    logging.info(f"Train+Val size: {len(train_val_dataset)}, Test size: {len(test_dataset)}")

    return train_val_dataset, test_dataset
