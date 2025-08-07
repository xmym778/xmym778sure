# from unet_model.unet_model import UNetModel
import argparse
import json
import logging
import os
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.transforms.functional import to_pil_image
from scipy.spatial.distance import directed_hausdorff
from dataloader import make_data
# from unet_model.unet_model import UNetModel
from model import AAUNet
from help_functions import convertToNumpy, calculateDiceScore, calculateDiceLoss

import torch
import torch.nn.functional as F
from skimage.morphology import dilation, disk

import numpy as np
from skimage.morphology import dilation, disk


def get_gt_boundary_np(gt, radius=2):
    """
    输入: gt: N×H×W，numpy array, 0-1 掩码图
    输出: bnd: N×H×W，边界图（值为0或1）
    """
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()

    gt = (gt > 0).astype(np.uint8)
    bnd = np.zeros_like(gt, dtype=np.uint8)
    selem = disk(radius)

    for i in range(gt.shape[0]):
        _mask = gt[i]
        if _mask.ndim != 2 or _mask.max() == 0:
            continue  # 跳过非二维图像或全背景图
        try:
            _gt_dil = dilation(_mask, selem)
            bnd[i] = ((_gt_dil - _mask) == 1).astype(np.uint8)
        except Exception as e:
            print(f"[Warning] Error processing slice {i}: {e}")
            continue
    return bnd


# 评价指标===========================================
def calculateJaccard(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - intersection
    return ((intersection + smooth) / (union + smooth)).mean()

def calculatePrecision(pred, target):
    smooth = 1e-6
    tp = (pred * target).sum(dim=(1,2,3))
    fp = (pred * (1 - target)).sum(dim=(1,2,3))
    return ((tp + smooth) / (tp + fp + smooth)).mean()

def calculateRecall(pred, target):
    smooth = 1e-6
    tp = (pred * target).sum(dim=(1,2,3))
    fn = ((1 - pred) * target).sum(dim=(1,2,3))
    return ((tp + smooth) / (tp + fn + smooth)).mean()

def calculateSpecificity(pred, target):
    smooth = 1e-6
    tn = ((1 - pred) * (1 - target)).sum(dim=(1,2,3))
    fp = (pred * (1 - target)).sum(dim=(1,2,3))
    return ((tn + smooth) / (tn + fp + smooth)).mean()

def calculateHausdorff(pred, target):
    """
    计算批量图像的平均 Hausdorff 距离
    pred: torch.Tensor, shape = [B, H, W] or [B, 1, H, W]
    target: torch.Tensor, shape = [B, H, W] or [B, 1, H, W]
    返回: float，平均 Hausdorff 距离
    """
    # 去掉通道维度（如果有）
    if pred.shape[1] == 1:
        pred = pred.squeeze(1)  # shape [B, H, W]
    if target.shape[1] == 1:
        target = target.squeeze(1)

    pred = pred.cpu().numpy().astype(np.uint8)  # 先转到CPU再转成numpy
    target = target.cpu().numpy().astype(np.uint8)

    hd_list = []
    for i in range(pred.shape[0]):
        p = pred[i]  # numpy array
        t = target[i]  # numpy array

        # 双向 Hausdorff 距离
        d1 = directed_hausdorff(p, t)[0]
        d2 = directed_hausdorff(t, p)[0]
        hd_list.append(max(d1, d2))

    return float(np.mean(hd_list))

# Implements training function with early stopping and learning rate decay
def trainModel(model, train_loader, val_loader, lr, num_epochs,
               max_early_stop, patience, momentum, weight_decay,
               device, args,fold, save_checkpoint=True,dir_checkpoint=Path('./')):
    #     max_early_stop设置最大早停步数，有几代损失不减反增则早停。
    #     STEPS:
    #     1. Define loss-criterion and optimizer
    #     2. Initialize logging
    #     3. Define lists for storing training & validation loss and accuracy history
    #     4. Use BCE/CrossEntropy and Dice Loss as loss function
    #     5. Initailize learning rate decay scheduler and early_stopping_counter.
    #     6. Calculate total loss as sum of BCE loss and dice-loss.
    #     7. Iterate over train_loader & val_loader and calculate total loss and dice-score for each sample
    #     8. If valid_loss is less than best_valid_loss, increase early_stopping_counter.
    #     9. Update learning rate by decay factor using scheduler step in order to maximize dice-score.
    #     10. Store the results in relevant lists.
    #     11. Save the best model as defined by epoch_val_score.
    #     12. Finish training after NUM_EPOCHS
    #     13. Return Training/Validation loss and dice history

    # Define loss-criterion and optimizer
    criterion = nn.BCEWithLogitsLoss()
    # 如果单通道则选择BCEWithLogitsLoss
    optimizer = optim.RMSprop(model.parameters(), lr=lr,
                              weight_decay=weight_decay, momentum=momentum,
                              )



    logging.info(f'''Training & validation loop started:
        Epochs:          {num_epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {lr}
        Training size:   {1.0 - args.val_size - args.test_size:.1f}
        Validation size: {args.val_size}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
    ''')

    model.to(device)

    best_valid_loss = float('inf')
    early_stopping_counter = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=patience)

    train_dice_loss_history = []
    train_dice_score_history = []
    val_dice_loss_history = []
    val_dice_score_history = []

    total_train_samples = 0
    total_val_samples = 0
    best_dice = 0.0


    for epoch in range(num_epochs):

        # Training loop
        model.train()
        train_dice_loss = 0.0
        train_dice_score = 0.0


        weight_bce = 0.7  # BCE 或 CrossEntropy 的权重
        weight_dice = 0.3

        for images, true_masks in train_loader:

            images, true_masks = images.to(device), true_masks.to(device)
            if true_masks.dim() == 3:  # 如果掩码没有通道维度
                true_masks = true_masks.unsqueeze(1)
            optimizer.zero_grad()

            pred_main, pred_aux, edge_att = model(images)
            # 这里model输出的是未经激活函数激活过的。---------------------------------
            pred_binary = (torch.sigmoid(pred_main) > 0.5).float()

            # Dice + BCE for main
            bce_main = criterion(pred_main, true_masks)
            dice_main = calculateDiceLoss(torch.sigmoid(pred_main), true_masks)

            # Dice + BCE for auxiliary
            bce_aux = criterion(pred_aux, true_masks)
            dice_aux = calculateDiceLoss(torch.sigmoid(pred_aux), true_masks)
            # 从 GT 掩码生成边界图 (输入为 tensor，需先转 numpy 后转回)
            with torch.no_grad():
                # gt_np = true_masks.detach().cpu().numpy()
                gt_np = true_masks.squeeze(1).detach().cpu().numpy()
                gt_bnd_np = get_gt_boundary_np(gt_np, radius=2)
            gt_bnd = torch.from_numpy(gt_bnd_np).float().unsqueeze(1).to(true_masks.device)

            # 使用 BCE 或 Dice 损失监督边界注意力图
            edge_loss = F.binary_cross_entropy(edge_att, gt_bnd)

            # 权重设置（建议主分支 1.0，辅助 0.4）
            loss = (0.7 * bce_main + 0.3 * dice_main) + 0.4 * (0.7 * bce_aux + 0.3 * dice_aux) + 0.1 * edge_loss
            # 两个损失一个要求未激活，一个要求激活过的。
            # EMCAD的dice损失则要求未激活的。
            train_dice_loss += loss.item()
            train_dice_score += calculateDiceScore(pred_binary, true_masks)
            # 用单次训练的loss进行反向传播，而不是用一个epoch出来的总损失进行反向传播。train_dice_loss只是用来记录的。
            loss.backward()
            optimizer.step()

        epoch_train_score = train_dice_score / len(train_loader)
        epoch_train_loss = train_dice_loss / len(train_loader)
        train_dice_score_history.append(convertToNumpy(epoch_train_score))
        train_dice_loss_history.append(epoch_train_loss)

        # Validation loop
        model.eval()
        val_dice_loss = 0.0
        val_dice_score = 0.0
        with torch.no_grad():
            for images, true_masks in val_loader:
                images, true_masks = images.to(device), true_masks.to(device)
                pred_main, pred_aux, att = model(images)
                # pred_binary = torch.where(torch.sigmoid(pred_masks) > 0.5, 1.0, 0.0)
                pred_binary = (torch.sigmoid(pred_main) > 0.5).float()

                # 这里的model输出的是未经过sigmoid的。
                bce_main = criterion(pred_main, true_masks)
                dice_main = calculateDiceLoss(torch.sigmoid(pred_main), true_masks)

                bce_aux = criterion(pred_aux, true_masks)
                dice_aux = calculateDiceLoss(torch.sigmoid(pred_aux), true_masks)
                with torch.no_grad():
                    # gt_np = true_masks.detach().cpu().numpy()
                    gt_np = true_masks.squeeze(1).detach().cpu().numpy()
                    gt_bnd_np = get_gt_boundary_np(gt_np, radius=2)
                val_gt_bnd = torch.from_numpy(gt_bnd_np).float().unsqueeze(1).to(true_masks.device)

                # 使用 BCE 或 Dice 损失监督边界注意力图
                edge_loss = F.binary_cross_entropy(att, val_gt_bnd)

                loss = (0.7 * bce_main + 0.3 * dice_main) + 0.4 * (0.7 * bce_aux + 0.3 * dice_aux) + 0.1 * edge_loss

                val_dice_loss += loss.item()
                # 这里的也经过了正确的调整。
                val_dice_score += calculateDiceScore(pred_binary, true_masks)

        epoch_val_score = val_dice_score / len(val_loader)
        # 计算的是每一代的dice，而不是所有代累加的dice
        # 是通过整个验证集的 Dice 分数累加得到的
        # len(val_loader) 是验证集批次的数量，因此计算的是验证集的平均 Dice 分数。

        epoch_val_loss = val_dice_loss / len(val_loader)
        val_dice_score_history.append(convertToNumpy(epoch_val_score))
        val_dice_loss_history.append(epoch_val_loss)
        logging.info(f'''
        	Epoch:			    {epoch + 1}/{num_epochs} 
        	Train Loss:		    {epoch_train_loss:.4f} 
        	Train Dice Score:	{epoch_train_score:.4f} 
        	Valid Loss:		    {epoch_val_loss:.4f} 
        	Valid Dice Score:	{epoch_val_score:.4f}
        ''')


        # Save best model
        if epoch_val_score > best_dice:
            best_dice = epoch_val_score
            best_epoch = epoch + 1
            best_model = model.state_dict()

        # Early stopping
        if epoch_val_loss < best_valid_loss:
            best_valid_loss = epoch_val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= max_early_stop:
                logging.info("Early stopping triggered!")
                if save_checkpoint:
                    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                    torch.save(best_model, str(dir_checkpoint / f'best_model_fold{fold}_epoch{best_epoch}.pth'))
                    state_dict = model.state_dict()
                    torch.save(state_dict, str(dir_checkpoint / f'checkpoint_fold{fold}_epoch{epoch}.pth'))
                    logging.info(f'Checkpoint for fold {fold}, epoch {epoch} saved!')
                break
                # return best_model, best_dice, best_epoch, train_dice_loss_history, val_dice_loss_history, train_dice_score_history, val_dice_score_history

            # Learning rate decay
        scheduler.step(epoch_val_score)


    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()

    # 保存最后的模型


    filename = f'best_TRANSFORMER-aauNET_fold{fold}_epoch{best_epoch}.pth'
    torch.save(best_model, f'./{filename}')
    logging.info(f'Best model for fold {fold} saved with dice {best_dice:.4f} at epoch {best_epoch}')

    logging.warning("No best model found to save.")

    # ===============================
    # Evaluate best model on val set
    # ===============================
    model.load_state_dict(best_model)
    model.eval()
    jaccard_total = 0.0
    precision_total = 0.0
    recall_total = 0.0
    specificity_total = 0.0
    hausdorff_total = 0.0

    with torch.no_grad():
        for images, true_masks in val_loader:
            images, true_masks = images.to(device), true_masks.to(device)
            if true_masks.dim() == 3:
                true_masks = true_masks.unsqueeze(1)

            pred_masks, aux, att = model(images)

            pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()

            jaccard_total += calculateJaccard(pred_binary, true_masks)
            precision_total += calculatePrecision(pred_binary, true_masks)
            recall_total += calculateRecall(pred_binary, true_masks)
            specificity_total += calculateSpecificity(pred_binary, true_masks)
            hausdorff_total += calculateHausdorff(pred_binary, true_masks)


    num_batches = len(val_loader)
    jaccard_avg = jaccard_total / num_batches
    precision_avg = precision_total / num_batches
    recall_avg = recall_total / num_batches
    specificity_avg = specificity_total / num_batches
    hausdorff_avg = hausdorff_total / num_batches

    logging.info(f'Best Model Metrics on Validation Set (Fold {fold}):')
    logging.info(f'  Jaccard:     {jaccard_avg:.4f}')
    logging.info(f'  Precision:   {precision_avg:.4f}')
    logging.info(f'  Recall:      {recall_avg:.4f}')
    logging.info(f'  Specificity: {specificity_avg:.4f}')
    logging.info(f'  hausdorff: {hausdorff_avg:.4f}')

    logging.info("Training finished.")
    return best_model, best_dice, best_epoch, train_dice_loss_history, val_dice_loss_history, train_dice_score_history, val_dice_score_history


class AugmentedSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, mask = self.subset[idx]  # 图像和掩码已是张量
        # print(f"Type of image_aug: {type(image)}")

        image_pil = to_pil_image(image)
        mask_pil = to_pil_image(mask)
        # 打印 PIL 图像的唯一值和统计信息

        image_np = np.array(image_pil)
        mask_np = np.array(mask_pil)

        augmented = self.transform(image=image_np, mask=mask_np)
        # 转换回张量
        image_aug = augmented['image']
        mask_aug = augmented['mask']

        image_aug = image_aug.float() / 255.0
        mask_aug = mask_aug.float() / 255.0
        return image_aug, mask_aug


def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.4),
        A.RandomRotate90(p=0.2),
        ToTensorV2(),  # 转换为张量
    ], additional_targets={'mask': 'mask'})


def cross_validation(args, Train_val_dataset, k_folds=5):
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=args.random_seed)
    fold_metrics = []
    dataset_indices = list(range(len(Train_val_dataset)))
    # 这一步是做什么的？
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset_indices)):
        logging.info(f'FOLD {fold+1}')
        logging.info('--------------------------------')
        fold = fold+1
        # Initialize the model for this fold
        model = AAUNet(1,1).to(device)

        train_subset = Subset(train_val_dataset, train_ids)
        val_subset = Subset(train_val_dataset, val_ids)
        train_subset = AugmentedSubset(train_subset, transform=get_train_transforms())
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)


        # Train the model on this fold
        best_model, best_dice, best_epoch, train_loss_history, val_loss_history, train_score_history, val_score_history = trainModel(
            model, train_loader, val_loader, args.lr, args.epochs, args.early_stopping, args.patience, args.momentum, args.weight_decay, device, args,fold=fold)

        # Record the performance of this fold
        fold_metrics.append({
            'fold': fold,
            'best_dice': best_dice,
            'best_epoch': best_epoch,
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
            'train_score_history': train_score_history,
            'val_score_history': val_score_history
        })
    # 这里讲的是每一折的情况，而不是具体的汇报每一代的情况。但是val_loss_history可以记录每一代的情况。


    return fold_metrics

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')

    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=80, required=False, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--early-stopping-factor', '-es', metavar='ES', dest='early_stopping',
                        type=int, default=20, help='Early stopping factor')
    parser.add_argument('--momentum', '-m', metavar='M', dest='momentum', type=float,
                        default=0.9, help='Momentum')
    parser.add_argument('--threshold', '-th', metavar='TH', dest='threshold', type=float,
                        default=0.5, help='Threshold')
    parser.add_argument('--patience', '-p', metavar='P', dest='patience', type=int,
                        default=20, help='Patience')
    parser.add_argument('--weight-decay', '-wd', metavar='WD', dest='weight_decay', type=float,
                        default=1e-5, help='Weight decay')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    parser.add_argument('--validation', '-v', dest='val_size', type=float, default=0.2,
                        help='Percent of data that is used as validation (0.0-1.0)')
    parser.add_argument('--test', '-t', dest='test_size', type=float, default=0.01,
                        help='Percent of data that is used as test (0.0-1.0)')
    parser.add_argument('--random-seed', '-rs', metavar='RS', dest='random_seed', type=int,
                        default=42, help='Random seed')

    return parser.parse_args()

def convert_tensor_to_list(data):
    """递归地将所有 Tensor 转换为 list，并保持其他数据结构不变。"""
    if isinstance(data, torch.Tensor):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: convert_tensor_to_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_tensor_to_list(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_tensor_to_list(item) for item in data)
    elif isinstance(data, (float, int, str, bool, type(None))):
        return data
    else:
        # 如果遇到其他类型的数据，尝试调用其 tolist() 方法（如果存在）
        try:
            return data.tolist()
        except AttributeError:
            raise TypeError(f"Object of type {type(data).__name__} is not JSON serializable")


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}\n')

    ROOT_DIR = r"./Dataset_BUSI_with_GT"
    train_val_dataset, test_dataset = make_data(ROOT_DIR, args)
    # 这个dataset时接下来需要做的。
    The_fold_metrics = cross_validation(args, train_val_dataset)
    # 将 The_fold_metrics 中的所有 Tensor 转换为 list
    The_fold_metrics = convert_tensor_to_list(The_fold_metrics)

    os.makedirs('./', exist_ok=True)
    with open('./cv_history.json', 'w') as f:
        json.dump(The_fold_metrics, f, indent=4)

