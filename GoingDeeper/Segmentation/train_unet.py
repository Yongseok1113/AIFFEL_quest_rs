import os
import json

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models.unet import UNet
from losses import BCEDiceLoss, dice_coefficient
from dataset import train_dataset, test_dataset

# -----------------------------
# 0. 설정
# -----------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 50
LR = 1e-4
BATCH_SIZE = 16

# Mixed Precision(AMP) 사용 시 메모리 사용량과 연산량을 줄일 수 있음 (CUDA에서만 적용)
USE_AMP = DEVICE.type == 'cuda'

SAVE_DIR = './checkpoints'
SAVE_PATH_BEST = os.path.join(SAVE_DIR, 'unet_best.pth')
SAVE_PATH_LAST = os.path.join(SAVE_DIR, 'unet_last.pth')
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# 1. 데이터로더
# -----------------------------
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# 2. 모델 / 손실함수 / 옵티마이저
# -----------------------------
model = UNet(input_channels=3, output_channels=1).to(DEVICE)
criterion = BCEDiceLoss(bce_weight=0.5)  # BCE + Dice 결합 loss
optimizer = Adam(model.parameters(), lr=LR)
# validation dice가 개선되지 않으면 lr을 낮춤 (mode='max' : dice는 클수록 좋으므로)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
scaler = torch.amp.GradScaler(enabled=USE_AMP)


# -----------------------------
# 3. 학습 / 검증 루프
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp=False):
    model.train()
    total_loss = 0.0
    total_dice = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device).float()

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)  # sigmoid까지 적용된 확률값 (N,1,H,W)

        # nn.BCELoss(sigmoid 확률값 입력)는 fp16 autocast에서 불안전(log(0) 위험)하므로
        # loss 계산은 항상 fp32로 수행합니다. (model forward만 AMP의 이점을 받음)
        outputs_fp32 = outputs.float()
        loss = criterion(outputs_fp32, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_dice += dice_coefficient(outputs_fp32.detach(), masks) * batch_size

    # KittiDataset은 매 epoch 종료 후 shuffle_data()로 학습 데이터 순서를 섞어줌
    if hasattr(loader.dataset, 'shuffle_data'):
        loader.dataset.shuffle_data()

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    n = len(loader.dataset)
    return total_loss / n, total_dice / n


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device).float()

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)

        outputs_fp32 = outputs.float()
        loss = criterion(outputs_fp32, masks)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_dice += dice_coefficient(outputs_fp32, masks) * batch_size

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    n = len(loader.dataset)
    return total_loss / n, total_dice / n


def main():
    best_dice = 0.0
    history = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': []}

    print(f"[UNet] BATCH_SIZE={BATCH_SIZE}, AMP={USE_AMP}, device={DEVICE}")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, scaler, use_amp=USE_AMP
        )
        val_loss, val_dice = validate(model, test_loader, criterion, DEVICE, use_amp=USE_AMP)
        scheduler.step(val_dice)

        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)

        print(f"[UNet] Epoch {epoch}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} | "
              f"val_loss={val_loss:.4f} val_dice={val_dice:.4f}")

        # 매 epoch 마지막 상태 저장 (재학습 재개용)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_dice': val_dice,
        }, SAVE_PATH_LAST)

        # validation dice 기준 best 모델 저장
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
            }, SAVE_PATH_BEST)
            print(f"  -> best model saved (val_dice={best_dice:.4f}) to {SAVE_PATH_BEST}")

    print(f"[UNet] Training finished. Best val_dice={best_dice:.4f}")

    # visualize.py에서 loss/dice 곡선을 그릴 수 있도록 history 저장
    history_path = os.path.join(SAVE_DIR, 'unet_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  -> training history saved to {history_path}")

    return history


if __name__ == '__main__':
    main()