import os
import json

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models.unet_plusplus import UNetPlusPlus
from losses import BCEDiceLoss, dice_coefficient
from dataset import train_dataset, test_dataset

# -----------------------------
# 0. 설정
# -----------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 50
LR = 1e-4
DEEP_SUPERVISION = False  # True로 바꾸면 X_0,1~X_0,depth 다중 출력에 대해 loss 평균 계산

# UNet++는 dense skip pathway 때문에 backward 시 저장해야 하는 activation이 UNet보다 훨씬 많아
# 동일한 batch_size(16)에서 CUDA OOM이 발생할 수 있습니다.
# -> 실제 GPU에 올리는 배치는 줄이고(BATCH_SIZE), 대신 여러 스텝의 gradient를 누적(GRAD_ACCUM_STEPS)해서
#    유효 배치 크기(EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRAD_ACCUM_STEPS)는 UNet과 동일하게 16을 유지합니다.
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRAD_ACCUM_STEPS  # = 16

# Mixed Precision(AMP) 사용 시 activation 메모리와 연산량을 크게 줄일 수 있음 (CUDA에서만 적용)
# USE_AMP = DEVICE.type == 'cuda'
USE_AMP = False  # 학습 불안정해져서 허용하지 않음

SAVE_DIR = './checkpoints'
SAVE_PATH_BEST = os.path.join(SAVE_DIR, 'unetpp_best.pth')
SAVE_PATH_LAST = os.path.join(SAVE_DIR, 'unetpp_last.pth')
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# 1. 데이터로더
# -----------------------------
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# 2. 모델 / 손실함수 / 옵티마이저
# -----------------------------
# UNet 코드와 동일 구조(depth=4, base_filters=64)로 맞춰 공정 비교가 가능하도록 설정
model = UNetPlusPlus(
    input_channels=3,
    output_channels=1,
    depth=4,
    base_filters=64,
    deep_supervision=DEEP_SUPERVISION,
).to(DEVICE)

criterion = BCEDiceLoss(bce_weight=0.5)
optimizer = Adam(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
scaler = torch.amp.GradScaler(enabled=USE_AMP)


def compute_loss(outputs, masks, criterion):
    """
    deep_supervision=True이면 outputs가 list(여러 해상도의 출력)이므로
    각 출력에 대한 loss를 평균내어 사용.
    deep_supervision=False이면 outputs가 단일 텐서.
    """
    if isinstance(outputs, list):
        losses = [criterion(out, masks) for out in outputs]
        return sum(losses) / len(losses)
    return criterion(outputs, masks)


def get_final_output(outputs):
    """Dice 계산 등 평가에는 가장 깊은(마지막) 출력을 최종 예측으로 사용."""
    if isinstance(outputs, list):
        return outputs[-1]
    return outputs


# -----------------------------
# 3. 학습 / 검증 루프
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler,
                     use_amp=False, accum_steps=1):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    optimizer.zero_grad()

    num_batches = len(loader)
    for step, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device).float()

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)

        # nn.BCELoss(sigmoid 확률값 입력)는 fp16 autocast에서 불안전(log(0) 위험)하므로
        # loss 계산은 항상 fp32로 수행합니다. (model forward만 AMP의 이점을 받음)
        outputs_fp32 = [o.float() for o in outputs] if isinstance(outputs, list) else outputs.float()
        loss = compute_loss(outputs_fp32, masks, criterion) / accum_steps

        scaler.scale(loss).backward()

        is_last_batch = (step + 1) == num_batches
        if (step + 1) % accum_steps == 0 or is_last_batch:
            # gradient explosion으로 인한 학습 붕괴 방지
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        final_output = get_final_output(outputs_fp32)
        batch_size = images.size(0)
        # loss는 accum_steps로 나눴으므로 로깅 시 다시 원래 스케일로 복원
        total_loss += loss.item() * accum_steps * batch_size
        total_dice += dice_coefficient(final_output.detach(), masks) * batch_size

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

        outputs_fp32 = [o.float() for o in outputs] if isinstance(outputs, list) else outputs.float()
        loss = compute_loss(outputs_fp32, masks, criterion)

        final_output = get_final_output(outputs_fp32)
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_dice += dice_coefficient(final_output, masks) * batch_size

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    n = len(loader.dataset)
    return total_loss / n, total_dice / n


def main():
    best_dice = 0.0
    history = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': []}

    print(f"[UNet++] BATCH_SIZE={BATCH_SIZE}, GRAD_ACCUM_STEPS={GRAD_ACCUM_STEPS} "
          f"(effective batch={EFFECTIVE_BATCH_SIZE}), AMP={USE_AMP}, device={DEVICE}")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, scaler,
            use_amp=USE_AMP, accum_steps=GRAD_ACCUM_STEPS
        )
        val_loss, val_dice = validate(model, test_loader, criterion, DEVICE, use_amp=USE_AMP)
        scheduler.step(val_dice)

        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)

        print(f"[UNet++] Epoch {epoch}/{EPOCHS} | "
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

    print(f"[UNet++] Training finished. Best val_dice={best_dice:.4f}")

    # visualize.py에서 loss/dice 곡선을 그릴 수 있도록 history 저장
    history_path = os.path.join(SAVE_DIR, 'unetpp_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  -> training history saved to {history_path}")

    return history


if __name__ == '__main__':
    main()