import torch
import torch.nn as nn


def dice_coefficient(preds, targets, threshold=0.5, smooth=1e-6):
    """
    검증/평가용 Dice coefficient (이진화 후 계산, gradient 불필요).

    preds, targets : (N, C, H, W) 텐서. preds는 모델의 sigmoid 출력(0~1 확률값)이라고 가정.
    반환값 : 배치 평균 Dice coefficient (float)
    """
    preds = (preds > threshold).float()
    targets = (targets > threshold).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()


class SoftDiceLoss(nn.Module):
    """
    학습(backward)용 미분 가능한 soft dice loss.
    이진화하지 않고 sigmoid 확률값을 그대로 사용하여 gradient가 흐르도록 함.
    """

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = preds.contiguous().view(preds.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        intersection = (preds * targets).sum(dim=1)
        union = preds.sum(dim=1) + targets.sum(dim=1)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    BCE + Dice Loss 결합.
    - BCE: 픽셀 단위 분류 성능(class imbalance에 다소 취약)
    - Dice: 영역(마스크) 겹침 정도를 직접 최적화, class imbalance에 강건
    두 손실을 함께 쓰면 세그멘테이션 태스크에서 안정적으로 잘 동작하는 경우가 많음.

    주의: 모델의 forward가 이미 sigmoid를 적용한 확률값을 출력한다고 가정하므로
    nn.BCELoss()를 사용함 (BCEWithLogitsLoss 아님).
    """

    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCELoss()
        self.dice = SoftDiceLoss()

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        dice_loss = self.dice(preds, targets)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss