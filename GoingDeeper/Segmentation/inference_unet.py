import os
import json

import torch
from PIL import Image

from models.unet import UNet
from dataset import test_dataset, augmentation_test, get_inference_image_paths
from inference_utils import get_output, calculate_iou_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = './checkpoints/unet_best.pth'
TOP_K = 3

RESULT_DIR = './inference_results/unet'
LABELED_EVAL_DIR = os.path.join(RESULT_DIR, 'labeled_eval')      # mIoU 계산용 (라벨 있는 holdout)
TOP_K_DIR = os.path.join(RESULT_DIR, 'top_k')                    # IoU 상위 K개 결과
UNLABELED_DIR = os.path.join(RESULT_DIR, 'unlabeled_testing')    # 라벨 없는 실제 testing set 추론 결과

for d in (LABELED_EVAL_DIR, TOP_K_DIR, UNLABELED_DIR):
    os.makedirs(d, exist_ok=True)


def load_model(checkpoint_path):
    model = UNet(input_channels=3, output_channels=1).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"[UNet] Loaded checkpoint: {checkpoint_path} "
          f"(epoch={ckpt.get('epoch')}, val_dice={ckpt.get('val_dice'):.4f})")
    return model


def evaluate_labeled_holdout(model):
    """
    dataset.py의 test_dataset(= training 폴더에서 held-out한 라벨 있는 30장)으로
    이미지별 IoU를 계산하고 mIoU를 산출합니다.

    새로 알려주신 라벨 없는 실제 testing set(images_2)에는 정답 마스크가 없어
    IoU 계산이 불가능하므로, 정량 평가(mIoU)와 IoU 상위 K개 시각화는
    라벨이 있는 이 holdout set을 기준으로 수행합니다.
    """
    results = []  # (iou_score, image_path, label_path, saved_output_path)

    for idx, (image_path, label_path) in enumerate(test_dataset.data):
        tmp_output_path = os.path.join(LABELED_EVAL_DIR, f"sample_{idx:03d}.png")
        _, prediction, target = get_output(
            model, augmentation_test, image_path, tmp_output_path,
            label_path=label_path, device=DEVICE
        )
        iou = calculate_iou_score(target, prediction)
        results.append((iou, image_path, label_path, tmp_output_path))

    miou = sum(r[0] for r in results) / len(results)
    print(f"[UNet] mIoU on labeled holdout ({len(results)} images): {miou:.4f}")
    return results, miou


def save_top_k(results, k=TOP_K):
    top_k = sorted(results, key=lambda r: r[0], reverse=True)[:k]
    for rank, (iou, image_path, label_path, tmp_output_path) in enumerate(top_k, start=1):
        final_output_path = os.path.join(TOP_K_DIR, f"top{rank}_iou{iou:.4f}.png")
        Image.open(tmp_output_path).save(final_output_path)
        print(f"  Top-{rank}: IoU={iou:.4f}  image={os.path.basename(image_path)} -> {final_output_path}")
    return top_k


def run_inference_on_unlabeled_testing(model, max_images=None):
    """
    라벨이 없는 실제 testing set(images_2)에 대해 순수 추론(오버레이 시각화)만 수행합니다.
    정답 마스크가 없으므로 IoU/mIoU는 계산하지 않습니다.
    """
    image_paths = get_inference_image_paths()
    if max_images is not None:
        image_paths = image_paths[:max_images]

    for idx, image_path in enumerate(image_paths):
        output_path = os.path.join(UNLABELED_DIR, f"infer_{idx:03d}.png")
        get_output(model, augmentation_test, image_path, output_path, label_path=None, device=DEVICE)

    print(f"[UNet] Saved {len(image_paths)} inference visualizations (no GT) to {UNLABELED_DIR}")


def main():
    model = load_model(CHECKPOINT_PATH)

    results, miou = evaluate_labeled_holdout(model)
    save_top_k(results, k=TOP_K)
    run_inference_on_unlabeled_testing(model)

    with open(os.path.join(RESULT_DIR, 'miou_result.json'), 'w') as f:
        json.dump({'model': 'UNet', 'miou': miou, 'num_samples': len(results)}, f, indent=2)


if __name__ == '__main__':
    main()