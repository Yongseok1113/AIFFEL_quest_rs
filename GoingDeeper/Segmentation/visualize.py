import json
import os

import matplotlib.pyplot as plt
from PIL import Image


def load_history(history_path):
    """train_unet.py / train_unetpp.py가 저장한 history json을 불러옵니다."""
    with open(history_path, 'r') as f:
        return json.load(f)


def plot_training_curves(history, title='Training Curves', save_path=None):
    """
    학습이 잘 되었는지 확인하기 위한 loss / dice coefficient 곡선을 그립니다.

    history: {'train_loss': [...], 'train_dice': [...], 'val_loss': [...], 'val_dice': [...]}
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history['train_loss'], label='train_loss')
    axes[0].plot(epochs, history['val_loss'], label='val_loss')
    axes[0].set_title(f'{title} - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (BCE+Dice)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_dice'], label='train_dice')
    axes[1].plot(epochs, history['val_dice'], label='val_dice')
    axes[1].set_title(f'{title} - Dice Coefficient')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved training curve plot to {save_path}")
    plt.show()


def compare_models_curves(history_dict, metric='val_dice', save_path=None):
    """
    여러 모델(예: UNet vs UNet++)의 학습 곡선을 한 그래프에서 비교합니다.

    history_dict: {'UNet': history_unet, 'UNet++': history_unetpp}
    metric: 'train_loss' | 'val_loss' | 'train_dice' | 'val_dice'
    """
    plt.figure(figsize=(6, 4))
    for name, history in history_dict.items():
        epochs = range(1, len(history[metric]) + 1)
        plt.plot(epochs, history[metric], label=name)

    plt.title(f'Model Comparison - {metric}')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved comparison plot to {save_path}")
    plt.show()


def list_image_files(dir_path, extensions=('.png', '.jpg', '.jpeg')):
    """
    dir_path 안의 이미지 파일 경로만 정렬해서 반환합니다.
    Jupyter가 만드는 .ipynb_checkpoints 같은 하위 디렉토리나
    이미지가 아닌 파일은 자동으로 제외합니다.
    """
    if not os.path.isdir(dir_path):
        return []
    paths = []
    for name in os.listdir(dir_path):
        full_path = os.path.join(dir_path, name)
        if os.path.isfile(full_path) and name.lower().endswith(extensions):
            paths.append(full_path)
    return sorted(paths)


def show_topk_images(image_paths, titles=None, save_path=None):
    """
    inference_unet.py / inference_unetpp.py가 저장한 top-k 오버레이 결과 이미지를
    한 줄로 나열해서 보여줍니다.
    """
    n = len(image_paths)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for i, path in enumerate(image_paths):
        img = Image.open(path)
        axes[i].imshow(img)
        axes[i].axis('off')
        if titles:
            axes[i].set_title(titles[i])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved top-k comparison image to {save_path}")
    plt.show()


if __name__ == '__main__':
    # ------------------------------------------------------------------
    # 사용 예시
    # ------------------------------------------------------------------
    CKPT_DIR = './checkpoints'

    unet_history = load_history(os.path.join(CKPT_DIR, 'unet_history.json'))
    plot_training_curves(unet_history, title='UNet',
                          save_path=os.path.join(CKPT_DIR, 'unet_training_curve.png'))

    unetpp_history = load_history(os.path.join(CKPT_DIR, 'unetpp_history.json'))
    plot_training_curves(unetpp_history, title='UNet++',
                          save_path=os.path.join(CKPT_DIR, 'unetpp_training_curve.png'))

    compare_models_curves(
        {'UNet': unet_history, 'UNet++': unetpp_history},
        metric='val_dice',
        save_path=os.path.join(CKPT_DIR, 'compare_val_dice.png'),
    )

    # inference_unet.py 실행 후 top_k 폴더에 저장된 이미지가 있다면 아래처럼 확인 가능
    top_k_dir = './inference_results/unet/top_k'
    paths = list_image_files(top_k_dir)
    if paths:
        show_topk_images(paths, titles=[os.path.basename(p) for p in paths])