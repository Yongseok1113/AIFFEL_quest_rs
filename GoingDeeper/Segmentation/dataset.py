import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.io import imread
from albumentations import Compose, HorizontalFlip, RandomSizedCrop, Resize


# ----------------------------------------------------------------------------
# 데이터 경로
# ----------------------------------------------------------------------------
data_dir = os.path.join(os.getenv("HOME"), "work/semantic_segmentation/data/training")
test_infer_dir = os.path.join(os.getenv("HOME"), "work/semantic_segmentation/data/testing")

# ----------------------------------------------------------------------------
# Augmentation
# ----------------------------------------------------------------------------
def build_augmentation(is_train=True):
    if is_train:  # 훈련용 데이터일 경우
        return Compose([
            HorizontalFlip(p=0.5),  # 50%의 확률로 좌우대칭
            RandomSizedCrop(  # 50%의 확률로 RandomSizedCrop
                min_max_height=(300, 370),
                w2h_ratio=370 / 1242,
                size=(224, 224),  # 최신 albumentations는 height/width 대신 size=(h, w) 사용
                p=0.5
            ),
            Resize(  # 입력이미지를 224X224로 resize
                width=224,
                height=224
            )
        ])
    return Compose([  # 테스트용 데이터일 경우에는 224X224로 resize만 수행합니다.
        Resize(
            width=224,
            height=224
        )
    ])


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class KittiDataset(Dataset):
    '''
    KittiDataset은 PyTorch의 Dataset을 상속받습니다.
    우리가 KittiDataset을 원하는 방식으로 preprocess하기 위해서 Dataset을 커스텀하여 사용합니다.
    '''

    def __init__(self,
                 dir_path,
                 img_size=(224, 224, 3),
                 output_size=(224, 224),
                 is_train=True,
                 augmentation=None):
        '''
        dir_path: dataset의 directory path입니다.
        img_size: preprocess에 사용할 입력이미지의 크기입니다.
        output_size: ground_truth를 만들어주기 위한 크기입니다.
        is_train: 이 Dataset이 학습용인지 테스트용인지 구분합니다.
        augmentation: 적용하길 원하는 augmentation 함수를 인자로 받습니다.
        '''
        self.dir_path = dir_path
        self.is_train = is_train
        self.augmentation = augmentation
        self.img_size = img_size
        self.output_size = output_size
        # load_dataset()을 통해 kitti dataset의 경로에서 라벨과 이미지를 확인합니다.
        self.data = self.load_dataset()

    def load_dataset(self):
        # kitti dataset에서 필요한 정보(이미지 경로 및 라벨)를 directory에서 확인하고 로드하는 함수입니다.
        input_images = sorted(glob(os.path.join(self.dir_path, "image_2", "*.png")))
        label_images = sorted(glob(os.path.join(self.dir_path, "semantic", "*.png")))
        assert len(input_images) == len(label_images)
        data = list(zip(input_images, label_images))
        if self.is_train:
            return data[:-30]
        return data[-30:]

    def __len__(self):
        # Dataset의 length로서 전체 dataset 크기를 반환합니다.
        return len(self.data)

    def __getitem__(self, index):
        # 입력과 출력을 만듭니다.
        # 입력은 resize 및 augmentation이 적용된 input image이고
        # 출력은 semantic label입니다.
        input_img_path, output_path = self.data[index]
        _input = imread(input_img_path)
        _output = imread(output_path)
        # 특정 라벨을 이진 마스크로 변환
        _output = (_output == 7).astype(np.uint8) * 1
        data = {
            "image": _input,
            "mask": _output,
        }
        if self.augmentation:
            augmented = self.augmentation(**data)
            _input = augmented["image"] / 255.0  # Normalize
            _output = augmented["mask"]
        # target 차원 확장 (H, W) → (1, H, W)
        _output = np.expand_dims(_output, axis=0)
        return (
            torch.tensor(_input, dtype=torch.float32).permute(2, 0, 1),  # (H, W, C) → (C, H, W)
            torch.tensor(_output, dtype=torch.float32)  # (1, H, W) 형식 유지
        )

    def shuffle_data(self):
        # 한 epoch가 끝나면 실행되는 함수입니다. 학습 중인 경우에 데이터를 random shuffle합니다.
        if self.is_train:
            np.random.shuffle(self.data)


# ----------------------------------------------------------------------------
# Dataset 인스턴스 생성
# (train_unet.py / train_unetpp.py에서 이 변수들을 그대로 import해서 사용합니다)
# ----------------------------------------------------------------------------
augmentation_train = build_augmentation()
augmentation_test = build_augmentation(is_train=False)

train_dataset = KittiDataset(
    data_dir,
    augmentation=augmentation_train,
    is_train=True
)
test_dataset = KittiDataset(
    data_dir,
    augmentation=augmentation_test,
    is_train=False
)


# ----------------------------------------------------------------------------
# 라벨 없는 실제 testing set (images_2 폴더만 존재, semantic 라벨 없음)
# mIoU 계산은 불가능하며, 순수 추론(시각화)에만 사용합니다.
# ----------------------------------------------------------------------------
test_infer_dir = os.path.join(os.getenv("HOME"), "work/semantic_segmentation/data/testing")


def get_inference_image_paths():
    """라벨이 없는 실제 testing set(images_2 폴더)의 이미지 경로 리스트를 반환합니다."""
    return sorted(glob(os.path.join(test_infer_dir, "image_2", "*.png")))