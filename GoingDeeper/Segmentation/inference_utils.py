import numpy as np
import torch
from PIL import Image
from skimage.io import imread
from skimage.transform import resize


def get_output(model, preproc, image_path, output_path, label_path=None, device='cpu'):
    """
    단일 이미지에 대해 추론을 수행하고, 원본 이미지 위에 예측 마스크를 반투명하게
    합성한 결과를 output_path에 저장합니다.

    Args:
        model      : 학습된 세그멘테이션 모델 (forward가 sigmoid 확률값을 반환한다고 가정)
        preproc    : albumentations Compose (예: dataset.augmentation_test)
        image_path : 입력 이미지 경로
        output_path: 합성 결과 이미지 저장 경로
        label_path : 정답 라벨 이미지 경로. 있으면 IoU 계산용 target도 함께 반환, 없으면 None.
        device     : 추론에 사용할 device

    Returns:
        blended    : PIL.Image, 원본 + 예측 마스크 합성 이미지
        prediction : np.ndarray (H, W), 0/255 이진 예측 마스크
        target     : np.ndarray (H, W) 0/1 정답 마스크 또는 None
    """
    origin_img = imread(image_path)
    data = {"image": origin_img}
    processed = preproc(**data)
    input_tensor = torch.tensor(processed["image"] / 255.0, dtype=torch.float32)
    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, list):  # deep supervision인 경우 마지막(최종) 출력 사용
            output = output[-1]

    prediction = (output[0].squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    prediction_img = Image.fromarray(prediction).convert('L')

    background = Image.fromarray(origin_img).convert('RGBA')
    prediction_resized = prediction_img.resize(
        (origin_img.shape[1], origin_img.shape[0])
    ).convert('RGBA')
    blended = Image.blend(background, prediction_resized, alpha=0.5)
    blended.save(output_path)

    target = None
    if label_path:
        label_img = imread(label_path)
        label_data = {"image": label_img}
        label_processed = preproc(**label_data)["image"]
        target = (label_processed == 7).astype(np.uint8) * 1

    return blended, prediction, target


def calculate_iou_score(target, prediction, verbose=False):
    """
    target     : (H, W) 0/1 정답 마스크
    prediction : (H, W) 0/255 (혹은 0/1) 예측 마스크
    """
    prediction_bin = (prediction > 0).astype(np.uint8)
    if target.shape != prediction_bin.shape:
        prediction_bin = resize(
            prediction_bin, target.shape, mode='constant', preserve_range=True
        ).astype(np.uint8)

    intersection = np.logical_and(target, prediction_bin).sum()
    union = np.logical_or(target, prediction_bin).sum()
    iou_score = float(intersection / (union + 1e-7))

    if verbose:
        print(f"IoU : {iou_score:.6f}")
    return iou_score