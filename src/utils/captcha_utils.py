from siamese import Siamese
import numpy as np
from PIL import Image
from io import BytesIO


def open_image(file):
    if isinstance(file, np.ndarray):
        img = Image.fromarray(file)
    elif isinstance(file, bytes):
        img = Image.open(BytesIO(file))
    elif isinstance(file, Image.Image):
        img = file
    else:
        img = Image.open(file)
    img = img.convert("RGB")
    return img


def predict(img, detection_list):
    model = Siamese(model_path="model/best_epoch_weights.pth")

    targets = [i.get("crop") for i in detection_list if i.get("classes") == "target"]
    chars = [i.get("crop") for i in detection_list if i.get("classes") == "char"]
    # 根据坐标进行排序
    chars.sort(key=lambda x: x[0])

    img = open_image(img)
    chars = [img.crop(char) for char in chars]
    result = []
    for img_char in chars:
        img_target_list = [img.crop(target) for target in targets]
        slys = [
            model.detect_image(img_char, img_target) for img_target in img_target_list
        ]
        slys_index = slys.index(max(slys))
        result.append(targets[slys_index])
        targets.pop(slys_index)
        if len(targets) == 0:
            break
    return result
