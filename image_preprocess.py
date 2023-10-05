import cv2
import numpy as np
import torch
from torchvision import transforms
import torchvision
import math

def preprocess_function(img: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    '''Preprocess function used for YOLOX models'''
    input_size = (1024, 1024)
    #img_t = np.stack([img]*3, -1)
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
    resized_img = cv2.resize(
        img,
        (math.ceil(img.shape[1] * r), math.ceil(img.shape[0] * r)),
        interpolation=interp,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    
    preprocess = transforms.Compose([

        transforms.ToPILImage(),

        transforms.Grayscale(num_output_channels=3),

        transforms.ToTensor()

    ])
    new_img = preprocess(resized_img).numpy()
    #padded_img = new_img.transpose((2, 0, 1))
    padded_img = np.ascontiguousarray(new_img, dtype=np.float32)
    return padded_img, bboxes*r