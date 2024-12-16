from .imagenet_c import corrupt

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


def imagenet_C_corrupt(input_path, output_path, severity=1, corruption_number=0):
    '''
    :param severity: strength with which to corrupt x; an integer in [0, 5]
    :param corruption_number: index specifying which corruption to apply
    '''
    corruption_functions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
        'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
    ]
    if 0 <= corruption_number < len(corruption_functions):
        img = Image.open(input_path)
        img = np.array(img)

        if img.ndim == 2:  # 如果是灰度图像
            img = np.stack((img,) * 3, axis=-1)  # 转换为 RGB 图像

        img_t = corrupt(img, severity=severity, corruption_number=corruption_number)

        img_t = np.clip(img_t, 0, 255).astype(np.uint8)

        img_t = Image.fromarray(img_t)
        img_t.save(output_path)
        return output_path
    else:
        raise ValueError("Invalid corruption_number specified.")


def rotate_left(input_path, output_path):
    """
    将图像向左旋转 90 度并保存到指定路径。

    参数:
    input_path (str): 原始图像的路径。
    output_path (str): 保存旋转后图像的路径。
    """
    # 打开图像
    with Image.open(input_path) as img:
        # 将图像向左旋转 90 度
        rotated_img = img.rotate(90, expand=True)
        # 保存图像到新的路径
        rotated_img.save(output_path)

def origin_image(input_path, output_path):
    """
    将图像向左旋转 90 度并保存到指定路径。

    参数:
    input_path (str): 原始图像的路径。
    output_path (str): 保存旋转后图像的路径。
    """
    # 打开图像
    with Image.open(input_path) as img:
        # 将图像向左旋转 90 度
        origin_img = img
        # 保存图像到新的路径
        origin_img.save(output_path)


def rotate_right(input_path, output_path):
    """
    将图像向右旋转 90 度并保存到指定路径。

    参数:
    input_path (str): 原始图像的路径。
    output_path (str): 保存旋转后图像的路径。
    """
    # 打开图像
    with Image.open(input_path) as img:
        # 将图像向右旋转 90 度
        rotated_img = img.rotate(-90, expand=True)
        # 保存图像到新的路径
        rotated_img.save(output_path)


def rotate_180(input_path, output_path):
    """
    将图像旋转 180 度并保存到指定路径。

    参数:
    input_path (str): 原始图像的路径。
    output_path (str): 保存旋转后图像的路径。
    """
    # 打开图像
    with Image.open(input_path) as img:
        # 将图像旋转 180 度
        rotated_img = img.rotate(180, expand=True)
        # 保存图像到新的路径
        rotated_img.save(output_path)


def flip_left_right(input_path, output_path):
    """
    将图像左右翻转并保存到指定路径。

    参数:
    input_path (str): 原始图像的路径。
    output_path (str): 保存翻转后图像的路径。
    """
    # 打开图像
    with Image.open(input_path) as img:
        # 左右翻转图像
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # 保存翻转后的图像
        flipped_img.save(output_path)


import cv2
from PIL import Image


def blur_background_with_face_detection(input_path, output_path, blur_radius=31):
    """
    使用OpenCV进行人脸检测，保留人脸清晰，模糊背景。

    参数:
    input_path (str): 原始图像的路径。
    output_path (str): 保存虚化后图像的路径。
    blur_radius (int): 背景模糊程度，默认30。
    """
    # 加载OpenCV的人脸检测模型（使用预训练的Haar特征分类器）
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 读取图像
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 创建模糊版图像
    blurred_img = cv2.GaussianBlur(img, (blur_radius, blur_radius), 0)

    # 保留人脸部分的清晰度
    for (x, y, w, h) in faces:
        # 将模糊图像的人脸部分替换为原始图像中的清晰部分
        blurred_img[y:y + h, x:x + w] = img[y:y + h, x:x + w]

    # 保存最终图像
    cv2.imwrite(output_path, blurred_img)


