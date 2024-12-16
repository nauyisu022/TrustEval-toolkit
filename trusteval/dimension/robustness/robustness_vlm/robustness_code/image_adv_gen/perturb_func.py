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

        if img.ndim == 2:  
            img = np.stack((img,) * 3, axis=-1) 

        img_t = corrupt(img, severity=severity, corruption_number=corruption_number)

        img_t = np.clip(img_t, 0, 255).astype(np.uint8)

        img_t = Image.fromarray(img_t)
        img_t.save(output_path)
        return output_path
    else:
        raise ValueError("Invalid corruption_number specified.")


def rotate_left(input_path, output_path):
    with Image.open(input_path) as img:
        rotated_img = img.rotate(90, expand=True)
        rotated_img.save(output_path)

def origin_image(input_path, output_path):
    with Image.open(input_path) as img:
        origin_img = img
        origin_img.save(output_path)


def rotate_right(input_path, output_path):
    with Image.open(input_path) as img:
        rotated_img = img.rotate(-90, expand=True)
        rotated_img.save(output_path)


def rotate_180(input_path, output_path):
    with Image.open(input_path) as img:
        rotated_img = img.rotate(180, expand=True)
        rotated_img.save(output_path)


def flip_left_right(input_path, output_path):
    with Image.open(input_path) as img:
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_img.save(output_path)


import cv2
from PIL import Image


def blur_background_with_face_detection(input_path, output_path, blur_radius=31):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    blurred_img = cv2.GaussianBlur(img, (blur_radius, blur_radius), 0)

    for (x, y, w, h) in faces:
        blurred_img[y:y + h, x:x + w] = img[y:y + h, x:x + w]
    cv2.imwrite(output_path, blurred_img)


