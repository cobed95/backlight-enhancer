import cv2
import numpy as np


def log_trans(x, k, t):
    return ((2.0 * t + 2 * t / 50) / (
                ((2.0 * t + 2 * t / 50) / (-t)) * np.log((2 * k * t + t / 50) / (t / 50)))) * np.log(
        (2 * k * t + t / 50 - k * x) / (k * x + t / 50)) + t


def read_image_bgr(path):
    return cv2.imread(path)


def read_image_grayscale(path):
    return cv2.imread(path, 0)


def get_hsv_from_bgr(img_to_convert):
    return cv2.cvtColor(img_to_convert, cv2.COLOR_BGR2HSV)


def get_images(input_path):
    input_img_bgr = read_image_bgr(input_path)
    input_img_grayscale = read_image_grayscale(input_path)
    input_img_hsv = get_hsv_from_bgr(input_img_bgr)

    return [input_img_bgr, input_img_grayscale, input_img_hsv]


def get_hsv(images):
    [_, _, hsv] = images
    return hsv


def get_v(images):
    [_, _, hsv] = images
    [_, _, v] = cv2.split(hsv)
    return v


def apply_otsu_threshold(img_to_threshold):
    return cv2.threshold(img_to_threshold, 0, 255, cv2.THRESH_OTSU)


def normalize(img_to_normalize):
    return img_to_normalize / 255


def threshold_each_channel(img_to_threshold):
    if len(img_to_threshold.shape) < 3:
        channels = [img_to_threshold]
    else:
        channel_0, channel_1, channel_2 = cv2.split(img_to_threshold)
        channels = [channel_0, channel_1, channel_2]

    return [apply_otsu_threshold(channel) for channel in channels]


def main():
    file_name = 'input006.jpeg'
    input_path = f'input/{file_name}'
    images = get_images(input_path)
    img_hsv = get_hsv(images)
    h, s, v = cv2.split(img_hsv)
    thresholded = [threshold_each_channel(image) for image in images]
    [_, _, img_hsv_threshed] = thresholded
    [_, _, channel_v_threshed] = img_hsv_threshed
    threshold_v, img_otsu_v = channel_v_threshed

    v1 = (img_otsu_v / 255) * v
    v2 = ((255 - img_otsu_v) / 255) * v
    v2 = log_trans(v2, 1, threshold_v)
    v3 = (v1 + v2).astype(img_hsv.dtype)
    hsv_img_updata = np.dstack([h, s, v3])
    rgb = cv2.cvtColor(hsv_img_updata, cv2.COLOR_HSV2BGR)
    cv2.imwrite(f'output/{file_name}', rgb)


if __name__ == '__main__':
    main()
