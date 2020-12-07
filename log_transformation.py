import cv2
import numpy as np

from backlight_enhancer_io import read_image, write_image, DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_PATH


def get_hsv(img):
    def get_hsv_from_bgr(img_to_convert):
        return cv2.cvtColor(img_to_convert, cv2.COLOR_BGR2HSV)

    img_hsv = get_hsv_from_bgr(img)
    return cv2.split(img_hsv)


def apply_otsu_threshold(img_to_threshold):
    return cv2.threshold(img_to_threshold, 0, 255, cv2.THRESH_OTSU)


def enhance_channel(k, v_original, threshold_v, v_after_threshold):
    def log_trans(x, k, t):
        return ((2.0 * t + 2 * t / 50) / (
                ((2.0 * t + 2 * t / 50) / (-t)) * np.log((2 * k * t + t / 50) / (t / 50)))) * np.log(
            (2 * k * t + t / 50 - k * x) / (k * x + t / 50)) + t

    def normalize(img_to_normalize):
        return img_to_normalize / 255

    def invert(image_to_invert):
        return 255 - image_to_invert

    def get_bright_area_v():
        return normalize(v_after_threshold) * v_original

    def get_dark_area_v():
        return normalize(invert(v_after_threshold)) * v_original

    bright_area_v = get_bright_area_v()
    dark_area_v = get_dark_area_v()
    dark_area_v_enhanced = log_trans(dark_area_v, k, threshold_v)
    v_enhanced = (bright_area_v + dark_area_v_enhanced).astype(v_original.dtype)

    return v_enhanced


def get_bgr_from_hsv(h, s, v):
    img_hsv = np.dstack([h, s, v])
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def enhance(img):
    h, s, v = get_hsv(img)
    threshold_v, v_after_threshold = apply_otsu_threshold(v)
    v_enhanced = enhance_channel(k=1, v_original=v, threshold_v=threshold_v, v_after_threshold=v_after_threshold)

    return get_bgr_from_hsv(h, s, v_enhanced)


def run():
    input_path, output_path = DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_PATH
    img = read_image(input_path)
    enhanced_img = enhance(img)
    write_image(output_path, enhanced_img)


if __name__ == '__main__':
    run()
