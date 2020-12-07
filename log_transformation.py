import cv2
import numpy as np


def get_hsv(input_path):
    def read_image_bgr(path):
        return cv2.imread(path)

    def get_hsv_from_bgr(img_to_convert):
        return cv2.cvtColor(img_to_convert, cv2.COLOR_BGR2HSV)

    img_bgr = read_image_bgr(input_path)
    img_hsv = get_hsv_from_bgr(img_bgr)
    return cv2.split(img_hsv)


def apply_otsu_threshold(img_to_threshold):
    return cv2.threshold(img_to_threshold, 0, 255, cv2.THRESH_OTSU)


# def show_scaled(name, img):
#     resized = cv2.resize(img, dsize=(0, 0), fx=0.2, fy=0.2)
#     cv2.imshow(name, resized)


def enhance(k, v_original, threshold_v, v_after_threshold):
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


def write_img(output_path, img):
    cv2.imwrite(output_path, img)


def run(input_path, output_path):
    # file_name = 'input002.png'
    # input_path = f'input/{file_name}'

    h, s, v = get_hsv(input_path)
    threshold_v, v_after_threshold = apply_otsu_threshold(v)
    v_enhanced = enhance(k=1, v_original=v, threshold_v=threshold_v, v_after_threshold=v_after_threshold)

    img_bgr_enhanced = get_bgr_from_hsv(h, s, v_enhanced)

    # output_path = f'output/{file_name}'
    write_img(output_path, img_bgr_enhanced)


if __name__ == '__main__':
    run()
