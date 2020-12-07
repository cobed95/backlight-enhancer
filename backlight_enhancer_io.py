import os
import cv2

PWD = os.path.dirname(os.path.realpath(__file__))
DEFAULT_INPUT_PATH = os.path.join(PWD, 'input', 'input001.jpeg')
DEFAULT_OUTPUT_DIR = os.path.join(PWD, 'output')
DEFAULT_OUTPUT_PATH = os.path.join(DEFAULT_OUTPUT_DIR, os.path.basename(DEFAULT_INPUT_PATH))


def read_image(input_path):
    return cv2.imread(input_path)


def write_image(output_path, img):
    cv2.imwrite(output_path, img)
