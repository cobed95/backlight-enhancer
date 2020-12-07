import os
import cv2
from argparse import ArgumentParser

from backlight_enhancer_io import read_image, write_image, DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_DIR
from log_transformation import enhance as enhance_with_log_transformation
# from histogram_equalization import run as run_histogram_equalization
# from pyramid_fusion import run as run_pyramid_fusion

LOG_TRANSFORMATION = 'log-transformation'
HISTOGRAM_EQUALIZATION = 'histogram-equalization'
PYRAMID_FUSION = 'pyramid-fusion'


def get_parser():

    def add_arguments(target_parser):
        target_parser.add_argument('--method',
                                   required=True,
                                   help='enhancement method to use',
                                   choices=[LOG_TRANSFORMATION,
                                            HISTOGRAM_EQUALIZATION,
                                            PYRAMID_FUSION])
        target_parser.add_argument('--input-path',
                                   required=False,
                                   help='path to the input image',
                                   default=DEFAULT_INPUT_PATH)
        target_parser.add_argument('--output-dir',
                                   required=False,
                                   help='path to the directory in which the output image will be saved',
                                   default=DEFAULT_OUTPUT_DIR)

    arg_parser = ArgumentParser(description='Enhance a backlit image')
    add_arguments(arg_parser)

    return arg_parser


def run():
    arg_parser = get_parser()
    args = arg_parser.parse_args()
    print(args.method, args.input_path, args.output_dir)
    print(os.path.basename(args.input_path))
    output_path = os.path.join(args.output_dir, os.path.basename(args.input_path))

    img_to_enhance = read_image(args.input_path)
    if args.method == LOG_TRANSFORMATION:
        enhanced_img = enhance_with_log_transformation(img_to_enhance)
    elif args.method == HISTOGRAM_EQUALIZATION:
        # enhanced_img = run_histogram_equalization(input_path=args.input_path, output_path=output_path)
        pass
    else:
        # enhanced_img = run_pyramid_fusion(input_path=args.input_path, output_path=output_path)
        pass

    write_image(output_path, enhanced_img)


if __name__ == '__main__':
    run()
