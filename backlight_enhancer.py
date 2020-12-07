import os
from argparse import ArgumentParser

from log_transformation import run as run_log_transformation
# from histogram_equalization import run as run_histogram_equalization
# from pyramid_fusion import run as run_pyramid_fusion

LOG_TRANSFORMATION = 'log-transformation'
HISTOGRAM_EQUALIZATION = 'histogram-equalization'
PYRAMID_FUSION = 'pyramid-fusion'


def get_parser():
    pwd = os.path.dirname(os.path.realpath(__file__))

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
                                   default=f'{pwd}/input/input001.jpeg')
        target_parser.add_argument('--output-dir',
                                   required=False,
                                   help='path to the directory in which the output image will be saved',
                                   default=f'{pwd}/output')

    arg_parser = ArgumentParser(description='Enhance a backlit image')
    add_arguments(arg_parser)

    return arg_parser


def run():
    arg_parser = get_parser()
    args = arg_parser.parse_args()
    print(args.method, args.input_path, args.output_dir)
    print(os.path.basename(args.input_path))
    output_path = os.path.join(args.output_dir, os.path.basename(args.input_path))
    if args.method == LOG_TRANSFORMATION:
        run_log_transformation(input_path=args.input_path, output_path=output_path)
    elif args.method == HISTOGRAM_EQUALIZATION:
        # run_histogram_equalization(input_path=args.input_path, output_path=output_path)
        pass
    elif args.method == PYRAMID_FUSION:
        # run_pyramid_fusion(input_path=args.input_path, output_path=output_path)
        pass
    else:
        arg_parser.error(f'Invalid value for argument \'method\': {args.method}')


if __name__ == '__main__':
    run()
