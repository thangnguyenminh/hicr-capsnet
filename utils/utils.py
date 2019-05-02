import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    argparser.add_argument(
        '-s', '--show_img',
        dest='show_img',
        action='store_true',
        help='Show Sample Images')
    args = argparser.parse_args()
    return args