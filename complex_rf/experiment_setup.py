"""
Hyper-parameters
"""
import argparse


def get_arguments():
    """ Hyper-parameters """

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--architecture", type=str, choices=['reim',
                                                                   'reim2x',
                                                                   'reimsqrt2x',
                                                                   'magnitude',
                                                                   'phase',
                                                                   're',
                                                                   'im',
                                                                   'modrelu',
                                                                   'crelu'],
                        default='modrelu',
                        help="Architecture")

    args = parser.parse_args()

    return args
