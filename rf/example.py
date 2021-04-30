#!/usr/bin/python3

import os
import sys
import argparse
import pdb


ROOT_DIR = os.getenv('ROOT_DIR')
resDir = os.path.join(ROOT_DIR, 'resDir')
modelDir = os.path.join(resDir, 'modelDir')
os.makedirs(modelDir, exist_ok=True)


def main(opts):
    pass


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-m', '--modelType', help='')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseArgs(sys.argv)
    main(opts)
