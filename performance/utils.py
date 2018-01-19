from os import listdir
from os.path import isdir, join
import sys


def get_files_from_argv(image=True):
    if len(sys.argv) < 2:
        raise ValueError("No directories given")

    if image:
        exts = ('pgm', 'ppm')

    dirs = [dir for dir in sys.argv[1:] if isdir(dir)]
    files = [join(dir, file) for dir in dirs for file in listdir(dir) if file.endswith(exts)]
    return files
