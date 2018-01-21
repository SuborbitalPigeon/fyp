from os import listdir
from os.path import isdir, join
import sys


def get_files_from_array(string, image=True):
    if image:
        exts = ('pgm', 'ppm')

    if isinstance(string, str):
        string = string.split()

    dirs = [dir for dir in string if isdir(dir)]
    files = [join(dir, file) for dir in dirs for file in listdir(dir) if file.endswith(exts)]
    return files


def get_files_from_string(string, image=True):
    string = string.split()
    return get_files_from_array(string, image)


def get_files_from_argv(image=True):
    if len(sys.argv) < 2:
        raise ValueError("No directories given")

    return get_files_from_array(sys.argv[1:], image)
