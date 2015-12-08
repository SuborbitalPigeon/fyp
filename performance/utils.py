import sys

from os.path import isdir

def get_dirs_from_argv():
    if len(sys.argv) < 2:
        raise ValueError("No directories given")
    dirs = [dir for dir in sys.argv[1:] if isdir(dir)]
    return dirs
