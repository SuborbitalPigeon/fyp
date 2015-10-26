import os

def get_images_from_dir(dir):
    files = []

    for file in os.listdir(dir):
        if file.endswith('ppm'):
            files.append(os.path.join(dir, file))

    return files

def get_images_from_dirs(dirs):
    files = []

    for dir in dirs:
        files += get_images_from_dir(dir)

    return files
