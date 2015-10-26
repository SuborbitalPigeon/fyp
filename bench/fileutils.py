import os

def get_images_from_dir(dir):
	files = []

        for file in os.listdir(dir):
        	if file.endswith('ppm'):
	                files.append(os.path.join(dir, file))

        return files
