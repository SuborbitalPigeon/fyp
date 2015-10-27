import os

class Benchmark:
    def run_tests(self, files):
	    raise NotImplementedError("This shouldn't happen")

    def show_plots(self):
	    raise NotImplementedError("This shouldn't happen")

    def save_data(self):
    	raise NotImplementedError("This shouldn't happen")

    @staticmethod
    def get_files_from_dirs(dirs, fileexts):
        files = []

        for dir in dirs:
            for file in os.listdir(dir):
                if file.endswith(fileexts):
                    files.append(os.path.join(dir, file))

	    return files
