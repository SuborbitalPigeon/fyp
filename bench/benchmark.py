import os

class Benchmark(object):
    def __init__(self, dirs, fileexts):
        super(Benchmark, self).__init__()

        self.files = []

        for dir in dirs:
            for file in os.listdir(dir):
                if file.endswith(fileexts):
                    self.files.append(os.path.join(dir, file))

    def run_tests(self):
	    raise NotImplementedError("This shouldn't happen")

    def show_plots(self):
	    raise NotImplementedError("This shouldn't happen")

    def save_data(self):
    	raise NotImplementedError("This shouldn't happen")
