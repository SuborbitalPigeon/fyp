import os

class Benchmark(object):
    """ Base class for benchmark implementations.

    Parameters
    ----------
    dirs : List[str]
        A list of directories to scan.
    filexts : Tuple[str]
        A tuple containing the file extensions to allow for test images.

    """
    def __init__(self, dirs, fileexts):
        super(Benchmark, self).__init__()

        self.files = [os.path.join(dir, file) for dir in dirs for file in os.listdir(dir) if file.endswith(fileexts)]

    """ Run the tests in the benchmark.

    """
    def run_tests(self):
	    raise NotImplementedError("This shouldn't happen")

    """ Shows the results of the benchmark in a graphical way.

    """
    def show_plots(self):
	    raise NotImplementedError("This shouldn't happen")

    """ Saves the data obtained from the tests into CSV files.

    """
    def save_data(self):
    	raise NotImplementedError("This shouldn't happen")
