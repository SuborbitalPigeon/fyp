from collections import OrderedDict
from time import process_time

import cv2
from cv2 import xfeatures2d

from performancetest import PerformanceTest

class CombinedSpeedTest(PerformanceTest):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._algos = [cv2.AKAZE_create()]
        self._algos.append(cv2.KAZE_create())
        self._algos.append(cv2.ORB_create())
        self._algos.append(xfeatures2d.SIFT_create())
        self._algos.append(xfeatures2d.SURF_create())

        self._times = OrderedDict()
        self._nkps = OrderedDict()

        self._images = [cv2.imread(image) for image in self.files]

    def run_tests(self):
        for algo in self._algos:
            print("Running test {}".format(algo))
            for image in self._images:
                start = process_time()
                kps, des = algo.detectAndCompute(image, None)
                end = process_time()
                usperkp = ((end - start)*1000*1000) / len(kps)
                print("{} took {:.2f} us per keypoint ({} kps)".format(algo, usperkp, len(kps)))

    def save_data(self):
        pass

    def show_plots(self):
        pass

if __name__ == '__main__':
    dirs = PerformanceTest.get_dirs_from_argv()
    test = CombinedSpeedTest(dirs=dirs, filexts=('pgm', 'ppm'))

    test.run_tests()
    test.show_plots()
    test.save_data()
