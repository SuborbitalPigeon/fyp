#!/usr/bin/env python3

from os.path import join
from time import perf_counter

import cv2
from cv2 import xfeatures2d
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from utils import get_files_from_argv

class CombinedSpeedTest:
    def __init__(self, files):
        self._algos = [cv2.AKAZE_create()]
        self._algos.append(cv2.BRISK_create())
        self._algos.append(cv2.KAZE_create())
        self._algos.append(cv2.ORB_create())
        self._algos.append(xfeatures2d.SIFT_create())
        self._algos.append(xfeatures2d.SURF_create())

        self._images = [cv2.imread(image) for image in files]

    def run_tests(self):
        algos = []
        times = []
        nkps = []

        for algo in self._algos:
            label = str(algo).split()[0][1:] # Take only the algo name, and remove <
            label = label.split('_') # Remove xfeatures2d bit
            if len(label) == 2:
                label = label[1]
            else:
                label = label[0] 
            print("Running test {}".format(label))

            for image in self._images:
                start = perf_counter()
                kps, des = algo.detectAndCompute(image, None)
                end = perf_counter()

                algos.append(label)
                times.append((end-start)*1000)
                nkps.append(len(kps))

        self.data = pd.DataFrame({'algo': algos, 'time': times, 'nkp': nkps})

    def save_data(self):
        self.data.to_csv(join('results', 'combined.csv'))

    def show_plots(self):
        fig, ax = plt.subplots()

        sns.swarmplot(data=self.data, x='algo', y='time', ax=ax)

        ax.set_title("Combined speed test")
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Time taken / ms")
        ax.set_yscale('log')

        fig.savefig(join("results", "combinedspeed.pdf"))

if __name__ == '__main__':
    cv2.ocl.setUseOpenCL(False)
    sns.set_style("whitegrid")

    files = get_files_from_argv()
    test = CombinedSpeedTest(files)

    test.run_tests()
    test.save_data()
    test.show_plots()
