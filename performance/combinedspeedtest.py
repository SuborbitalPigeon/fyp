from os.path import join
from time import perf_counter

import cv2
from cv2 import xfeatures2d
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from detectordescriptor import DetectorDescriptor
from utils import get_files_from_argv

class CombinedSpeedTest:
    def __init__(self, files):
        algos_s = ['AKAZE', 'BRISK', 'KAZE', 'ORB', 'SIFT', 'SURF']

        self._algos = [DetectorDescriptor(s) for s in algos_s]
        self._images = [cv2.imread(image) for image in files]

    def run_tests(self):
        algos = []
        times = []
        nkps = []

        for algo in self._algos:
            print("Running test {}".format(algo.detector_s))

            for image in self._images:
                start = perf_counter()
                kps = algo.detect_and_compute(image)[0]
                end = perf_counter()

                algos.append(algo.detector_s)
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
        ax.grid(axis='y')

        fig.savefig(join("results", "combinedspeed.pdf"))
        return fig
