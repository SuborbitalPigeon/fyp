import itertools
from os.path import join
from time import perf_counter

import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from detectordescriptor import DetectorDescriptor
from utils import get_files_from_argv

class SpeedTest:
    def __init__(self, files):
        self.images = [cv2.imread(image, 0) for image in files]

    def run_tests(self):
        count = 0
        detectors_s = []
        descriptors_s = []
        times = []
        nkps = []

        det_s, des_s = DetectorDescriptor.det_s, DetectorDescriptor.des_s

        for detector, descriptor in itertools.product(det_s, des_s):
            count += 1
            label = "{}/{}".format(detector, descriptor)
            print("Running test {}/{} - {}/{}".format(count,
                                                      len(det_s) * len(des_s),
                                                      detector, descriptor))

            algo = DetectorDescriptor(detector, descriptor)
            if algo.desc is None:
                print("Invalid combination - {}/{}".format(detector, descriptor))
                continue

            for image in self.images:
                start = perf_counter()
                keypoints = algo.detect(image)
                keypoints, _ = algo.compute(image, keypoints)
                end = perf_counter()

                detectors_s.append(detector)
                descriptors_s.append(descriptor)
                times.append((end - start) * 1000) # Milliseconds
                nkps.append(len(keypoints))

        self.data = pd.DataFrame({'detector': detectors_s, 'descriptor': descriptors_s,
                                  'time': times, 'nkp': nkps})

    def show_plots(self):
        # One graph per detector
        for detector in DetectorDescriptor.det_s:
            data = self.data[self.data.detector == detector]

            fig, ax = plt.subplots()

            sns.swarmplot(data=data, x='descriptor', y='time', ax=ax)

            ax.set_title("Detector = {}".format(detector))
            ax.set_xlabel("Descriptor")
            ax.set_ylabel("Time taken / ms")
            ax.set_yscale('log')
            ax.grid(axis='y')

            fig.savefig(join("results", "speed", detector.lower() + ".pdf"))
            
        # Heatmap
        fig, ax = plt.subplots()
        
        df = self.data.pivot_table(['nkp', 'time'], ['detector', 'descriptor'], aggfunc=np.sum)
        df = ((df.time / df.nkp) * 1000).unstack()
        df = df.replace(np.inf, np.nan)

        sns.heatmap(df, annot=True, fmt=".0f", cmap='viridis', ax=ax)
        ax.set_title("Speed test")

        fig.savefig(join("results", "speed.pdf"))
        return fig

    def save_data(self):
        self.data.to_csv(join('results', 'speed.csv'))
