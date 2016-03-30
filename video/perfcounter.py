import csv

import numpy as np

class PerfCounter:
    def __init__(self):
        self.detect = []
        self.compute = []
        self.match = []
        self.nkps = []

    def report_last(self, values):
        det = np.average(self.detect[-values:])
        des = np.average(self.compute[-values:])
        match = np.average(self.match[-values:])
        nkps = np.average(self.nkps[-values:])
        print("detection={:02.3f}, description={:02.3f}, matching={:02.3f} ms with {} keypoints".format(det, des, match, nkps))

    def save_data(self):
        rows = zip(self.detect, self.compute, self.match, self.nkps)

        with open('perf.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['detect', 'compute', 'match', 'nkps'])
            writer.writerows(rows)
