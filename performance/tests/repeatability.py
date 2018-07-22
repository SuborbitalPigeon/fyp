import os

import cv2
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.spatial import distance

from .detectordescriptor import DetectorDescriptor
from .utils import create_mask, ensure_path, h_for_file, point_in_mask, transform_point


def run_test(files, full=False, threshold=2):
    """Run the repeatability test"""
    columns = ['detector', 'image', 'common', 'repeat']
    data = []

    if full:
        det_s = {**DetectorDescriptor.detectors, **DetectorDescriptor.xdetectors}.keys()
    else:
        det_s = DetectorDescriptor.detectors.keys()

    for detector in det_s:
        algo = DetectorDescriptor(detector)
        print("Running test {}".format(detector))

        for f in files:
            filename = os.path.basename(f).split('.')[0]
            image = cv2.imread(f, 0)
            kps = algo.detect(image)

            h = h_for_file(f)
            if h is None:  # This will be the case for the base image (img1)
                baseimg = image
                basepts = kps

                data.append([detector, filename, len(basepts), len(basepts)])
                continue

            hi = linalg.inv(h)
            mask = create_mask(baseimg.shape, hi)

            # Only those that are common
            bpts = []
            for pt in basepts:
                if point_in_mask(pt.pt, mask):
                    bpts.append(pt)
            bptst = np.vstack([pt.pt for pt in bpts])

            rep = 0
            for point in kps:
                tp = transform_point(point.pt, hi)
                if point_in_mask(tp, mask):
                    dists = distance.cdist([tp], bptst)
                    if np.min(dists) < threshold:
                        rep += 1

            data.append([detector, filename, len(bpts), rep])

    df = pd.DataFrame(data, columns=columns)
    df['repeatability'] = df['repeat'] / df['common']
    return df

def generate_plots(df):
    data = df.pivot_table('repeatability', 'image', 'detector')

    for algo in data:
        fig, ax = plt.subplots()

        data[algo].plot(ax=ax)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        ax.set_title("Repeatability for {}".format(data[algo].name))

        plt.tight_layout()

        path = os.path.join("results", "repeatability", '{}.pdf'.format(data[algo].name))
        ensure_path(path)
        fig.savefig(path)
