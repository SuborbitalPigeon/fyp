import itertools
from os.path import join
from time import perf_counter

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .detectordescriptor import DetectorDescriptor
from .utils import ensure_path


def run_test(images, full=False):
    """Run the speed test"""
    columns = ['detector', 'descriptor', 'time', 'nkp']
    data = []
    count = 0

    if full:
        det_s = {**DetectorDescriptor.detectors, **DetectorDescriptor.xdetectors}.keys()
        des_s = {**DetectorDescriptor.descriptors, **DetectorDescriptor.xdescriptors}.keys()
    else:
        det_s = DetectorDescriptor.detectors.keys()
        des_s = DetectorDescriptor.descriptors.keys()

    print("{} tests to run".format(len(det_s) * len(des_s)))
    for detector, descriptor in itertools.product(det_s, des_s):
        algo = DetectorDescriptor(detector, descriptor)

        if count % 10 == 0 and count != 0:
            print(" {} \n.".format(count), end='')
        else:
            print(".", end='')
        count += 1
        
        if algo.desc is None:
            continue

        for image in images:
            start = perf_counter()
            keypoints = algo.detect(image)
            keypoints = algo.compute(image, keypoints)[0]
            end = perf_counter()

            time = (end-start) * 1000  # s -> ms
            nkps = len(keypoints)
            data.append([detector, descriptor, time, nkps])

    print("\nDone!")
            
    return pd.DataFrame(data, columns=columns)


def generate_plots(data):
    """ Generate single detector plots, and save to PDF."""
    detectors = {**DetectorDescriptor.detectors, **DetectorDescriptor.xdetectors}.keys()
    for detector in detectors:
        df = data[data.detector == detector]

        fig, ax = plt.subplots()

        sns.swarmplot(data=df, x='descriptor', y='time', ax=ax)

        ax.set_title("Detector = {}".format(detector))
        ax.set(xlabel="Descriptor", ylabel="CPU time per image / ms")
        ax.set(yscale='log', ylim=(1, data.time.max()))
        ax.grid(which='major', axis='y')
        ax.grid(which='minor', axis='y', linestyle=':', linewidth=0.5)
        plt.tight_layout()

        path = join("results", "speed", detector.lower()) + ".pdf"
        ensure_path(path)
        fig.savefig(path)


def generate_heatmap(data):
    """Generate heatmap with all combinations of detector and descriptor."""
    fig, ax = plt.subplots(figsize=(8, 6))

    df = data.pivot_table(['nkp', 'time'], ['detector', 'descriptor'], aggfunc=np.sum)
    df = ((df.time / df.nkp) * 1000).unstack()  # ms -> us
    df = df.replace(np.inf, np.nan)

    sns.heatmap(df, vmin=0, cmap='viridis_r', annot=True, fmt=".0f", ax=ax,
                linewidths=1, cbar_kws={'label': 'Average CPU time per keypoint / µs'})
    ax.set_title("Speed test")
    ax.minorticks_on()
    plt.tight_layout()

    path = join("results", "speed.pdf")
    ensure_path(path)
    fig.savefig(path)
    return fig
