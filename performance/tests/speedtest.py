import itertools
from os.path import join
from time import perf_counter

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .detectordescriptor import DetectorDescriptor


def run_test(images, full=False):
    """Run the speed test"""
    count = 0
    detectors_s = []
    descriptors_s = []
    times = []
    nkps = []

    if full:
        det_s = {**DetectorDescriptor.detectors, **DetectorDescriptor.xdetectors}.keys()
        des_s = {**DetectorDescriptor.descriptors, **DetectorDescriptor.xdescriptors}.keys()
    else:
        det_s = DetectorDescriptor.detectors.keys()
        des_s = DetectorDescriptor.descriptors.keys()

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

        for image in images:
            start = perf_counter()
            keypoints = algo.detect(image)
            keypoints = algo.compute(image, keypoints)[0]
            end = perf_counter()

            detectors_s.append(detector)
            descriptors_s.append(descriptor)
            times.append((end - start) * 1000) # Milliseconds
            nkps.append(len(keypoints))

    return pd.DataFrame({'detector': detectors_s, 'descriptor': descriptors_s,
                        'time': times, 'nkp': nkps})


def generate_plots(data):
    """ Generate single detector plots, and save to PDF."""
    for detector in DetectorDescriptor.detectors.keys():
        df = data[data.detector == detector]

        fig, ax = plt.subplots()

        sns.swarmplot(data=df, x='descriptor', y='time', ax=ax)

        ax.set_title("Detector = {}".format(detector))
        ax.set(xlabel="Descriptor", ylabel="Time taken / ms")
        ax.set(yscale='log', ylim=(1, data.time.max()))
        ax.grid(which='major', axis='y')
        ax.grid(which='minor', axis='y', linestyle=':', linewidth=0.5)

        fig.savefig(join("results", "speed", detector.lower()) + ".pdf")

        return fig


def generate_heatmap(data):
    """Generate heatmap with all combinations of detector and descriptor."""
    # Heatmap
    fig, ax = plt.subplots()

    df = data.pivot_table(['nkp', 'time'], ['detector', 'descriptor'], aggfunc=np.sum)
    df = ((df.time / df.nkp) * 1000).unstack()
    df = df.replace(np.inf, np.nan)

    sns.heatmap(df, vmin=0, cmap='viridis_r', annot=True, fmt=".0f", ax=ax)
    ax.set_title("Speed test")
    ax.minorticks_on()

    fig.savefig(join("results", "speed.pdf"))
    return fig
