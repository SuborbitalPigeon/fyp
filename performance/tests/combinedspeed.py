from os.path import join
from time import perf_counter

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from .detectordescriptor import DetectorDescriptor


def run_test(images, in_algos):
    algos = []
    times = []
    nkps = []

    for algo in in_algos:
        print("Running test {}".format(algo.detector_s))

        for image in images:
            start = perf_counter()
            kps = algo.detect_and_compute(image)[0]
            end = perf_counter()

            algos.append(algo.detector_s)
            times.append((end-start)*1000)
            nkps.append(len(kps))

    return pd.DataFrame({'algo': algos, 'time': times, 'nkp': nkps})


def generate_plots(data):
    fig, ax = plt.subplots()

    sns.reset_orig()
    sns.swarmplot(data=data, x='algo', y='time', ax=ax)

    ax.set_title("Combined speed test")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Time taken / ms")
    ax.set_yscale('log')
    ax.grid(which='major', axis='y')
    ax.grid(which='minor', axis='y', linestyle=':', linewidth=0.5)

    fig.savefig(join("results", "combinedspeed.pdf"))
    return fig