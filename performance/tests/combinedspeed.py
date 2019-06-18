from os.path import join
from time import perf_counter

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from .detectordescriptor import DetectorDescriptor
from .utils import ensure_path


def run_test(images, in_algos):
    columns = ['algo', 'time', 'nkp']
    data = []

    for algo in in_algos:
        print("Running test {}".format(algo.detector_s))

        for image in images:
            start = perf_counter()
            kps = algo.detect_and_compute(image)[0]
            end = perf_counter()

            time = (end-start) * 1000  # s -> ms
            nkp = len(kps)
            data.append([algo.detector_s, time, nkp])

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(join('results', 'combinedspeed.csv'))
    return df


def generate_plots(data):
    fig, ax = plt.subplots(figsize=(8,6))

    sns.reset_orig()
    sns.swarmplot(data=data, x='algo', y='time', ax=ax)

    ax.set_title("Combined speed test")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Time taken / ms")
    ax.set_yscale('log')
    ax.grid(which='major', axis='y')
    ax.grid(which='minor', axis='y', linestyle=':', linewidth=0.5)

    path = join("results", "combinedspeed.pdf")
    ensure_path(path)
    fig.savefig(path)
    return fig
