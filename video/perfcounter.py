import numpy as np

class PerfCounter:
    def __init__(self):
        self._detect = []
        self._describe = []
        self._match = []

    @property
    def detect(self):
        return self._detect

    def append_detect(self, value):
        self._detect.append(value)

    @property
    def describe(self):
        return self._describe

    def append_describe(self, value):
        self._describe.append(value)

    @property
    def match(self):
        return self._match

    def append_match(self, value):
        self._match.append(value)

    def report_last(self, values):
        det = np.average(self._detect[-values:])
        des = np.average(self._describe[-values:])
        match = np.average(self._match[-values:])
        print("detection={:2.3f}, description={:2.3f}, match={:1.3f} ms".format(det, des, match))
