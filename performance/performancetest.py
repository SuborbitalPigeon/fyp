import itertools
import os

import cv2
from cv2 import xfeatures2d

class PerformanceTest(object):
    def __init__(self, dirs, fileexts):
        """ Base class for benchmark implementations.

        Parameters
        ----------
        dirs : List[str]
            A list of directories to scan.
        filexts : Tuple[str]
            A tuple containing the file extensions to allow for test images.

        """
        super(PerformanceTest, self).__init__()

        self.files = [os.path.join(dir, file) for dir in dirs for file in os.listdir(dir) if file.endswith(fileexts)]

        self.detectors = ['AKAZE', 'BRISK', 'FAST', 'GFTT', 'KAZE', 'MSER', 'ORB', 'SIFT', 'SURF', 'Star'] # missing: 'LUCID'
        self.descriptors = ['AKAZE', 'BRISK', 'FREAK', 'KAZE', 'ORB', 'SIFT', 'SURF']

    def _create_detector_descriptor(self, detector, descriptor):
        if detector not in self.detectors:
            raise ValueError("Unsupported detector")
        if descriptor not in self.descriptors:
            raise ValueError("Unsupported descriptor")

        if detector is 'AKAZE':
            det = cv2.AKAZE_create()
        elif detector is 'BRISK':
            det = cv2.BRISK_create()
        elif detector is 'FAST':
            det = cv2.FastFeatureDetector_create()
        elif detector is 'GFTT':
            det = cv2.GFTTDetector_create()
        elif detector is 'KAZE':
            det = cv2.KAZE_create()
        elif detector is 'LUCID':
            det = xfeatures2d.LUCID_create()
        elif detector is 'MSER':
            det = cv2.MSER_create()
        elif detector is 'ORB':
            det = cv2.ORB_create()
        elif detector is 'SIFT':
            det = xfeatures2d.SIFT_create()
        elif detector is 'SURF':
            det = xfeatures2d.SURF_create()
        elif detector is 'Star':
            det = xfeatures2d.StarDetector_create()
        else:
            raise ValueError("Unsupported detector")

        if descriptor is 'AKAZE':
            if detector is 'AKAZE' or detector is 'KAZE':
                desc = cv2.AKAZE_create()
            else:
                return None, None
        elif descriptor is 'BRISK':
            desc = cv2.BRISK_create()
        elif descriptor is 'FREAK':
            desc = xfeatures2d.FREAK_create()
        elif descriptor is 'KAZE':
            if detector is 'AKAZE' or detector is 'KAZE':
                desc = cv2.KAZE_create()
            else:
                return None, None
        elif descriptor is 'ORB':
            desc = cv2.ORB_create()
        elif descriptor is 'SIFT':
            desc = xfeatures2d.SIFT_create()
        elif descriptor is 'SURF':
            desc = xfeatures2d.SURF_create()
        else:
            raise ValueError("Unsupported descriptor")

        return (det, desc)

    def get_keypoints(self, image, detector, descriptor):
        try:
            keypoints = detector.detect(image)
            (keypoints, descriptors) = descriptor.compute(image, keypoints)
        except:
            return ([], [])
        else:
            return (keypoints, descriptors)

    def run_tests(self):
        """ Run the tests in the benchmark.

        """
        count = 0
        for detector, descriptor in itertools.product(self.detectors, self.descriptors):
            count += 1
            label = "{}/{}".format(detector, descriptor)
            print("Running test {}/{}  - {}/{}".format(count, len(self.detectors) * len(self.descriptors), detector, descriptor))

            det, desc = self._create_detector_descriptor(detector, descriptor)
            if det == None or desc == None:
                print("Invalid combination - {}/{}".format(detector, descriptor))
                continue

            self.run_test(det, desc, label)

    def run_test(self, detector, descriptor, label):
        """ Run a test on one detector/descriptor combination.

        Should not be called manually.

        Parameters
        ----------
        detector : str
            Name of the detector to use.
        descriptor : str
            Name of the descriptor to use.
        label : str
            Label to give this combination.

        """
        raise NotImplementedError("Subclasses should implement this method")

    def show_plots(self):
        """ Shows the results of the benchmark in a graphical way.

        """
        raise NotImplementedError("Subclasses should implement this method")

    def save_data(self):
        """ Saves the data obtained from the tests into CSV files.

        """
        raise NotImplementedError("Subclasses should implement this method")
