import itertools
import os

import cv2

class Benchmark(object):
    def __init__(self, dirs, fileexts):
        """ Base class for benchmark implementations.

        Parameters
        ----------
        dirs : List[str]
            A list of directories to scan.
        filexts : Tuple[str]
            A tuple containing the file extensions to allow for test images.

        """
        super(Benchmark, self).__init__()

        self.files = [os.path.join(dir, file) for dir in dirs for file in os.listdir(dir) if file.endswith(fileexts)]

        # removed SIFT and SURF from both lists
        self.detectors = ['FAST', 'STAR', 'ORB', 'BRISK', 'MSER', 'GFTT', 'HARRIS', 'Dense', 'SimpleBlob']
        self.descriptors = ['BRIEF', 'BRISK', 'ORB', 'FREAK']

    def _create_detector_descriptor(self, detector, descriptor):
        if detector not in self.detectors:
            raise ValueError("Unsupported detector")
        if descriptor not in self.descriptors:
            raise ValueError("Unsupported descriptor")

        det = cv2.FeatureDetector_create(detector)
        desc = cv2.DescriptorExtractor_create(descriptor)
        return (det, desc)

    def get_keypoints(self, image, detector, descriptor):
        keypoints = detector.detect(image)
        (keypoints, descriptors) = descriptor.compute(image, keypoints)
        return (keypoints, descriptors)

    def run_tests(self):
        """ Run the tests in the benchmark.

        """
        count = 0
        for detector, descriptor in itertools.product(self.detectors, self.descriptors):
            count += 1
            label = "{}/{}".format(detector, descriptor)
            print("Running test {}/{} - {}/{}".format(count, len(self.detectors) * len(self.descriptors), detector, descriptor))

            det, desc = self._create_detector_descriptor(detector, descriptor)
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
        raise NotImplementedError("This shouldn't happen")

    def save_data(self):
        """ Saves the data obtained from the tests into CSV files.

        """
    	raise NotImplementedError("This shouldn't happen")
