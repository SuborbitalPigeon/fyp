from abc import ABCMeta, abstractmethod
from os import listdir
from os.path import isdir, join
import sys

import cv2
from cv2 import xfeatures2d

class PerformanceTest(metaclass=ABCMeta):
    def __init__(self, dirs, fileexts):
        """ Base class for benchmark implementations.

        Parameters
        ----------
        dirs : List[str]
            A list of directories to scan.
        filexts : Tuple[str]
            A tuple containing the file extensions to allow for test images.

        """
        self.files = [join(dir, file) for dir in dirs for file in listdir(dir) if file.endswith(fileexts)]

        self.detectors = ['Agast', 'AKAZE', 'BRISK', 'Fast', 'GFTT', 'KAZE', 'MSER', 'ORB']
        self.detectors += ['SIFT', 'SURF', 'Star'] # xfeatures2d module
        self.descriptors = ['AKAZE', 'BRISK', 'KAZE', 'ORB']
        self.descriptors += ['DAISY', 'FREAK', 'LATCH', 'SIFT', 'SURF'] # xfeatures2d module, removed 'BRIEF', 'LUCID'

    def create_detector(self, detector):
        """ Create detector object.

        Parameters
        ----------
        detector : str
            The detector type to create.
        """
        if detector is 'Agast':
            det = cv2.AgastFeatureDetector_create()
        elif detector is 'AKAZE':
            det = cv2.AKAZE_create()
        elif detector is 'BRISK':
            det = cv2.BRISK_create()
        elif detector is 'Fast':
            det = cv2.FastFeatureDetector_create()
        elif detector is 'GFTT':
            det = cv2.GFTTDetector_create()
        elif detector is 'KAZE':
            det = cv2.KAZE_create()
        elif detector is 'MSER':
            det = cv2.MSER_create()
        elif detector is 'ORB':
            det = cv2.ORB_create()

        elif detector is 'MSD':
            det = xfeatures2d.MSDDetector_create()
        elif detector is 'SIFT':
            det = xfeatures2d.SIFT_create()
        elif detector is 'SURF':
            det = xfeatures2d.SURF_create()
        elif detector is 'Star':
            det = xfeatures2d.StarDetector_create()
        else:
            raise ValueError("Unsupported detector")

        return det


    def create_descriptor(self, descriptor, detector):
        """ Create descriptor object.

        Parameters
        ----------
        descriptor : str
            An optional descriptor type to create.
        detector: str
            Detector name, to check if valid combination.
        """
        if descriptor is 'AKAZE': # AKAZE only allows AKAZE or KAZE detectors
            if detector is 'AKAZE' or detector is 'KAZE':
                desc = cv2.AKAZE_create()
            else:
                return None
        elif descriptor is 'BRISK':
            desc = cv2.BRISK_create()
        elif descriptor is 'FREAK':
            desc = xfeatures2d.FREAK_create()
        elif descriptor is 'KAZE': # KAZE only allows KAZE or AKAZE detectors
            if detector is 'AKAZE' or detector is 'KAZE':
                desc = cv2.KAZE_create()
            else:
                return None
        elif descriptor is 'ORB':
            desc = cv2.ORB_create()
        elif descriptor is 'BRIEF':
            desc = cv2.BRIEF_create()
        elif descriptor is 'DAISY':
            desc = xfeatures2d.DAISY_create()
        elif descriptor is 'FREAK':
            desc = xfeatures2d.FREAK_create()
        elif descriptor is 'LATCH':
            desc = xfeatures2d.LATCH_create()
        elif descriptor is 'SIFT':
            desc = xfeatures2d.SIFT_create()
        elif descriptor is 'SURF':
            desc = xfeatures2d.SURF_create()
        else:
            raise ValueError("Unsupported descriptor")

        return desc

    def get_keypoints(self, image, detector):
        """ Get the keypoints for an image

        Parameters
        ----------
        image : array_like
            The image to run the detector on.
        detector : cv2.FeatureDetector
            The detector object.
        """
        try:
            keypoints = detector.detect(image)
        except:
            return ([])
        else:
            return keypoints

    def get_descriptors(self, image, keypoints, descriptor):
        """ Get the descriptors for an image

        Parameters
        ----------
        image : array_like
            The image to run the descriptor on.
        keypoints: array_like
            The keypoints found in the image.
        descriptor: DescriptorExtractor
            The descriptor object.
        """
        try:
            (keypoints, descriptors) = descriptor.compute(image, keypoints)
        except:
            return ([], [])
        else:
            return (keypoints, descriptors)

    @abstractmethod
    def run_tests(self):
        """ Run the tests in the benchmark.

        """

    @abstractmethod
    def show_plots(self):
        """ Shows the results of the benchmark in a graphical way.

        """
        pass

    @abstractmethod
    def save_data(self):
        """ Saves the data obtained from the tests into CSV files.

        """
        pass

    @staticmethod
    def get_dirs_from_argv():
        if len(sys.argv) < 2:
            raise ValueError("No directories given")
        dirs = [dir for dir in sys.argv[1:] if isdir(dir)]
        return dirs

