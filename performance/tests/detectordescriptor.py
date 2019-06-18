import re

import cv2
from cv2 import xfeatures2d

class DetectorDescriptor:
    detectors = {
        'Agast': cv2.AgastFeatureDetector_create(),
        'AKAZE': cv2.AKAZE_create(),
        'BRISK': cv2.BRISK_create(),
        'Fast': cv2.FastFeatureDetector_create(),
        'GFTT': cv2.GFTTDetector_create(),
        'KAZE': cv2.KAZE_create(),
        'MSER': cv2.MSER_create(),
        'ORB': cv2.ORB_create()
    }
    xdetectors = {
#        'Boost': xfeatures2d.BoostDesc_create(),
        'Harris': xfeatures2d.HarrisLaplaceFeatureDetector_create(),
#        'PCT': xfeatures2d.PCTSignatures_create(),
        'Star': xfeatures2d.StarDetector_create()
    }
    
    descriptors = {
        'AKAZE': None,
        'BRISK': cv2.BRISK_create(),
        'KAZE': None,
        'ORB': cv2.ORB_create(),
    }
    xdescriptors = {
#        'Boost': xfeatures2d.BoostDesc_create(),
        'BRIEF': xfeatures2d.BriefDescriptorExtractor_create(),
        'DAISY': xfeatures2d.DAISY_create(),
        'FREAK': xfeatures2d.FREAK_create(),
        'LATCH': xfeatures2d.LATCH_create(),
#        'LUCID': xfeatures2d.LUCID_create(),
        'VGG': xfeatures2d.VGG_create()
    }

    def __init__(self, det_s, des_s=None):
        self._string_re = re.compile(r'<([^\W_]+)_?(\w+)?')

        try:
            self.det = self.detectors[det_s]
        except KeyError:
            try:
                self.det = self.xdetectors[det_s]
            except KeyError:
                raise ValueError("Unsupported detector")

        if des_s:
            try:
                self.desc = self.descriptors[des_s]
            except KeyError:
                try:
                    self.desc = self.xdescriptors[des_s]
                except KeyError:
                    raise ValueError("Unsupported descriptor")

            # AKAZE and KAZE special case
            if self.desc is None:
                self.desc = self._create_kaze_descriptor(des_s)
        else:
            self.desc = None

    def _create_kaze_descriptor(self, des_s):
        """AKAZE only allows AKAZE or KAZE detectors."""

        if isinstance(self.det, cv2.AKAZE) or isinstance(self.det, cv2.KAZE):
            if des_s == 'AKAZE':
                return cv2.AKAZE_create()
            else:
                return cv2.KAZE_create()
        else:
            return None

    def _stringify(self, obj):
        match = self._string_re.match(str(obj))
        if match.group(2):      # In the case of 'xfeatures2d_SIFT' etc.
            return match.group(2)
        else:
            return match.group(1)

    @property
    def detector_s(self):
        return self._stringify(self.det)

    @property
    def descriptor_s(self):
        return self._stringify(self.desc)

    def detect_and_compute(self, image):
        return self.det.detectAndCompute(image, None)

    def detect(self, image):
        try:
            keypoints = self.det.detect(image)
        except:
            return ([])
        else:
            return keypoints

    def compute(self, image, keypoints):
        try:
            (keypoints, descriptors) = self.desc.compute(image, keypoints)
        except:
            return ([], [])
        else:
            return (keypoints, descriptors)
