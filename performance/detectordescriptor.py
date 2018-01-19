import cv2
from cv2 import xfeatures2d

class DetectorDescriptor:
    det_s = ['Agast', 'AKAZE', 'BRISK', 'Fast', 'GFTT', 'KAZE', 'MSER', 'ORB']
    det_s += ['SIFT', 'SURF', 'Star']
    des_s = ['AKAZE', 'BRISK', 'KAZE', 'ORB']
    des_s += ['BRIEF', 'DAISY', 'FREAK', 'LATCH', 'SIFT', 'SURF']

    def __init__(self, det_s, des_s=None):
        self._create_detector(det_s)
        
        if des_s is not None:
            self.desc = None
            self._create_descriptor(des_s, det_s)

    def _create_detector(self, detector):
        if detector is 'Agast':
            self.det = cv2.AgastFeatureDetector_create()
        elif detector is 'AKAZE':
            self.det = cv2.AKAZE_create()
        elif detector is 'BRISK':
            self.det = cv2.BRISK_create()
        elif detector is 'Fast':
            self.det = cv2.FastFeatureDetector_create()
        elif detector is 'GFTT':
            self.det = cv2.GFTTDetector_create()
        elif detector is 'KAZE':
            self.det = cv2.KAZE_create()
        elif detector is 'MSER':
            self.det = cv2.MSER_create()
        elif detector is 'ORB':
            self.det = cv2.ORB_create()

        elif detector is 'MSD':
            self.det = xfeatures2d.MSDDetector_create()
        elif detector is 'SIFT':
            self.det = xfeatures2d.SIFT_create()
        elif detector is 'SURF':
            self.det = xfeatures2d.SURF_create()
        elif detector is 'Star':
            self.det = xfeatures2d.StarDetector_create()
        else:
            raise ValueError("Unsupported detector")


    def _create_descriptor(self, descriptor, detector):
        if descriptor is 'AKAZE': # AKAZE only allows AKAZE or KAZE detectors
            if detector is 'AKAZE' or detector is 'KAZE':
                self.desc = cv2.AKAZE_create()
            else:
                return None
        elif descriptor is 'BRISK':
            self.desc = cv2.BRISK_create()
        elif descriptor is 'FREAK':
            self.desc = xfeatures2d.FREAK_create()
        elif descriptor is 'KAZE': # KAZE only allows KAZE or AKAZE detectors
            if detector is 'AKAZE' or detector is 'KAZE':
                self.desc = cv2.KAZE_create()
            else:
                return None
        elif descriptor is 'ORB':
            self.desc = cv2.ORB_create()
        elif descriptor is 'BRIEF':
            self.desc = xfeatures2d.BriefDescriptorExtractor_create()
        elif descriptor is 'DAISY':
            self.desc = xfeatures2d.DAISY_create()
        elif descriptor is 'FREAK':
            self.desc = xfeatures2d.FREAK_create()
        elif descriptor is 'LATCH':
            self.desc = xfeatures2d.LATCH_create()
        elif descriptor is 'SIFT':
            self.desc = xfeatures2d.SIFT_create()
        elif descriptor is 'SURF':
            self.desc = xfeatures2d.SURF_create()
        else:
            raise ValueError("Unsupported descriptor")

    @staticmethod
    def _stringify(obj):
        label = str(obj).split()[0][1:]
        label = label.split('_')
        if len(label) == 2:
            label = label[1]
        else:
            label = label[0] 

        return label

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
