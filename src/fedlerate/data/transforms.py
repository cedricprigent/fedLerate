from torchvision.transforms.functional import rotate
from torchvision.transforms import Normalize

def get_normalization(channels=3):
    if channels == 1:
        return Normalize(mean=[0.449], std=[0.226])
    else:
        return Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class Rotate(object):
    """Rotate PIL sample."""

    def __init__(self, rotation):
        self.rotation = rotation
    
    def __call__(self, sample):
        
        return rotate(sample, self.rotation)



class LabelFlip(object):
    """Flip labels of the dataset."""

    def __init__(self, shift):
        self.shift = shift

    def __call__(self, label):

        return (label-self.shift) % 10


class MinMaxScaler(object):
    """Apply Min-Max Normalization to data sample."""

    def __init__(self, min_value, max_value):
        self.min = min_value
        self.max = max_value

    def __call__(self, sample):

        return (sample - self.min) / (self.max - self.min)