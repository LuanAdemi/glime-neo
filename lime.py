from abc import abstractmethod, ABC
import numpy as np
from typing import Callable, List, Tuple, Final

from concurrent.futures import ThreadPoolExecutor

from skimage.filters import gaussian
from tqdm import tqdm

from samplers import AbstractSampler

import torch


class ImageExplainer(object):
    def __init__(self, model: torch.nn.Module, segmentation_fn: Callable, sampler: AbstractSampler) -> None:  
        self.model: Final = model
        self.segmentation_fn: Final = segmentation_fn
        self.sampler: Final = sampler

    def collect_samples(self, image: np.ndarray) -> List[np.ndarray]:
        pass

    def collect_predictions(self, samples: List[np.ndarray]) -> np.ndarray:
        pass

    def segment_instance(self, instance: np.ndarray) -> np.ndarray:
        return self.segmentation_fn(instance)

    def explain(self, instance: np.ndarray, cls: int) -> Tuple[np.ndarray, np.ndarray]:
        assert instance.shape[0] == 3 and instance.ndim == 3, f"Instance must be a 3D array with 3 channels (C, H, W). Got {instance.shape}"
        pass

class ImageExplanation(object):
    def __init__(self, image: np.ndarray, segments: np.ndarray, importances: np.ndarray, samples: np.ndarray) -> None:
        """
        
        Args:
            image (np.ndarray): A 3D array representing the image with shape (C, H, W)
            segments (np.ndarray): A 2D array representing the segments of the image with shape (H, W)
        """
        assert image.shape[0] == 3 and image.ndim == 3, f"Image must be a 3D array with 3 channels (C, H, W). Got {image.shape}"
        assert segments.ndim == 2, f"Segments must be a 2D array (H, W). Got {segments.shape}"
        assert image.shape[1:] == segments.shape, f"Image and segments must have the same shape starting from the second dimension. Got {image.shape} and {segments.shape}"
        assert len(importances) == segments.max() + 1, f"The length of importances must be equal to the number of segments. Got {len(importances)} and {segments.max() + 1}"


        self.image: Final = image
        self._segments: Final = segments

        self.importances: Final = importances

        self._samples: Final = samples