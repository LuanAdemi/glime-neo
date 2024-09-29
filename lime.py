from abc import abstractmethod, ABC
from matplotlib import pyplot as plt
import numpy as np
from typing import Callable, List, Tuple, Final, Type

from concurrent.futures import ThreadPoolExecutor

from skimage.filters import gaussian
from tqdm import tqdm

from samplers import AbstractSampler

import torch

from sklearn.linear_model import Ridge

import seaborn as sns


class ImageExplainer(object):
    def __init__(self, model: torch.nn.Module, segmentation_fn: Callable, sampler_cls: Type[AbstractSampler]) -> None:  
        self.model: Final = model
        self.segmentation_fn: Final = segmentation_fn
        self.sampler_cls: Final = sampler_cls

        self.device = "cuda" # TODO: Check if cuda is available

    def _prepare_batch(self, samples: List[np.ndarray]) -> torch.Tensor:
        batch = []
        for sample in samples:
            perturbation = sample[1].squeeze()
            batch.append(torch.from_numpy(perturbation).to(self.device))

        return torch.stack(batch)

    @torch.no_grad()
    def collect_predictions(self, batch: torch.Tensor) -> np.ndarray:
        predictions = self.model(batch).cpu().numpy()
        predictions = torch.nn.functional.sigmoid(torch.from_numpy(predictions)).numpy()

        return predictions

    def explain(self, instance: np.ndarray, cls: int, n_samples: int, progress: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        assert instance.shape[0] == 3 and instance.ndim == 3, f"Instance must be a 3D array with 3 channels (C, H, W). Got {instance.shape}"
        

        segments = self.segmentation_fn(instance.transpose(1, 2, 0))

        sampler = self.sampler_cls(instance, segments)
        samples = sampler.sample(n_samples, progress)

        perturbations_batch = self._prepare_batch(samples)
        predictions = self.collect_predictions(perturbations_batch)[:, cls]
        
        X = np.stack([sample[0] for sample in samples], axis=0)
        Y = predictions
        sample_weights = np.stack([sample[2] for sample in samples], axis=0)

        linear_model = Ridge(alpha=1.0, fit_intercept=True)
        linear_model.fit(X, Y, sample_weight=sample_weights)

        importances = linear_model.coef_

        return ImageExplanation(instance, segments, importances, samples)



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
        assert len(importances) == np.unique(segments).shape[0], f"The length of importances must be equal to the number of segments. Got {len(importances)} and {segments.max() + 1}"


        self.image: Final = image
        self._segments: Final = segments

        self.importances: Final = importances

        self._samples: Final = samples

    def plot_importances(self) -> None:
        sns.barplot(x=np.arange(len(self.importances)), y=self.importances)
        plt.xlabel("Segment")
        plt.ylabel("Importance")
        plt.title("Importances of each segment")
        plt.show()

    def render(self, positive_only=False) -> np.ndarray:
        """
        Renders the explanation in a heatmap.
        
        Args:
            positive_only (bool, optional): If True, only positive importances will be rendered. Defaults to False.
        
        Returns:
            np.ndarray: A 3D array representing the image with the explanation rendered on top of it.
        """
        explanation = np.zeros_like(self._segments, dtype=np.float32)

        for i, s in enumerate(np.unique(self._segments)):
            if positive_only and self.importances[i] < 0:
                continue
            explanation[self._segments == s] = self.importances[i]

        plt.imshow(explanation, cmap='viridis')