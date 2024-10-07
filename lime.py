from abc import abstractmethod, ABC
from matplotlib import pyplot as plt
import numpy as np
from typing import Callable, List, Literal, Tuple, Final, Type

from concurrent.futures import ThreadPoolExecutor

from skimage.filters import gaussian
from scipy.interpolate import griddata
from tqdm import tqdm

from samplers import AbstractSampler

import torch

from sklearn.linear_model import Ridge

import seaborn as sns


class ImageExplainer(object):
    def __init__(self, model: torch.nn.Module, segmentation_fn: Callable, sampler: Type[AbstractSampler] | AbstractSampler) -> None:  
        self.model: Final = model
        self.segmentation_fn: Final = segmentation_fn
        if isinstance(sampler, AbstractSampler):
            self.sampler: Final = sampler
            self.sampler_cls: Final = None
        else:
            self.sampler: Final = None
            self.sampler_cls: Final = sampler

        self.device = "cpu" # TODO: Check if cuda is available

    def _prepare_batch(self, samples: List[np.ndarray]) -> torch.Tensor:
        batch = []
        for sample in samples:
            perturbation = sample[1].squeeze()
            batch.append(torch.from_numpy(perturbation).to(self.device))

        return torch.stack(batch)

    @torch.no_grad()
    def collect_predictions(self, batch: torch.Tensor) -> np.ndarray:
        predictions = self.model(batch)
        predictions = torch.nn.functional.sigmoid(predictions)

        return predictions.cpu().numpy()

    def explain(self, instance: np.ndarray, cls: int, n_samples: int, progress: bool = True) -> "ImageExplanation":
        assert instance.shape[0] == 3 and instance.ndim == 3, f"Instance must be a 3D array with 3 channels (C, H, W). Got {instance.shape}"
        
        segments = self.segmentation_fn(instance.transpose(1, 2, 0))

        if self.sampler is None:
            self.sampler = self.sampler_cls(instance, segments)

        samples = self.sampler.sample(n_samples, progress)

        perturbations_batch = self._prepare_batch(samples)
        predictions = self.collect_predictions(perturbations_batch)
        
        X = np.stack([sample[0] for sample in samples], axis=0)
        Y = predictions[:, cls]
        sample_weights = np.stack([sample[2] for sample in samples], axis=0)

        linear_model = Ridge(alpha=1.0, fit_intercept=True)
        linear_model.fit(X, Y, sample_weights)

        return ImageExplanation(instance, segments, linear_model, samples)



class ImageExplanation(object):
    def __init__(self, image: np.ndarray, segments: np.ndarray, linear_model: Ridge, samples: np.ndarray) -> None:
        """
        
        Args:
            image (np.ndarray): A 3D array representing the image with shape (C, H, W)
            segments (np.ndarray): A 2D array representing the segments of the image with shape (H, W)
        """
        assert image.shape[0] == 3 and image.ndim == 3, f"Image must be a 3D array with 3 channels (C, H, W). Got {image.shape}"
        assert segments.ndim == 2, f"Segments must be a 2D array (H, W). Got {segments.shape}"
        assert image.shape[1:] == segments.shape, f"Image and segments must have the same shape starting from the second dimension. Got {image.shape} and {segments.shape}"
        assert len(linear_model.coef_) == np.unique(segments).shape[0], f"The length of importances must be equal to the number of segments. Got {len(linear_model.coef_)} and {segments.max() + 1}"


        self.image: Final = image
        self.segments: Final = segments

        self.importances: Final = linear_model.coef_
        self.linear_model: Final = linear_model

        self._samples: Final = samples

        self._sorted_importances: Final = None

    def plot_importances(self) -> None:
        sns.barplot(x=np.arange(len(self.importances)), y=self.importances)
        plt.xlabel("Segment")
        plt.ylabel("Importance")
        plt.title("Importances of each segment")
        plt.show()

    def render(self, type: Literal['smooth', 'standard', 'blend'] = 'standard', positive_only: bool = False, negative_only: bool = False) -> plt.Figure:
        """
        Renders the explanation in a heatmap.
        
        Args:
            type (Literal['smooth', 'positive_only', 'standard', 'blend']): The type of rendering to use. Defaults to 'standard'.
                - 'standard': Renders the explanation as is.
                - 'positive_only': Renders only the positive importances.
                - 'blend': Blends the explanation on top of the image.
                - 'smooth': blend but with a gaussian filter applied to the explanation.
        
        Returns:
            np.ndarray: A 3D array representing the image with the explanation rendered on top of it.
        """

        # mutual exlusion
        assert not (positive_only and negative_only), "positive_only and negative_only cannot be True at the same time."

        explanation = np.zeros_like(self.segments, dtype=np.float32)

        for i, s in enumerate(np.unique(self.segments)):
            if positive_only and self.importances[i] < 0: continue
            if negative_only and self.importances[i] > 0: continue

            explanation[self.segments == s] = -self.importances[i] if negative_only else self.importances[i]

        fig = plt.figure()

        if type == 'standard':
            plt.imshow(explanation, cmap='viridis')
            
        elif type == 'blend':
            plt.imshow(self.image.transpose(1, 2, 0))
            plt.imshow(explanation, cmap='viridis', alpha=0.4 if type == 'blend' else 1)

        elif type == 'smooth':
            explanation = gaussian(explanation, sigma=2)
            plt.imshow(self.image.transpose(1, 2, 0))
            plt.imshow(explanation, cmap='viridis', alpha=0.6)



        return fig


    @property
    def sorted_importances(self) -> List[Tuple[int, float]]:
        """
        Returns a list of tuples containing the segment index and its importance sorted in **descending** order.

        Result is cached after the first call.
        
        Returns:
            List[Tuple[int, float]]: A list of tuples containing the segment index and its importance.
        """
        if self._sorted_importances is None:
            self._sorted_importances = sorted(enumerate(self.importances), key=lambda x: x[1], reverse=True)
        return self._sorted_importances