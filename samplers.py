from abc import abstractmethod, ABC
from copy import copy
import numpy as np
from typing import Callable, List, Tuple, Final

from concurrent.futures import ThreadPoolExecutor

from skimage.filters import gaussian
from tqdm import tqdm

import torch

from models import LatentModel, GLOW


def exponential_kernel(instance, perturbation, kernel_width=0.5):
    # use cosine similarity
    instance = instance.flatten()
    perturbation = perturbation.flatten()
    d = np.dot(instance, perturbation) / (np.linalg.norm(instance) * np.linalg.norm(perturbation))
    return np.exp(-d **2 / kernel_width ** 2)

class AbstractSampler(ABC):
    def __init__(self, instance, segments: np.ndarray, kernel_fn: Callable) -> None:
        self.instance: Final = instance
        self.kernel: Final = kernel_fn
        self.segments: Final = segments

        assert self.segments.ndim == 2, f"Segments must be a 2D array (H, W). Got {self.segments.shape}"
        assert self.instance.shape[1:] == self.segments.shape, f"Image and segments must have the same shape starting from the second dimension. Got {self.instance.shape} and {self.segments.shape}"

    @abstractmethod
    def sample(self, n_samples: np.ndarray, progress=True) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        pass

class LIMESampler(AbstractSampler):
    def __init__(self, instance: np.ndarray, segments: np.ndarray, kernel_fn: Callable, fudged_image: np.ndarray) -> None:
        super().__init__(instance, segments, kernel_fn)
        self.fudged_image: Final = fudged_image

        assert self.fudged_image.shape == self.instance.shape, f"Fudged image must have the same shape as the instance. Got {self.fudged_image.shape} and {self.instance.shape}"

    def generate_sample(instance: np.ndarray, fudged_image: np.ndarray, segments: np.ndarray, n_features: int, kernel: Callable) -> Tuple[np.ndarray, np.ndarray]:
        representation = np.random.randint(0, 2, n_features)
        perturbation = np.zeros_like(instance)

        for i in range(n_features):
            segment_id = i+1  # skimage.segmentation.slic starts from 1
            perturbation[:, segments == segment_id] = fudged_image[:, segments == segment_id] if representation[i] == 0 else instance[:, segments == segment_id]

        weight = kernel(instance, perturbation)

        return representation, perturbation, weight
    
    def sample(self, n_samples: np.ndarray, progress=True) -> List[Tuple[np.ndarray, np.ndarray, float]]:

        samples = []

        n_features = np.unique(self.segments).shape[0]

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(LIMESampler.generate_sample, self.instance, self.fudged_image, self.segments, n_features, self.kernel) for _ in range(n_samples)]
            for future in tqdm(futures, total=n_samples, disable=not progress):
                samples.append(future.result())

        return samples

class BlackOutSampler(LIMESampler):
    def __init__(self, instance: np.ndarray, segments: np.ndarray, kernel_fn: Callable) -> None:
        super().__init__(instance, segments, kernel_fn, np.zeros_like(instance))

class BlurSampler(LIMESampler):
    def __init__(self, instance: np.ndarray,  segments: np.ndarray, kernel_fn: Callable) -> None:
        blurred_instance = gaussian(instance, channel_axis=0, sigma=10) * 255
        super().__init__(instance, segments, kernel_fn, blurred_instance)

class MeanSampler(LIMESampler):
    def __init__(self, instance: np.ndarray, segments: np.ndarray, kernel_fn: Callable) -> None:
        # calculate instance mean for each segment
        mean_instance = np.zeros_like(instance)
        for i in np.unique(segments):
            mean_instance[:, segments == i] = instance[:, segments == i].mean(axis=1).reshape(-1, 1)

        super().__init__(instance, segments, kernel_fn, mean_instance)


class FlowSampler(AbstractSampler):
    def __init__(self, instance: np.ndarray, segments: np.ndarray, kernel_fn: Callable, flow: GLOW, preprocessor: Callable, basis: List[np.ndarray]) -> None:
        super().__init__(instance, segments, kernel_fn)
        self.flow: Final = flow
        self.basis: Final = basis
        self.preprocessor: Final = preprocessor

        self.device = self.flow.device

        self.instance_tensor = self.preprocessor(self.instance.transpose(1, 2, 0)).to(self.device)
        

    def generate_sample(flow, instance_tensor, latent_instance: np.ndarray, basis: np.ndarray, kernel: Callable) -> Tuple[np.ndarray, np.ndarray, float]:
        z = copy(latent_instance)
        z[0] += torch.normal(mean=0, std=0.2, size=z[0].shape).to(z[0].device)

        with torch.no_grad():
            perturbation = flow.decode(z)[0].cpu().numpy()

        weight = kernel(instance_tensor.cpu().numpy(), perturbation)

        return z, perturbation, weight
        
    def sample(self, n_samples: np.ndarray, progress=True) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        with torch.no_grad():
            latent_instance = self.flow.encode(self.instance_tensor.unsqueeze(0))

        samples = []

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(FlowSampler.generate_sample, self.flow, self.instance_tensor, latent_instance, self.basis, self.kernel) for _ in range(n_samples)]
            for future in tqdm(futures, total=n_samples, disable=not progress):
                samples.append(future.result())

            return samples
