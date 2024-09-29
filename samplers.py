from abc import abstractmethod, ABC
from copy import deepcopy
import numpy as np
from typing import Any, Callable, List, Tuple, Final, Literal, override

from concurrent.futures import ThreadPoolExecutor

from skimage.filters import gaussian
from tqdm import tqdm

import torch

from models import LatentModel

from matplotlib import pyplot as plt

import random

def latent_to_device(latent: List[torch.Tensor], device: torch.device) -> List[torch.Tensor]:
    return [l.to(device) for l in latent]

def plot_representation(representation: np.ndarray, segments: np.ndarray) -> None:
    """
    Plot the representation of a perturbation.

    Args:
    representation (np.ndarray): The representation of the perturbation.
    segments (np.ndarray): A 2D array of the same shape as the instance where each pixel is assigned an integer label (H, W).
    title (str): The title of the plot.
    """
    representation_map = np.zeros_like(segments)
    for i, s in enumerate(np.unique(segments)):
        representation_map[segments == s] = representation[i]
        plt.imshow(representation_map, cmap='gray')

def exponential_kernel(instance, perturbation, kernel_width=0.5, distance_metric: Literal['cosine', 'euclidean'] = 'cosine') -> float:
    """
    Compute the similarity between an instance and a perturbation using an exponential kernel.
    
    Args:
    instance (np.ndarray): The instance to be explained.
    perturbation (np.ndarray): The perturbation to be applied to the instance.

    Returns:
    float: The similarity between the instance and the perturbation.
    """
    instance = instance.flatten()
    perturbation = perturbation.flatten()

    if distance_metric == 'cosine':
        d = np.dot(instance, perturbation) / (np.linalg.norm(instance) * np.linalg.norm(perturbation))
    elif distance_metric == 'euclidean':
        d = np.linalg.norm(instance - perturbation)
    return np.exp(-d **2 / kernel_width ** 2)

class AbstractSampler(ABC):
    """
    An abstract class for samplers. Samplers are used to generate samples for LIME explanations.

    Args:
    instance (np.ndarray): The input instance to be explained (C, H, W).
    segments (np.ndarray): A 2D array of the same shape as the instance where each pixel is assigned an integer label (H, W).
    kernel_fn (Callable): A function that computes the similarity between two images.
    """
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
    """
    A boilerplate for standard LIME samplers. This class should not be used directly, but rather subclassed to implement specific sampling strategies.

    Args:
    instance (np.ndarray): The input instance to be explained (C, H, W).
    segments (np.ndarray): A 2D array of the same shape as the instance where each pixel is assigned an integer label (H, W).
    kernel_fn (Callable): A function that computes the similarity between two images.
    fudged_image (np.ndarray): A fudged version of the input instance used to generate samples.
    """
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
    
    @override
    def sample(self, n_samples: np.ndarray, progress=True) -> List[Tuple[np.ndarray, np.ndarray, float]]:

        samples = []

        n_features = np.unique(self.segments).shape[0]

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(LIMESampler.generate_sample, self.instance, self.fudged_image, self.segments, n_features, self.kernel) for _ in range(n_samples)]
            for future in tqdm(futures, total=n_samples, disable=not progress, desc='Generating Samples'):
                samples.append(future.result())

        return samples

class BlackOutSampler(LIMESampler):
    """
    A sampler that generates samples by blacking out segments of the input instance.
    """
    def __init__(self, instance: np.ndarray, segments: np.ndarray, kernel_fn: Callable) -> None:
        super().__init__(instance, segments, kernel_fn, np.zeros_like(instance))

class BlurSampler(LIMESampler):
    """
    A sampler that generates samples by blurring segments of the input instance.

    Args:
    instance (np.ndarray): The input instance to be explained (C, H, W).
    segments (np.ndarray): A 2D array of the same shape as the instance where each pixel is assigned an integer label (H, W).
    kernel_fn (Callable): A function that computes the similarity between two images.
    """
    def __init__(self, instance: np.ndarray,  segments: np.ndarray, kernel_fn: Callable) -> None:
        blurred_instance = gaussian(instance, channel_axis=0, sigma=10) * 255
        super().__init__(instance, segments, kernel_fn, blurred_instance)

class MeanSampler(LIMESampler):
    """
    A sampler that generates samples by replacing segments of the input instance with the mean of the segment.

    Args:
    instance (np.ndarray): The input instance to be explained (C, H, W).
    segments (np.ndarray): A 2D array of the same shape as the instance where each pixel is assigned an integer label (H, W).
    kernel_fn (Callable): A function that computes the similarity between two images.
    """
    def __init__(self, instance: np.ndarray, segments: np.ndarray, kernel_fn: Callable) -> None:
        # calculate instance mean for each segment
        mean_instance = np.zeros_like(instance)
        for i in np.unique(segments):
            mean_instance[:, segments == i] = instance[:, segments == i].mean(axis=1).reshape(-1, 1)

        super().__init__(instance, segments, kernel_fn, mean_instance)


class LatentSampler(AbstractSampler):
    """A sampler that generates samples using latent space manipulation.

    The idea is to encode the input instance into a latent space using a model and manipulate the latent representation to generate samples.
    Manipulation is done using a list of latent vectors (manipulators) added to the latent representation of the instance with random weights.
    
    Args:
    
    instance (np.ndarray): The input instance to be explained (C, H, W).
    segments (np.ndarray): A 2D array of the same shape as the instance where each pixel is assigned an integer label (H, W).
    kernel_fn (Callable): A function that computes the similarity between two images.
    model (LatentModel): A model that can encode and decode instances into a latent space.
    preprocessor (Callable): A function that preprocesses the input instance before feeding it to the model.
    manipulators (List[np.ndarray]): A list of latent vectors to be added to the latent representation of the instance.
    """
    def __init__(self, instance, segments: np.ndarray, kernel_fn: Callable, model: LatentModel, preprocessor: Callable, manipulators: List[np.ndarray], radius: float) -> None:
        super().__init__(instance, segments, kernel_fn)

        self.model: Final = model
        self.manipulators: Final = manipulators
        self.preprocessor: Final = preprocessor
        self.radius: Final = radius

        self.device = self.model.device

        self.instance_tensor = self.preprocessor(self.instance.transpose(1, 2, 0)).to(self.device)

    @staticmethod
    def get_representation(instance: np.ndarray, pertubation: np.ndarray, segments: np.ndarray) -> np.ndarray:
        """
        Compute the representation of a perturbation. 
        
        The representation is the norm (Euclidean distance) between the instance and perturbation segments.

        Args:
        instance (np.ndarray): The instance to be explained (C, H, W).
        perturbation (np.ndarray): The perturbation to be applied to the instance (C, H, W).
        segments (np.ndarray): A 2D array of the same shape as the instance where each pixel is assigned an integer label (H, W).

        Returns:
        np.ndarray: The representation of the perturbation.
        """
        
        representation = np.zeros(np.unique(segments).shape[0])

        for i in np.unique(segments):
            # the current segment
            segment = segments == i

            # Extract the instance and perturbation data for the current segment
            instance_segment = instance[:, segment]
            perturbation_segment = pertubation.squeeze()[:, segment]

             # Use rescaled euclidean distance as the representation
            representation[i-1] = np.linalg.norm(instance_segment.flatten() - perturbation_segment.flatten())
            
        
        return representation
    
    @staticmethod
    def random_walk(latent_instance: torch.Tensor, manipulators: torch.Tensor, radius: float) -> Tuple[torch.Tensor, float]:
        """
        Walk in the latent space in the direction of a random manipulator until a certain distance is reached.

        The distance is controlled by the radius parameter. The radius is the maximum distance that can be traveled in the latent space from the instance.
        
        Args:
        latent_instance (torch.Tensor): The latent representation of the input instance.
        manipulators (List[torch.Tensor]): A list of latent vectors to be added to the latent representation of the instance.
        radius (float): The maximum distance that can be traveled in the latent space from the instance.

        Returns:
        Tuple[torch.Tensor, float]: A tuple containing the new latent representation and the distance traveled.
        """
        z = deepcopy(latent_instance)

        assert len(manipulators) > 0, "At least one manipulator is required."
        assert radius > 0, "The radius must be greater than zero."

        cummulated_direction = torch.zeros_like(z[0])

        while True:
            if torch.norm(cummulated_direction) >= radius:
                break

            # get a random manipulator
            manipulator = random.choice(manipulators)

            # sample a random weight
            r = torch.normal(0, 0.3, size=(1,))

            # walk in the direction of the manipulator
            z[0] += r * manipulator

            # update the distance traveled
            cummulated_direction += r * manipulator

        return z, torch.norm(cummulated_direction).item()
    
    @torch.no_grad()
    def _get_latent_instance(self):
        latent_instance = self.model.encode(self.instance_tensor.unsqueeze(0))
        latent_instance = latent_to_device(latent_instance, self.device)
        return latent_instance
        
    @override
    def sample(self, n_samples: np.ndarray, progress=True) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        latent_instance = self._get_latent_instance()

        latents = []
        samples = []
        mean_distance = 0

        # multi-threaded random walking
        with ThreadPoolExecutor() as executor:
            futures = []
            for _ in range(n_samples):
                futures.append(executor.submit(LatentSampler.random_walk, latent_instance, self.manipulators, self.radius))

            for future in tqdm(futures, desc='Random Walking', total=n_samples, disable=not progress):
                latent, distance = future.result()
                latents.append(latent)
                mean_distance += distance
        
        # single-threaded decoding
        for latent in tqdm(latents, total=n_samples, disable=not progress, desc='Decoding'):
            latent = latent_to_device(latent, self.device)
            
            with torch.no_grad():
                perturbation = self.model.decode(latent)[0].cpu().numpy()

            weight = self.kernel(self.instance, perturbation)

            representation = LatentSampler.get_representation(self.instance, perturbation, self.segments)

            samples.append((representation, perturbation, weight))

        return samples
    

class InpaintingSampler(AbstractSampler):
    def __init__(self, instance, segments: np.ndarray, kernel_fn: Callable[..., Any]) -> None:
        super().__init__(instance, segments, kernel_fn)

class DiffusionSampler(AbstractSampler):
    def __init__(self, instance, segments: np.ndarray, kernel_fn: Callable[..., Any]) -> None:
        super().__init__(instance, segments, kernel_fn)

