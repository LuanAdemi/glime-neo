from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Final, List, Tuple, Type, override
from samplers import AbstractSampler
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import seaborn as sns

import pickle as pkl

from lime import ImageExplainer, ImageExplanation

from sklearn.decomposition import PCA

from sklearn.linear_model import Ridge

import matplotlib.pyplot as plt

class AbstractMetric:
    """
    Abstract class for a metric.

    A metric object evaluates a sampler and returns a result. The result can be anything, but it should be
    a single object that can be used to compare different samplers. For example, a metric could return a list of z-scores
    for a sampler. The metric should be able to render itself by for example a plot.

    The call order of the methods is as follows:
    evaluate() -> render() and result
    """
    def __init__(self, instance: np.ndarray, sampler: AbstractSampler, num_samples: int):
        assert instance.shape[0] == 3 and instance.ndim == 3, "Instance should be a 3D tensor (C, H, W)"
        self.instance: Final = instance
        self.sampler: Final = sampler
        self.num_samples: Final = num_samples

    @abstractmethod
    def render(self, save_path: str = None) -> plt.Figure:
        pass

    @abstractmethod
    def evaluate(self, progress: bool = False) -> None:
        pass

    def save(self, root: str, name: str) -> None:
        assert self.result is not None, "Result not calculated yet."
        assert root.endswith("/"), "Root should end with a /"
        assert not "/" in name, "Name should not contain /"

        self.render(root + name + "_plot.png")
        with open(root + name + "_results.pkl", "wb") as f:
            pkl.dump(self.result, f)

    @property
    def result(self) -> Any:
        pass


class ZScoreMetric(AbstractMetric):
    """
    Metric that calculates the z-scores of the samples generated by a sampler. It captures "how far away" the samples are from the
    training dataset distribution.

    We first calculate the mean and standard deviation of the training dataset, and then calculate the z-scores of the samples generated
    by the sampler. The z-scores are calculated as (x - mu) / sigma, where x is the sample, mu is the mean of the training dataset and
    sigma is the standard deviation of the training dataset.
    """
    def __init__(self, instance: np.ndarray, sampler: AbstractSampler, num_samples: int, dataset: DataLoader):
        super().__init__(instance, sampler, num_samples)
        self.dataset = dataset
        self.mean = None
        self.std = None
        self.z_scores = None

    def _calculate_mean_and_std(self, progress=False):
        assert self.mean is None and self.std is None, "Mean and std already calculated. This call shouldn't happen."
        
        # calculate mean and std of the dataset
        mean = np.zeros_like(self.instance)
        std = np.zeros_like(self.instance)

        for (x, _) in tqdm(self.dataset, desc="Calculating mean and std", disable=not progress):
            mean += x
            
        self.mean = mean / len(self.dataset)

        for (x, _) in tqdm(self.dataset, desc="Calculating mean and std", disable=not progress):
            std += (x - self.mean) ** 2

        self.std = np.sqrt(std / len(self.dataset))


    @staticmethod
    def _calculate_mean_z_scores(sample: np.ndarray, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
        return ((sample[1] - mean) / std).mean()

    @override
    def evaluate(self, progress=False):
        self._calculate_mean_and_std(progress)

        z_scores = []

        with ThreadPoolExecutor() as executor:
            futures = []
            for sample in self.sampler.sample(self.num_samples, progress=progress):
                futures.append(executor.submit(self._calculate_mean_z_scores, sample, self.mean, self.std))
            
            for future in tqdm(futures, desc="Calculating z-scores", disable=not progress):
                z_scores.append(future.result())

        self.z_scores = np.array(z_scores)

    @override
    def render(self, save_path: str = None):
        assert self.z_scores is not None, "Z-scores not calculated yet."

        fig = plt.figure()
        sns.histplot(self.z_scores, bins=50)

        plt.xlabel("Z-score")
        plt.ylabel("Frequency")
        plt.title("Z-scores of samples")

        if save_path is not None:
            plt.savefig(save_path)

        return fig

    @override
    @property
    def result(self):
        assert self.z_scores is not None, "Z-scores not calculated yet."
        return self.z_scores

class PredictionDistributionMetric(AbstractMetric):
    def __init__(self, instance: np.ndarray, 
                 sampler: AbstractSampler, 
                 num_samples: int, 
                 model: torch.nn.Module | Callable, 
                 transform: Callable = lambda x: x,
                 activation_fn: Callable = lambda x: x):
        super().__init__(instance, sampler, num_samples)

        self.predictions: np.ndarray = None
        self.variance: np.ndarray = None
        self.means: np.ndarray = None
        self._pca_model: PCA = None
        self.pca: np.ndarray = None
        self.model: Final = model
        self.transform: Final = transform
        self.activation_fn: Final = activation_fn

    @torch.no_grad
    def _inference(self, samples: torch.Tensor) -> torch.Tensor:
        logits = self.model(samples)
        return self.activation_fn(logits).cpu().numpy()
    
    def _pca(self, samples: np.ndarray) -> np.ndarray:
        pca = PCA(n_components=2)
        pca.fit(samples)
        self._pca_model = pca
        return pca.transform(samples)
    
    @override
    def evaluate(self, progress: bool = False) -> None:
        # generate samples
        samples = self.sampler.sample(self.num_samples, progress=progress)

        # build a batch of perturbed samples (images)
        representations = np.concat([np.expand_dims(sample[1], axis=0) for sample in samples])
        representations = torch.tensor(representations)

        # run inference
        self.predictions = self._inference(representations)

        # calculate variance by class
        self.variance = np.var(self.predictions, axis=0)

        # calculate mean by class
        self.means = np.mean(self.predictions, axis=0)

        # calculate PCA
        self.pca = self._pca(self.predictions)

    @override
    def render(self, save_path: str = None) -> plt.Figure:
        assert self.predictions is not None, "Predictions not calculated yet."

        fig = plt.figure(figsize=(15, 5))

        # boxplot of the predictions for each class
        sns.violinplot(data=self.predictions, density_norm="width", inner="quartile", color="steelblue")

        plt.xlabel("Class")
        plt.ylabel("Predictions")
        plt.title(f"Prediction distribution by class (N={self.num_samples})")

        if save_path is not None:
            plt.savefig(save_path)

        return fig
    
    def plot_pca(self, save_path: str = None) -> plt.Figure:
        assert self.pca is not None, "PCA not calculated yet."

        fig = plt.figure(figsize=(10, 10))

        sns.scatterplot(x=self.pca[:, 0], y=self.pca[:, 1])
        sns.rugplot(x=self.pca[:, 0], alpha=0.5)
        sns.rugplot(y=self.pca[:, 1], alpha=0.5)

        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.title("PCA of predictions")

        if save_path is not None:
            plt.savefig(save_path)

        return fig

    @override
    @property
    def result(self) -> np.ndarray:
        assert self.predictions is not None, "Predictions not calculated yet."
        return self.predictions, self.variance, self.means, self.pca


class ExplanationFidelityMetric(AbstractMetric):
    def __init__(self, instance: np.ndarray,
                 instance_cls: int,
                 sampler: AbstractSampler,
                 num_samples: int,
                 model: torch.nn.Module | Callable,
                 segmentation_fn: Callable,
                 transform: Callable = lambda x: x,
                 activation_fn: Callable = lambda x: x
                 ): 
        super().__init__(instance, sampler, num_samples)

        self.model = model
        self.transform = transform
        self.activation_fn = activation_fn

        self.sampler: Final = sampler
        self.explainer: Final = ImageExplainer(model, segmentation_fn, sampler)

        self._instance_cls: Final = instance_cls
        self._explanation: ImageExplanation = None

        self.fidelity: Final = None
        self._model_predictions: Final = None
        self._lime_predictions: Final
    
    @torch.no_grad
    def _inference_model(self, samples: torch.Tensor) -> np.ndarray:
        logits = self.model(samples)
        return self.activation_fn(logits).cpu().numpy()
    
    def _inference_lime_model(self, samples: np.ndarray) -> np.ndarray:
        return self._explanation.linear_model.predict(samples)
    
    @override
    def evaluate(self, progress: bool = False) -> None:

        # explain the instance
        self._explanation: ImageExplanation = self.explainer.explain(self.instance, self._instance_cls, self.num_samples, progress=progress)

        # generate samples
        samples = self.sampler.sample(self.num_samples, progress=progress)

        # build a batch of perturbed samples (images)
        representations = np.concat([np.expand_dims(sample[0], axis=0) for sample in samples])
        perturbations = np.concat([np.expand_dims(sample[1], axis=0) for sample in samples])
        perturbations = torch.tensor(perturbations)

        # run inference on the model we are trying to explain
        self.model_predictions = self._inference_model(perturbations)[: , self._instance_cls]

        # run inference on LIME model
        self.lime_predictions = self._inference_lime_model(representations)

        # calculate fidelity
        self.fidelity = np.mean(np.abs(self.model_predictions - self.lime_predictions))
    

    @override
    @property
    def result(self) -> np.ndarray:
        assert self.fidelity is not None, "Fidelity not calculated yet."
        return self.fidelity
    

class DescriptiveAccuracy(AbstractMetric):
    def __init__(self, instance: np.ndarray,
                 instance_cls: int,
                 sampler: AbstractSampler,
                 num_samples: int,
                 model: torch.nn.Module | Callable,
                 segmentation_fn: Callable,
                 transform: Callable = lambda x: x,
                 activation_fn: Callable = lambda x: x
                 ): 
        super().__init__(instance, sampler, num_samples)

        self.model = model
        self.transform = transform
        self.activation_fn = activation_fn

        self.sampler: Final = sampler
        self.explainer: Final = ImageExplainer(model, segmentation_fn, sampler)

        self._instance_cls: Final = instance_cls
        self._explanation: ImageExplanation = None

        self._drawdown_probs: List[float] = None

    @torch.no_grad
    def _inference(self, samples: torch.Tensor) -> np.ndarray:
        logits = self.model(samples)
        return self.activation_fn(logits).cpu().numpy()


    @property
    def drawdown_probs(self):
        assert self._explanation is not None, "Explanation not calculated yet."
        if self._drawdown_probs is None:
            segments = self._explanation.segments

            self._drawdown_probs = []

            img = self.instance.copy()
            
            # go through every segment in decreasing order of importance
            for (idx, imp) in self._explanation.sorted_importances:
                if imp < 0:
                    break

                # zero out the segment
                img[:, segments == idx+1] = 0


                # collect the probabilities
                probs = self._inference(torch.from_numpy(img).unsqueeze(0))
                self._drawdown_probs.append(probs[0, self._instance_cls].item())

        return self._drawdown_probs

    
    @property
    def monotonicity(self) -> float:
        assert self._explanation is not None, "Explanation not calculated yet."

        # calculate the pearson correlation between the importance and the drawdown probabilities
        return np.corrcoef([t[1] for t in self._explanation.sorted_importances if t[1] >= 0], self.drawdown_probs)[0, 1]

    def draw_down_plot(self, save_path: str = None) -> plt.Figure:
        assert self._explanation is not None, "Explanation not calculated yet."

        fig = plt.figure(figsize=(10, 5))

        sns.despine()

        X = []
        Y = []

        for t in self._explanation.sorted_importances:
            if t[1] < 0:
                break
            X.append(t[0])
            Y.append(t[1])

        bar1 = sns.barplot(x=X, y=Y, order=X, color='steelblue', label="Importance", legend=False)
        for i, bar in enumerate(bar1.patches):
            bar.set_edgecolor("black")

        plt.ylabel("Importance")

        ax2 = plt.twinx()
        bar2 = sns.barplot(x=X, y=self.drawdown_probs, ax=ax2, order=X, color="coral", label="Probability", legend=False)

        for i, bar in enumerate(bar2.patches):
            bar.set_hatch("/")
            bar.set_edgecolor("black")

        plt.xlabel("Segment")
        plt.ylabel("Probability")
        plt.title("Drawdown probability by segment with positive importance")

        plt.legend(handles=[bar1.patches[0], bar2.patches[0]], labels=["Importance", "Probability"], loc="upper right", title=f"Monotonicity: {self.monotonicity:.3f}")

        if save_path is not None:
            plt.savefig(save_path)

        return fig

    def evaluate(self, progress: bool = False) -> None:
        # generate explanation
        self._explanation: ImageExplanation = self.explainer.explain(self.instance, self._instance_cls, self.num_samples, progress=progress)

        # get the sorted importances and segment ids
        self._sorted_importances = self._explanation.sorted_importances


class SamplerEvaluation:
    """
    Class to evaluate a sampler.

    Note that that an evaluation object is tied to a specific evaluation run, so all attributes are declared as final.
    """
    def __init__(self, sampler: AbstractSampler, num_samples: int, metrics: List[AbstractMetric]):
        self.sampler: Final = sampler
        self.num_samples: Final = num_samples
        self.metrics: Final = metrics

    