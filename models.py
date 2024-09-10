from abc import abstractmethod, ABC
import torch
import torchvision as tv
import numpy as np
import normflows as nf

from matplotlib import pyplot as plt
from tqdm import trange

class LatentModel(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def encode(self, x):
        pass

    @abstractmethod
    def decode(self, z):
        pass

class GLOW(LatentModel):
    def __init__(self, L, K, input_shape, num_classes, hidden_channels=256, split_mode='channel', scale=True) -> None:
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_glow(L, K, input_shape, num_classes, hidden_channels, split_mode, scale).to(self.device)

    def _build_glow(self, L, K, input_shape, num_classes, hidden_channels, split_mode, scale):
        # from the glow_colab notebook from the normflows library
        q0 = []
        merges = []
        flows = []
        for i in range(L):
            flows_ = []
            for j in range(K):
                flows_ += [nf.flows.GlowBlock(input_shape[0] * 2 ** (L + 1 - i), hidden_channels,
                                            split_mode=split_mode, scale=scale)]
            flows_ += [nf.flows.Squeeze()]
            flows += [flows_]
            if i > 0:
                merges += [nf.flows.Merge()]
                latent_shape = (input_shape[0] * 2 ** (L - i), input_shape[1] // 2 ** (L - i), 
                                input_shape[2] // 2 ** (L - i))
            else:
                latent_shape = (input_shape[0] * 2 ** (L + 1), input_shape[1] // 2 ** L, 
                                input_shape[2] // 2 ** L)
            q0 += [nf.distributions.ClassCondDiagGaussian(latent_shape, num_classes)]


        # Construct flow model with the multiscale architecture
        model = nf.MultiscaleFlow(q0, flows, merges)
        return model
    
    def fit(self, train_loader, max_iters=10000, lr=1e-3, weight_decay=1e-5, report=True):
        loss_history = np.array([])
        optimizer = torch.optim.Adamax(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        train_iter = iter(train_loader)

        with trange(max_iters) as t:
            for i in t:
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x, y = next(train_iter)
                
                optimizer.zero_grad()
                x, y = x.to(self.device), y.to(self.device).float()
                
                loss = self.model.forward_kld(x)

                if not (torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()
                    optimizer.step()

                loss_history = np.append(loss_history, loss.detach().to('cpu').numpy())
                t.set_postfix(loss=loss.item())

        if report:
            plt.plot(loss_history)
            plt.show()

    def encode(self, x):
        x = torch.tensor(x).to(self.device)
        return self.model.inverse_and_log_det(x)[0]
    
    def decode(self, z):
        return self.model.forward_and_log_det(z)