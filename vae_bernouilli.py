import torch
import torch.nn as nn
import torch.distributions as td

class GaussianPrior(nn.Module):
    """
    Standard isotropic Gaussian prior:
        p(z) = N(0, I)
    """
    def __init__(self, M: int):
        super().__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(M), requires_grad=False)

    def forward(self) -> td.Distribution:
        return td.Independent(td.Normal(self.mean, self.std), 1)

class GaussianEncoder(nn.Module):
    r"""
    q(z|x) = N(mean(x), diag(std(x)^2))
    encoder_net(x) outputs 2M numbers: [mean, log_std]
    """
    def __init__(self, encoder_net: nn.Module):
        super().__init__()
        self.encoder_net = encoder_net

    def forward(self, x: torch.Tensor) -> td.Distribution:
        mean, log_std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        std = torch.exp(log_std)
        return td.Independent(td.Normal(mean, std), 1)

class BernoulliDecoder(nn.Module):
    r"""
    p(x|z) = Bernoulli(logits(z)) for binarized MNIST
    decoder_net(z) returns logits shaped (B, 28, 28)
    """
    def __init__(self, decoder_net: nn.Module):
        super().__init__()
        self.decoder_net = decoder_net

    def forward(self, z: torch.Tensor) -> td.Distribution:
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)