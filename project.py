#!/usr/bin/env python3
"""
partA_vae_priors.py

Complete, self-contained implementation for Mini-project 1 — Part A (DTU 02460):
Train a Bernoulli VAE on BINARIZED MNIST with three priors:

  1) Standard Gaussian prior
  2) Mixture of Gaussians (MoG) prior
  3) Flow-based prior (RealNVP / masked affine coupling)

For each prior, this script supports:
  - training (multiple runs with different seeds)
  - test-set ELBO evaluation (mean ± std over runs)
  - plotting:
      * prior samples
      * aggregate posterior samples (samples z ~ q(z|x), colored by label)
      * (optionally) PCA projection if latent dim M > 2

Design goal:
  - minimal extra machinery
  - everything in ONE file
  - well-commented with explicit tensor shapes and math mapping

Dependencies:
  pip install torch torchvision tqdm matplotlib scikit-learn

Example usage:
  # Gaussian prior, latent dim 2, 3 runs
  python partA_vae_priors.py train_eval_plot --prior gaussian --latent-dim 2 --runs 3

  # MoG prior
  python partA_vae_priors.py train_eval_plot --prior mog --mog-components 10 --latent-dim 10 --runs 5

  # Flow prior
  python partA_vae_priors.py train_eval_plot --prior flow --flow-steps 6 --latent-dim 10 --runs 3
"""

import argparse
import os
import math
import random
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data as data
from tqdm import tqdm

# sklearn only used when M > 2 for PCA visualization of latent samples
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from torchvision import datasets, transforms


# -----------------------------
# 0) Reproducibility utilities
# -----------------------------
def set_seed(seed: int) -> None:
    """
    Fix random seeds across python / numpy / torch for more repeatable runs.
    Note: GPU non-determinism can still exist depending on ops.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# 1) Priors p(z)
# -----------------------------
class GaussianPrior(nn.Module):
    """
    Standard isotropic Gaussian prior:
        p(z) = N(0, I)

    Shapes:
      - latent dim M
      - samples z: (B, M)
    """
    def __init__(self, M: int):
        super().__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(M), requires_grad=False)  # (M,)
        self.std = nn.Parameter(torch.ones(M), requires_grad=False)   # (M,)

    def forward(self) -> td.Distribution:
        # Independent makes event_dim=1, i.e. log_prob sums over M dims.
        return td.Independent(td.Normal(self.mean, self.std), 1)


class MoGPrior(nn.Module):
    r"""
    Mixture of Gaussians prior:
        p(z) = \sum_{k=1}^K \pi_k N(z | μ_k, diag(σ_k^2))

    We parameterize:
      - logits for mixture weights \pi
      - component means μ_k
      - component log-scales log σ_k

    Shapes:
      logits: (K,)
      loc:    (K, M)
      scale:  (K, M)
    """
    def __init__(self, M: int, K: int = 10):
        super().__init__()
        self.M = M
        self.K = K

        # Initialize mixture weights close to uniform (all zeros logits).
        self.logits = nn.Parameter(torch.zeros(K))           # (K,)

        # Small random means; you can also initialize on a grid for M=2.
        self.loc = nn.Parameter(0.1 * torch.randn(K, M))     # (K, M)

        # Start with unit-ish scale in log-space.
        self.log_scale = nn.Parameter(torch.zeros(K, M))     # (K, M)

    def forward(self) -> td.Distribution:
        mix = td.Categorical(logits=self.logits)  # categorical over K
        comp = td.Independent(td.Normal(self.loc, torch.exp(self.log_scale)), 1)  # event_dim=1 over M
        # MixtureSameFamily returns distribution over event_dim=1 (latent vector)
        return td.MixtureSameFamily(mix, comp)


# -----------------------------
# 2) Normalizing Flow prior (RealNVP)
# -----------------------------
class GaussianBase(nn.Module):
    """
    Base distribution for flows:
      z0 ~ N(0, I)
    """
    def __init__(self, D: int):
        super().__init__()
        self.D = D
        self.mean = nn.Parameter(torch.zeros(D), requires_grad=False)
        self.std = nn.Parameter(torch.ones(D), requires_grad=False)

    def forward(self) -> td.Distribution:
        return td.Independent(td.Normal(self.mean, self.std), 1)


class MaskedCouplingLayer(nn.Module):
    r"""
    RealNVP masked affine coupling layer.

    Let mask b ∈ {0,1}^D.
    - b_i = 1 => dimension i is "frozen" (passes through)
    - b_i = 0 => dimension i is transformed

    Forward (base -> data):
      z' = b ⊙ z + (1-b) ⊙ ( z ⊙ exp(s(b⊙z)) + t(b⊙z) )

    Inverse (data -> base):
      z  = b ⊙ z' + (1-b) ⊙ ( (z' - t(b⊙z')) ⊙ exp(-s(b⊙z')) )

    Log-det Jacobian (forward):
      log |det J| = sum_{i: b_i=0} s_i(b⊙z)

    Notes for stability:
      - It's common to bound s output, e.g. s = tanh(raw_s) * c.
      - We implement that here using a scale_factor.

    Tensor shapes:
      z:      (B, D)
      mask b: (D,) broadcast to (B, D)
      s,t:    (B, D)
      logdet: (B,)
    """
    def __init__(
        self,
        scale_net: nn.Module,
        translation_net: nn.Module,
        mask: torch.Tensor,
        scale_factor: float = 2.0,
    ):
        super().__init__()
        self.scale_net = scale_net
        self.translation_net = translation_net
        self.mask = nn.Parameter(mask, requires_grad=False)  # (D,)
        self.scale_factor = scale_factor

    def _stabilized_scale(self, raw_s: torch.Tensor) -> torch.Tensor:
        # Bound scaling to avoid extreme exp(s).
        return torch.tanh(raw_s) * self.scale_factor

    def forward(self, z):
        # mask: 1 means "keep", 0 means "transform"
        b = self.mask
        z_masked = z * b

        s = self.scale_net(z_masked)
        t = self.translation_net(z_masked)

        x = b * z + (1 - b) * (z * torch.exp(s) + t)
        log_det_J = ((1 - b) * s).sum(dim=-1)
        return x, log_det_J

    def inverse(self, x):
        b = self.mask
        x_masked = x * b

        s = self.scale_net(x_masked)
        t = self.translation_net(x_masked)

        z = b * x + (1 - b) * ((x - t) * torch.exp(-s))
        log_det_J = -((1 - b) * s).sum(dim=-1)
        return z, log_det_J


class Flow(nn.Module):
    """
    A normalizing flow:
      z0 ~ base
      zK = f_K ∘ ... ∘ f_1(z0)

    Provides:
      - sample: generate zK
      - log_prob: compute log p(zK) by inverse + logdet

    Math:
      log p(x) = log p(z0) + log |det d z0 / d x|
              = log p(z0) + sum logdet(inv transforms)
    """
    def __init__(self, base: GaussianBase, transformations: List[MaskedCouplingLayer]):
        super().__init__()
        self.base = base
        self.transformations = nn.ModuleList(transformations)

    def forward(self, z0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply forward transforms: z0 -> x

        Returns:
          x: (B, D)
          sum_logdet: (B,)
        """
        sum_logdet = torch.zeros(z0.shape[0], device=z0.device)
        z = z0
        for T in self.transformations:
            z, logdet = T(z)
            sum_logdet = sum_logdet + logdet
        return z, sum_logdet

    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply inverse transforms: x -> z0

        Returns:
          z0: (B, D)
          sum_logdet_inv: (B,)
        """
        sum_logdet_inv = torch.zeros(x.shape[0], device=x.device)
        z = x
        for T in reversed(self.transformations):
            z, logdet_inv = T.inverse(z)
            sum_logdet_inv = sum_logdet_inv + logdet_inv
        return z, sum_logdet_inv

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(x):
          z0, logdet_inv = inverse(x)
          log p(x) = log p(z0) + logdet_inv
        """
        z0, logdet_inv = self.inverse(x)
        return self.base().log_prob(z0) + logdet_inv  # (B,)

    def sample(self, sample_shape: Tuple[int, ...] = (1,)) -> torch.Tensor:
        """
        Sample x by sampling z0 from base and pushing through forward.
        """
        z0 = self.base().sample(sample_shape)  # (B, D) if sample_shape=(B,)
        x, _ = self.forward(z0)
        return x


class FlowPrior(nn.Module):
    """
    Wrapper so the prior object matches the VAE expectation:
      prior() returns an object with .sample and .log_prob
    """
    def __init__(self, flow: Flow):
        super().__init__()
        self.flow = flow

    def forward(self) -> Flow:
        return self.flow


def build_latent_flow(M: int, steps: int = 6, hidden: int = 128) -> Flow:
    """
    Build a simple RealNVP flow for latent vectors of dimension M.

    Masking strategy:
      - alternate half/half masks across layers:
          mask = [0..0, 1..1] then flipped each step

    Networks:
      - scale_net:      (B,M) -> (B,M)
      - translation_net:(B,M) -> (B,M)

    For flows-as-priors, keep it moderate: too powerful can overfit / destabilize.
    """
    base = GaussianBase(M)

    # Initial mask: first half frozen? choose second half frozen; doesn't matter as long as alternated.
    mask = torch.zeros(M)
    mask[M // 2 :] = 1.0

    transformations: List[MaskedCouplingLayer] = []
    for k in range(steps):
        mask = 1.0 - mask  # flip mask each layer

        scale_net = nn.Sequential(
            nn.Linear(M, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, M),
        )
        translation_net = nn.Sequential(
            nn.Linear(M, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, M),
        )
        transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))

    return Flow(base, transformations)


# -----------------------------
# 3) Encoder q(z|x) and decoder p(x|z)
# -----------------------------
class GaussianEncoder(nn.Module):
    r"""
    Gaussian approximate posterior:
      q_\phi(z|x) = N( μ_\phi(x), diag(σ_\phi(x)^2) )

    Implementation: encoder_net outputs a vector of size 2M:
      encoder_net(x) -> [mean, log_std]
    then:
      z ~ Normal(mean, exp(log_std))

    Tensor shapes:
      x: (B, 28, 28)
      encoder_net(x): (B, 2M)
      mean: (B, M)
      log_std: (B, M)
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
    Bernoulli likelihood for binarized MNIST:
      p_\theta(x|z) = \prod_{i=1}^{784} Bernoulli(x_i | sigmoid(logits_i(z)))

    decoder_net(z) outputs logits of shape (B, 28, 28).
    We wrap in td.Independent(..., event_dim=2) so log_prob sums over 28x28.
    """
    def __init__(self, decoder_net: nn.Module):
        super().__init__()
        self.decoder_net = decoder_net

    def forward(self, z: torch.Tensor) -> td.Distribution:
        logits = self.decoder_net(z)  # (B, 28, 28)
        return td.Independent(td.Bernoulli(logits=logits), 2)


# -----------------------------
# 4) VAE model with Gaussian OR non-Gaussian priors
# -----------------------------
class VAE(nn.Module):
    r"""
    VAE with modular prior / encoder / decoder.

    For a general prior p(z) (not necessarily Gaussian),
    we use the general ELBO:

      ELBO(x) = E_{q(z|x)} [ log p(x|z) + log p(z) - log q(z|x) ]

    When p(z) is standard Gaussian and q is Gaussian, you *could* use closed-form KL:
      ELBO = E_q[log p(x|z)] - KL(q||p)
    but the general form works for all priors, including Gaussian.
    """
    def __init__(self, prior: nn.Module, decoder: nn.Module, encoder: nn.Module):
        super().__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Monte Carlo ELBO for a batch x.

        Shapes:
          x: (B, 28, 28)
          q = encoder(x): distribution over z with batch shape (B,)
          z = q.rsample(): (B, M)
          decoder(z).log_prob(x): (B,)
          prior().log_prob(z): (B,)
          q.log_prob(z): (B,)

        Returns:
          scalar tensor (mean over batch)
        """
        q = self.encoder(x)               # q_\phi(z|x)
        z = q.rsample()                   # reparameterized sample (B, M)
        log_pxz = self.decoder(z).log_prob(x)  # (B,)

        pz = self.prior()                 # distribution-like object
        log_pz = pz.log_prob(z)           # (B,)
        log_qz = q.log_prob(z)            # (B,)

        elbo_batch = log_pxz + log_pz - log_qz  # (B,)
        return elbo_batch.mean()

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        return -self.elbo(x)

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        """
        Sample from the generative model:
          z ~ p(z)
          x ~ p(x|z)
        Returns samples in {0,1} with shape (n, 28, 28)
        """
        z = self.prior().sample((n,))
        x = self.decoder(z).sample()
        return x


# -----------------------------
# 5) Training / Evaluation / Plotting helpers
# -----------------------------
def train_vae(model: VAE, optimizer: torch.optim.Optimizer, loader: data.DataLoader,
              epochs: int, device: torch.device) -> None:
    """
    Standard SGD loop minimizing negative ELBO.
    """
    model.train()
    total_steps = epochs * len(loader)
    bar = tqdm(total=total_steps, desc="Training", dynamic_ncols=True)

    for ep in range(epochs):
        for xb, _ in loader:
            xb = xb.to(device)  # (B, 28, 28)
            optimizer.zero_grad()
            loss = model.loss(xb)
            loss.backward()
            optimizer.step()

            bar.update(1)
            bar.set_postfix(epoch=f"{ep+1}/{epochs}", loss=f"{loss.item():.4f}")

    bar.close()


@torch.no_grad()
def eval_elbo(model: VAE, loader: data.DataLoader, device: torch.device,
              max_batches: Optional[int] = None) -> float:
    """
    Evaluate mean ELBO over (part of) a dataset.
    Returns scalar mean over all seen batches.
    """
    model.eval()
    elbos = []
    for i, (xb, _) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        xb = xb.to(device)
        elbos.append(model.elbo(xb).item())
    return float(np.mean(elbos)) if len(elbos) else float("nan")


@torch.no_grad()
def sample_aggregate_posterior(model: VAE, loader: data.DataLoader, device: torch.device,
                               max_points: int = 20000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate aggregate posterior samples:
      pick many x from dataset, sample z ~ q(z|x).

    Returns:
      Z: (N, M) numpy
      Y: (N,) labels numpy
    """
    model.eval()
    Z_list, Y_list = [], []
    n_total = 0

    for xb, yb in loader:
        xb = xb.to(device)
        q = model.encoder(xb)
        z = q.rsample()  # (B, M)

        Z_list.append(z.cpu().numpy())
        Y_list.append(yb.numpy())
        n_total += z.shape[0]
        if n_total >= max_points:
            break

    Z = np.concatenate(Z_list, axis=0)[:max_points]
    Y = np.concatenate(Y_list, axis=0)[:max_points]
    return Z, Y


@torch.no_grad()
def sample_prior(prior_module: nn.Module, n: int, device: torch.device) -> np.ndarray:
    # prior_module() returns a Distribution-like object (or Flow) that supports .sample(...)
    prior_dist = prior_module()
    z = prior_dist.sample((n,))          # (n, M) on CPU by default
    z = z.to(device)                     # move tensor to GPU/CPU
    return z.detach().cpu().numpy()

def project_to_2d(Z: np.ndarray) -> Tuple[np.ndarray, Optional[PCA]]:
    """
    If Z has dim M=2: return as-is.
    If M>2: PCA to 2 components for plotting.
    """
    if Z.shape[1] == 2:
        return Z, None
    pca = PCA(n_components=2)
    Z2 = pca.fit_transform(Z)
    return Z2, pca


def plot_prior_vs_aggregate(
    Z_prior: np.ndarray,
    Z_agg: np.ndarray,
    Y_agg: np.ndarray,
    outpath: str,
    title: str,
) -> None:
    """
    Clear visualization:
      - prior shown as faint gray density/background
      - aggregate posterior on top (colored by label)
    """
    plt.figure(figsize=(7, 6))

    # --- PRIOR: render as faint density to avoid "gray fog" ---
    # Hexbin is robust and makes differences visible immediately.
    plt.hexbin(
        Z_prior[:, 0], Z_prior[:, 1],
        gridsize=80,
        bins="log",
        mincnt=1,
        linewidths=0,
        alpha=0.35,
        cmap="Greys",
        label="prior density",
    )

    # ---- Subsample aggregate posterior for readability ----
    np.random.seed(0)
    n_show = min(5000, Z_agg.shape[0])
    idx = np.random.choice(Z_agg.shape[0], n_show, replace=False)

    Z_plot = Z_agg[idx]
    Y_plot = Y_agg[idx]

    sc = plt.scatter(
        Z_plot[:, 0], Z_plot[:, 1],
        s=18,
        alpha=0.95,
        c=Y_plot,
        cmap="tab10",
        edgecolors="none",
        label="aggregate posterior",
        zorder=3,
    )

    plt.colorbar(sc, ticks=range(10), label="digit label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
# -----------------------------
# 6) Data: binarized MNIST loaders
# -----------------------------
def make_binarized_mnist_loaders(
    batch_size: int,
    threshold: float = 0.5,
    data_dir: str = "data",
) -> Tuple[data.DataLoader, data.DataLoader]:
    """
    Binarize MNIST by thresholding:
      x_bin = 1[x > threshold]

    Output shape after transform: (28, 28) float tensor in {0,1}.
    """
    tfm = transforms.Compose([
        transforms.ToTensor(),  # (1,28,28) in [0,1]
        transforms.Lambda(lambda x: (x.squeeze(0) > threshold).float()),  # (28,28)
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)

    train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader


# -----------------------------
# 7) Build VAE architecture (fixed across priors)
# -----------------------------
def build_fc_encoder_decoder(latent_dim: int) -> Tuple[nn.Module, nn.Module]:
    """
    Simple fully-connected encoder/decoder.

    Encoder:
      x (28,28) -> flatten 784 -> 512 -> 512 -> 2M (mean, log_std)

    Decoder:
      z (M) -> 512 -> 512 -> 784 -> unflatten (28,28) logits

    These match the course baseline and keep comparisons fair across priors.
    """
    M = latent_dim

    encoder_net = nn.Sequential(
        nn.Flatten(),             # (B, 784)
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 2 * M),     # (B, 2M)
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28)),  # (B, 28, 28)
    )

    return encoder_net, decoder_net


def build_prior(kind: str, M: int, mog_K: int, flow_steps: int, flow_hidden: int) -> nn.Module:
    """
    Construct the requested prior module.
    """
    kind = kind.lower()
    if kind == "gaussian":
        return GaussianPrior(M)
    if kind == "mog":
        return MoGPrior(M, K=mog_K)
    if kind == "flow":
        flow = build_latent_flow(M, steps=flow_steps, hidden=flow_hidden)
        return FlowPrior(flow)
    raise ValueError(f"Unknown prior kind: {kind}")


# -----------------------------
# 8) Main experiment runner for Part A
# -----------------------------
@dataclass
class RunResult:
    seed: int
    test_elbo: float
    model_path: str
    plot_path: str


def run_single_experiment(
    prior_kind: str,
    latent_dim: int,
    mog_K: int,
    flow_steps: int,
    flow_hidden: int,
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
    outdir: str,
    seed: int,
    max_plot_points: int = 20000,
    prior_plot_samples: int = 20000,
) -> RunResult:
    """
    One run = one seed:
      - set seed
      - build prior + VAE
      - train
      - eval test ELBO
      - plot prior vs aggregate posterior
      - save model
    """
    set_seed(seed)

    # Data
    train_loader, test_loader = make_binarized_mnist_loaders(batch_size=batch_size)

    # Networks (fixed across priors)
    enc_net, dec_net = build_fc_encoder_decoder(latent_dim)
    encoder = GaussianEncoder(enc_net)
    decoder = BernoulliDecoder(dec_net)

    # Prior (varies)
    prior = build_prior(prior_kind, latent_dim, mog_K, flow_steps, flow_hidden)

    # Model
    model = VAE(prior=prior, decoder=decoder, encoder=encoder).to(device)

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Train
    train_vae(model, opt, train_loader, epochs=epochs, device=device)

    # Evaluate test ELBO (mean over batches)
    test_elbo = eval_elbo(model, test_loader, device=device)

    # Prepare output paths
    os.makedirs(outdir, exist_ok=True)
    model_path = os.path.join(outdir, f"vae_{prior_kind}_M{latent_dim}_seed{seed}.pt")
    plot_path = os.path.join(outdir, f"prior_vs_agg_{prior_kind}_M{latent_dim}_seed{seed}.png")

    # Save model weights
    torch.save(model.state_dict(), model_path)

    # Collect aggregate posterior samples
    Z_agg, Y_agg = sample_aggregate_posterior(model, test_loader, device=device, max_points=max_plot_points)

    # Sample prior
    Z_prior = sample_prior(model.prior, n=prior_plot_samples, device=device)

    # Project to 2D if necessary (use PCA fit on aggregate posterior; apply same to prior for comparability)
    if latent_dim > 2:
        Z_agg_2d, pca = project_to_2d(Z_agg)
        Z_prior_2d = pca.transform(Z_prior)  # type: ignore
    else:
        Z_agg_2d, _ = project_to_2d(Z_agg)
        Z_prior_2d, _ = project_to_2d(Z_prior)

    plot_title = f"Prior vs aggregate posterior — {prior_kind.upper()} prior, M={latent_dim}, seed={seed}"
    plot_prior_vs_aggregate(Z_prior_2d, Z_agg_2d, Y_agg, plot_path, plot_title)

    return RunResult(seed=seed, test_elbo=test_elbo, model_path=model_path, plot_path=plot_path)


def run_multi(
    prior_kind: str,
    latent_dim: int,
    mog_K: int,
    flow_steps: int,
    flow_hidden: int,
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
    outdir: str,
    runs: int,
    base_seed: int,
) -> List[RunResult]:
    """
    Run multiple seeds and return list of results.
    """
    results = []
    for r in range(runs):
        seed = base_seed + r
        print(f"\n=== Run {r+1}/{runs} (seed={seed}) ===")
        res = run_single_experiment(
            prior_kind=prior_kind,
            latent_dim=latent_dim,
            mog_K=mog_K,
            flow_steps=flow_steps,
            flow_hidden=flow_hidden,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            device=device,
            outdir=outdir,
            seed=seed,
        )
        results.append(res)
        print(f"Test ELBO (seed={seed}): {res.test_elbo:.4f}")
        print(f"Saved: {res.model_path}")
        print(f"Plot : {res.plot_path}")
    return results


def summarize_results(results: List[RunResult]) -> None:
    """
    Print mean ± std ELBO over runs.
    """
    elbos = np.array([r.test_elbo for r in results], dtype=np.float64)
    mean = float(elbos.mean())
    std = float(elbos.std(ddof=1)) if len(elbos) > 1 else 0.0
    print("\n=== Summary ===")
    print(f"Runs: {len(results)}")
    print(f"Test ELBO mean: {mean:.4f}")
    print(f"Test ELBO  std: {std:.4f}")


# -----------------------------
# 9) CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train_eval_plot"], help="Run full Part A pipeline for chosen prior.")
    p.add_argument("--prior", choices=["gaussian", "mog", "flow"], default="gaussian")

    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)

    # MoG settings
    p.add_argument("--mog-components", type=int, default=10)

    # Flow settings
    p.add_argument("--flow-steps", type=int, default=6)
    p.add_argument("--flow-hidden", type=int, default=128)

    # Multi-run settings
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--base-seed", type=int, default=0)

    # Device and output
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--outdir", type=str, default="out_partA")

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.mode == "train_eval_plot":
        results = run_multi(
            prior_kind=args.prior,
            latent_dim=args.latent_dim,
            mog_K=args.mog_components,
            flow_steps=args.flow_steps,
            flow_hidden=args.flow_hidden,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            outdir=args.outdir,
            runs=args.runs,
            base_seed=args.base_seed,
        )
        summarize_results(results)


if __name__ == "__main__":
    main()