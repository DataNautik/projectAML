#!/usr/bin/env python3
"""
partA_runner.py

Run Mini-project 1 (02460) — Part A using the existing course files in the same folder:

  - vae_bernoulli.py  (encoder/decoder modules + training loop style)
  - flow.py           (Flow, GaussianBase, MaskedCouplingLayer)  [must be completed]
  - (optional) any other helpers you already have

This file DOES NOT rewrite those libraries. It imports them and adds only the minimal glue code:
  - MoG prior module (small)
  - FlowPrior wrapper (small)
  - VAE ELBO for non-Gaussian priors (general ELBO)
  - test ELBO evaluation
  - aggregate posterior sampling + PCA plot
  - prior vs aggregate posterior plot
  - multiple runs with mean ± std

Usage examples (from the folder containing the .py files):
  python partA_runner.py --prior gaussian --latent-dim 2 --runs 3 --epochs 10 --device cpu
  python partA_runner.py --prior mog --latent-dim 10 --mog-components 10 --runs 5 --epochs 10 --device cuda
  python partA_runner.py --prior flow --latent-dim 10 --flow-steps 6 --runs 3 --epochs 10 --device cuda

Outputs saved in --outdir (default: out_partA):
  - models
  - prior-vs-aggregate plots
  - printed mean ± std test ELBO

Requirements:
  pip install torch torchvision tqdm matplotlib scikit-learn
"""

import os
import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ------------------------------------------------------------
# Import the provided course code instead of rewriting it
# ------------------------------------------------------------
import vae_bernoulli as vae_lib
import flow as flow_lib

class GaussianPrior(nn.Module):
    """
    Standard normal prior p(z) = N(0, I)
    """

    def __init__(self, M):
        super().__init__()
        self.M = M

        self.register_buffer("mean", torch.zeros(M))
        self.register_buffer("std", torch.ones(M))

    def forward(self):
        return torch.distributions.Independent(
            torch.distributions.Normal(self.mean, self.std),
            1
        )
# ============================================================
# 0) Reproducibility
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 1) Minimal additional priors
# ============================================================

class MoGPrior(nn.Module):
    r"""
    Mixture of Gaussians prior:
      p(z) = \sum_{k=1}^K \pi_k N(z | μ_k, diag(σ_k^2))

    Minimal module: returns a torch.distributions object with .sample and .log_prob.
    """
    def __init__(self, M: int, K: int = 10):
        super().__init__()
        self.M, self.K = M, K
        self.logits = nn.Parameter(torch.zeros(K))              # (K,)
        self.loc = nn.Parameter(0.1 * torch.randn(K, M))        # (K, M)
        self.log_scale = nn.Parameter(torch.zeros(K, M))        # (K, M)

    def forward(self) -> td.Distribution:
        mix = td.Categorical(logits=self.logits)
        comp = td.Independent(td.Normal(self.loc, torch.exp(self.log_scale)), 1)
        return td.MixtureSameFamily(mix, comp)


class FlowPrior(nn.Module):
    """
    Wrap a flow_lib.Flow so it can be used as a VAE prior:
      prior().sample((n,)) and prior().log_prob(z)
    """
    def __init__(self, flow_model: flow_lib.Flow):
        super().__init__()
        self.flow_model = flow_model

    def forward(self):
        # flow_lib.Flow already has .sample(sample_shape) and .log_prob(x)
        return self.flow_model


def build_latent_flow_prior(M: int, steps: int = 6, hidden: int = 128) -> FlowPrior:
    """
    Construct a RealNVP flow in latent space using flow.py classes.
    Requires that flow.py's MaskedCouplingLayer forward/inverse are implemented.
    """
    base = flow_lib.GaussianBase(M)

    # Alternating half/half masks (vector shape (M,))
    mask = torch.zeros(M)
    mask[M // 2 :] = 1.0

    transformations = []
    for _ in range(steps):
        mask = 1.0 - mask

        scale_net = nn.Sequential(
            nn.Linear(M, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, M),
            nn.Tanh(),  # stability (recommended in week 2)
        )
        translation_net = nn.Sequential(
            nn.Linear(M, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, M),
        )

        transformations.append(flow_lib.MaskedCouplingLayer(scale_net, translation_net, mask))

    f = flow_lib.Flow(base, transformations)
    return FlowPrior(f)


# ============================================================
# 2) Minimal VAE wrapper with general ELBO (non-Gaussian priors)
# ============================================================
class VAEPartA(nn.Module):
    r"""
    VAE that reuses encoder/decoder modules from vae_bernoulli.py but uses the general ELBO:

      ELBO(x) = E_{q(z|x)} [ log p(x|z) + log p(z) - log q(z|x) ].

    This works for:
      - Gaussian prior
      - MoG prior
      - Flow prior (via log_prob)
    """
    def __init__(self, prior: nn.Module, decoder: nn.Module, encoder: nn.Module):
        super().__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 28, 28)
        Returns scalar mean ELBO over batch.
        """
        q = self.encoder(x)          # distribution q(z|x)
        z = q.rsample()              # (B, M) reparameterized sample

        log_pxz = self.decoder(z).log_prob(x)  # (B,)

        pz = self.prior()            # distribution-like (has .log_prob)
        log_pz = pz.log_prob(z)      # (B,)
        log_qz = q.log_prob(z)       # (B,)

        return (log_pxz + log_pz - log_qz).mean()

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        return -self.elbo(x)


# ============================================================
# 3) Data, evaluation, plotting
# ============================================================
def make_binarized_mnist_loaders(batch_size: int, threshold: float = 0.5, data_dir: str = "data"):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x.squeeze(0) > threshold).float()),  # (28,28) in {0,1}
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader


@torch.no_grad()
def eval_test_elbo(model: VAEPartA, loader, device: torch.device, max_batches: Optional[int] = None) -> float:
    model.eval()
    vals = []
    for i, (x, _) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device)
        vals.append(model.elbo(x).item())
    return float(np.mean(vals)) if vals else float("nan")


@torch.no_grad()
def sample_aggregate_posterior(model: VAEPartA, loader, device: torch.device, max_points: int = 20000):
    model.eval()
    Zs, Ys = [], []
    n = 0
    for x, y in loader:
        x = x.to(device)
        q = model.encoder(x)
        z = q.rsample()  # (B,M)
        Zs.append(z.cpu().numpy())
        Ys.append(y.numpy())
        n += z.shape[0]
        if n >= max_points:
            break
    Z = np.concatenate(Zs, axis=0)[:max_points]
    Y = np.concatenate(Ys, axis=0)[:max_points]
    return Z, Y


@torch.no_grad()
def sample_prior(prior_module: nn.Module, n: int, device: torch.device) -> np.ndarray:
    z = prior_module().sample((n,)).to(device)
    return z.detach().cpu().numpy()


def project_2d(Z: np.ndarray):
    if Z.shape[1] == 2:
        return Z, None
    pca = PCA(n_components=2)
    Z2 = pca.fit_transform(Z)
    return Z2, pca


def plot_prior_vs_agg(Z_prior_2d, Z_agg_2d, Y, outpath, title):
    plt.figure(figsize=(7, 6))

    # Prior as density background (much more readable)
    plt.hexbin(
        Z_prior_2d[:, 0], Z_prior_2d[:, 1],
        gridsize=70, mincnt=1, linewidths=0, alpha=0.9
    )

    # Aggregate posterior on top
    sc = plt.scatter(
        Z_agg_2d[:, 0], Z_agg_2d[:, 1],
        s=4, alpha=0.8, c=Y, cmap="tab10"
    )

    plt.colorbar(sc, ticks=range(10), label="digit label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# ============================================================
# 4) Build encoder/decoder using vae_bernoulli's structure
# ============================================================
def build_encoder_decoder(latent_dim: int):
    """
    Recreate the same FC architecture as in vae_bernoulli.py main section,
    but using its GaussianEncoder and BernoulliDecoder classes.
    """
    M = latent_dim

    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 2 * M),
    )
    decoder_net = nn.Sequential(
        nn.Linear(M, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28)),
    )

    encoder = vae_lib.GaussianEncoder(encoder_net)
    decoder = vae_lib.BernoulliDecoder(decoder_net)
    return encoder, decoder


def build_prior(prior_kind: str, latent_dim: int, mog_K: int, flow_steps: int, flow_hidden: int):
    if prior_kind == "gaussian":
        return vae_lib.GaussianPrior(latent_dim)  # import from vae_bernoulli.py
    if prior_kind == "mog":
        return MoGPrior(latent_dim, K=mog_K)
    if prior_kind == "flow":
        return build_latent_flow_prior(latent_dim, steps=flow_steps, hidden=flow_hidden)
    raise ValueError(f"Unknown prior: {prior_kind}")


# ============================================================
# 5) Train loop (minimal, mirrors vae_bernoulli.py style)
# ============================================================
def train(model: VAEPartA, optimizer, loader, epochs: int, device: torch.device):
    model.train()
    total_steps = epochs * len(loader)
    bar = tqdm(total=total_steps, desc="Training", dynamic_ncols=True)

    for ep in range(epochs):
        for x, _ in loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            bar.update(1)
            bar.set_postfix(epoch=f"{ep+1}/{epochs}", loss=f"{loss.item():.4f}")

    bar.close()


# ============================================================
# 6) Multi-run orchestration
# ============================================================
@dataclass
class RunResult:
    seed: int
    test_elbo: float
    model_path: str
    plot_path: str


def run_one(args, seed: int) -> RunResult:
    set_seed(seed)
    device = torch.device(args.device)

    train_loader, test_loader = make_binarized_mnist_loaders(args.batch_size)

    encoder, decoder = build_encoder_decoder(args.latent_dim)
    prior = build_prior(args.prior, args.latent_dim, args.mog_components, args.flow_steps, args.flow_hidden)

    model = VAEPartA(prior=prior, decoder=decoder, encoder=encoder).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    train(model, opt, train_loader, args.epochs, device=device)
    test_elbo = eval_test_elbo(model, test_loader, device=device)

    os.makedirs(args.outdir, exist_ok=True)
    model_path = os.path.join(args.outdir, f"vae_{args.prior}_M{args.latent_dim}_seed{seed}.pt")
    plot_path = os.path.join(args.outdir, f"prior_vs_agg_{args.prior}_M{args.latent_dim}_seed{seed}.png")
    torch.save(model.state_dict(), model_path)

    # prior vs aggregate posterior plot
    Z_agg, Y = sample_aggregate_posterior(model, test_loader, device=device, max_points=args.plot_points)
    Z_prior = sample_prior(model.prior, n=args.plot_points, device=device)

    if args.latent_dim > 2:
        Z_agg_2d, pca = project_2d(Z_agg)
        Z_prior_2d = pca.transform(Z_prior)  # type: ignore
    else:
        Z_agg_2d, _ = project_2d(Z_agg)
        Z_prior_2d, _ = project_2d(Z_prior)

    title = f"{args.prior.upper()} prior vs aggregate posterior (M={args.latent_dim}, seed={seed})"
    plot_prior_vs_agg(Z_prior_2d, Z_agg_2d, Y, plot_path, title)

    return RunResult(seed=seed, test_elbo=test_elbo, model_path=model_path, plot_path=plot_path)


def summarize(results: List[RunResult]):
    elbos = np.array([r.test_elbo for r in results], dtype=np.float64)
    mean = float(elbos.mean())
    std = float(elbos.std(ddof=1)) if len(elbos) > 1 else 0.0
    print("\n=== Part A summary ===")
    print(f"prior: {results[0].model_path.split('vae_')[1].split('_')[0] if results else 'n/a'}")
    print(f"runs: {len(results)}")
    print(f"test ELBO mean: {mean:.4f}")
    print(f"test ELBO  std: {std:.4f}")


# ============================================================
# 7) CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prior", choices=["gaussian", "mog", "flow"], required=True)

    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--mog-components", type=int, default=10)

    p.add_argument("--flow-steps", type=int, default=6)
    p.add_argument("--flow-hidden", type=int, default=128)

    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--base-seed", type=int, default=0)

    p.add_argument("--plot-points", type=int, default=20000)
    p.add_argument("--outdir", type=str, default="out_partA")

    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    results = []
    for r in range(args.runs):
        seed = args.base_seed + r
        print(f"\n=== Run {r+1}/{args.runs} (seed={seed}) ===")
        res = run_one(args, seed)
        results.append(res)
        print(f"Test ELBO: {res.test_elbo:.4f}")
        print(f"Model: {res.model_path}")
        print(f"Plot : {res.plot_path}")

    summarize(results)


if __name__ == "__main__":
    main()