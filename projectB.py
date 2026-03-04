#!/usr/bin/env python3
import os, time, argparse, inspect
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data as data
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import ddpm as ddpm_lib
import unet as unet_lib
import vae_bernoulli as vae_lib

# fid.py is assumed to be provided in the same folder
from fid import compute_fid

@torch.no_grad()
def collect_real_images(test_loader, n: int, device):
    xs = []
    for x, _ in test_loader:
        # your loader currently returns flattened (B,784) in [-1,1]
        # convert back to (B,1,28,28)
        x_img = x.view(-1, 1, 28, 28)
        xs.append(x_img)
        if sum(t.shape[0] for t in xs) >= n:
            break
    x_real = torch.cat(xs, dim=0)[:n].to(device)
    return x_real
# -------------------------
# Data: standard MNIST
# -------------------------
def mnist_ddpm_loaders(batch_size: int, data_dir="data"):
    # Week 3 recommended preprocessing: dequantize + map to [-1,1] + flatten
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.rand_like(x) / 255.0),
        transforms.Lambda(lambda x: (x - 0.5) * 2.0),
        transforms.Lambda(lambda x: x.flatten()),  # (784,)
    ])
    tr = datasets.MNIST(data_dir, train=True, download=True, transform=tfm)
    te = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)
    tr_loader = data.DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=True)
    te_loader = data.DataLoader(te, batch_size=batch_size, shuffle=False)
    return tr_loader, te_loader


# -------------------------
# Beta-VAE with Gaussian likelihood
# -------------------------
class GaussianDecoder(nn.Module):
    """
    p(x|z) = Normal(mean(z), sigma)
    x is flattened (784,) in [-1,1]. We wrap as Independent with event_dim=1.
    """
    def __init__(self, decoder_net: nn.Module, learn_log_std: bool = True, init_std: float = 0.2):
        super().__init__()
        self.decoder_net = decoder_net
        if learn_log_std:
            self.log_std = nn.Parameter(torch.tensor(np.log(init_std), dtype=torch.float32))
        else:
            self.register_buffer("log_std", torch.tensor(np.log(init_std), dtype=torch.float32))

    def forward(self, z):
        mean = self.decoder_net(z)               # (B,784)
        std = torch.exp(self.log_std)            # scalar
        return td.Independent(td.Normal(mean, std), 1)


class BetaVAE(nn.Module):
    """
    ELBO_beta = E_q[log p(x|z)] - beta * KL(q||p)
    Uses Gaussian prior + Gaussian encoder from vae_bernoulli.py.
    """
    def __init__(self, prior, encoder, decoder, beta: float):
        super().__init__()
        self.prior = prior
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

    def elbo(self, x):
        q = self.encoder(x)             # q(z|x)
        z = q.rsample()                 # (B,M)
        recon = self.decoder(z).log_prob(x)   # (B,)
        kl = td.kl_divergence(q, self.prior())# (B,)
        return (recon - self.beta * kl).mean()

    def forward(self, x):
        return -self.elbo(x)

    @torch.no_grad()
    def encode(self, x, sample=True):
        q = self.encoder(x)
        return q.rsample() if sample else q.mean

    @torch.no_grad()
    def decode_mean(self, z):
        # Return mean image in [-1,1] as (B,784)
        mean = self.decoder.decoder_net(z)
        return mean


def build_beta_vae(M: int, beta: float, device):
    prior = vae_lib.GaussianPrior(M)

    encoder_net = nn.Sequential(
        nn.Linear(784, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 2*M),
    )
    decoder_net = nn.Sequential(
        nn.Linear(M, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 784),
    )

    encoder = vae_lib.GaussianEncoder(encoder_net)
    decoder = GaussianDecoder(decoder_net, learn_log_std=True, init_std=0.2)

    model = BetaVAE(prior, encoder, decoder, beta=beta).to(device)
    return model


def train_vae(model, loader, epochs, device, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        for x, _ in loader:
            x = x.to(device)
            opt.zero_grad()
            loss = model(x)
            loss.backward()
            opt.step()


# -------------------------
# Latent dataset + latent DDPM
# -------------------------
@torch.no_grad()
def encode_dataset_to_latents(vae: BetaVAE, loader, device, max_points=None):
    vae.eval()
    Z, Y = [], []
    n = 0
    for x, y in loader:
        x = x.to(device)
        z = vae.encode(x, sample=True)   # (B,M)
        Z.append(z.cpu())
        Y.append(y.cpu())
        n += z.shape[0]
        if max_points is not None and n >= max_points:
            break
    Z = torch.cat(Z, dim=0)
    Y = torch.cat(Y, dim=0)
    if max_points is not None:
        Z, Y = Z[:max_points], Y[:max_points]
    return Z, Y


def train_latent_ddpm(Z_train: torch.Tensor, T: int, hidden: int, epochs: int, device, lr=1e-3):
    # Z_train: (N,M)
    loader = data.DataLoader(data.TensorDataset(Z_train), batch_size=1024, shuffle=True, drop_last=True)

    M = Z_train.shape[1]
    net = ddpm_lib.FcNetwork(M, hidden).to(device)
    model = ddpm_lib.DDPM(net, T=T).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ddpm_lib.train(model, opt, loader, epochs=epochs, device=device)
    return model


# -------------------------
# Sampling + timing + plots
# -------------------------
@torch.no_grad()
def sample_pixel_ddpm(ddpm: ddpm_lib.DDPM, n: int, device):
    x = ddpm.sample((n, 784)).to(device)         # [-1,1]
    x = (x / 2.0 + 0.5).clamp(0, 1)              # [0,1]
    return x.view(n, 1, 28, 28)

@torch.no_grad()
def sample_latent_ddpm(lat_ddpm: ddpm_lib.DDPM, vae: BetaVAE, n: int, device):
    z = lat_ddpm.sample((n, vae.prior.M)).to(device)
    x = vae.decode_mean(z)                       # (n,784) in [-1,1]
    x = (x / 2.0 + 0.5).clamp(0, 1)
    return x.view(n, 1, 28, 28)

@torch.no_grad()
def sample_vae(vae: BetaVAE, n: int, device):
    z = vae.prior().sample((n,)).to(device)
    x = vae.decode_mean(z)
    x = (x / 2.0 + 0.5).clamp(0, 1)
    return x.view(n, 1, 28, 28)

def timed_samples(sample_fn, n: int, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = sample_fn(n)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return n / (t1 - t0)

def save_4_samples(imgs, path):
    grid = make_grid(imgs[:4], nrow=4)
    save_image(grid, path)

def pca_plot_three(Z_prior, Z_agg, Z_lat, outpath, title):
    Z_prior = Z_prior.numpy()
    Z_agg = Z_agg.numpy()
    Z_lat = Z_lat.numpy()

    if Z_agg.shape[1] > 2:
        pca = PCA(n_components=2).fit(Z_agg)
        A = pca.transform(Z_prior)
        B = pca.transform(Z_agg)
        C = pca.transform(Z_lat)
    else:
        A, B, C = Z_prior, Z_agg, Z_lat

    plt.figure(figsize=(7, 6))

    # Prior: faint density background
    plt.hexbin(
        A[:, 0], A[:, 1],
        gridsize=80,
        bins="log",
        mincnt=1,
        linewidths=0,
        alpha=0.35,
        cmap="Greys",
        label="VAE prior p(z)",
    )

    # Aggregate posterior: medium opacity on top
    plt.scatter(
        B[:, 0], B[:, 1],
        s=6,
        alpha=0.55,
        label="agg posterior q(z)",
    )

    # Latent DDPM samples: highest opacity so it's clearly distinguishable
    plt.scatter(
        C[:, 0], C[:, 1],
        s=6,
        alpha=0.85,
        label="latent DDPM samples",
    )

    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cpu","cuda","mps"])
    ap.add_argument("--outdir", default="out_partB")
    ap.add_argument("--batch-size", type=int, default=64)

    ap.add_argument("--ddpm-epochs", type=int, default=50)
    ap.add_argument("--ddpm-T", type=int, default=1000)

    ap.add_argument("--vae-epochs", type=int, default=20)
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--betas", nargs="+", type=float, default=[1e-6, 1e-3, 1e-1])

    ap.add_argument("--latent-ddpm-epochs", type=int, default=50)
    ap.add_argument("--latent-ddpm-T", type=int, default=500)
    ap.add_argument("--latent-hidden", type=int, default=128)

    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.outdir, exist_ok=True)

    # Data
    tr_loader, te_loader = mnist_ddpm_loaders(args.batch_size)

    # ---------------- Pixel DDPM ----------------
    unet = unet_lib.Unet().to(device)
    pix_ddpm = ddpm_lib.DDPM(unet, T=args.ddpm_T).to(device)
    opt = torch.optim.Adam(pix_ddpm.parameters(), lr=1e-4)
    ddpm_lib.train(pix_ddpm, opt, tr_loader, epochs=args.ddpm_epochs, device=device)

    torch.save(pix_ddpm.state_dict(), os.path.join(args.outdir, "ddpm_pixel.pt"))
    pix_imgs = sample_pixel_ddpm(pix_ddpm, 16, device)
    save_4_samples(pix_imgs, os.path.join(args.outdir, "samples_ddpm.png"))

    # ---------------- Loop betas for latent DDPM ----------------
    for beta in args.betas:
        tag = f"beta{beta:g}".replace(".","p")
        print(f"\n=== beta = {beta} ===")

        # Train beta-VAE
        vae = build_beta_vae(args.latent_dim, beta, device)
        train_vae(vae, tr_loader, epochs=args.vae_epochs, device=device, lr=1e-3)
        torch.save(vae.state_dict(), os.path.join(args.outdir, f"beta_vae_{tag}.pt"))

        vae_imgs = sample_vae(vae, 16, device)
        save_4_samples(vae_imgs, os.path.join(args.outdir, f"samples_vae_{tag}.png"))

        # Encode train set to latents for latent DDPM
        Z_train, _ = encode_dataset_to_latents(vae, tr_loader, device, max_points=60000)

        # Train latent DDPM
        lat_ddpm = train_latent_ddpm(Z_train, T=args.latent_ddpm_T, hidden=args.latent_hidden,
                                     epochs=args.latent_ddpm_epochs, device=device, lr=1e-3)
        torch.save(lat_ddpm.state_dict(), os.path.join(args.outdir, f"ddpm_latent_{tag}.pt"))

        lat_imgs = sample_latent_ddpm(lat_ddpm, vae, 16, device)
        save_4_samples(lat_imgs, os.path.join(args.outdir, f"samples_latent_ddpm_{tag}.png"))

        # Latent distribution plot: prior vs agg posterior vs latent ddpm
        Z_prior = vae.prior().sample((20000,)).cpu()
        Z_agg, _ = encode_dataset_to_latents(vae, te_loader, device, max_points=20000)
        Z_lat = lat_ddpm.sample((20000, args.latent_dim)).cpu()
        pca_plot_three(Z_prior, Z_agg, Z_lat,
                       os.path.join(args.outdir, f"latent_dist_{tag}.png"),
                       title=f"Latent distributions (beta={beta})")

        # Timing (samples/sec)
        s_vae = timed_samples(lambda n: sample_vae(vae, n, device), 256, device)
        s_pix = timed_samples(lambda n: sample_pixel_ddpm(pix_ddpm, n, device), 64, device)
        s_lat = timed_samples(lambda n: sample_latent_ddpm(lat_ddpm, vae, n, device), 256, device)
        print(f"samples/sec: VAE={s_vae:.2f}, DDPM={s_pix:.2f}, latentDDPM={s_lat:.2f}")

        # FID (adapt if your fid.py has a different signature)
        # Common pattern: compute_fid(real_loader, sample_fn, device)
        # --- FID (course-provided compute_fid expects tensors in [-1,1]) ---
        try:
            N_FID = 5000  # or 10000 if you can afford it
            x_real = collect_real_images(te_loader, N_FID, device)

            # pixel DDPM: sample returns images in [0,1] in my earlier code;
            # convert back to [-1,1] for FID.
            x_pix = sample_pixel_ddpm(pix_ddpm, N_FID, device)          # (N,1,28,28) in [0,1]
            x_pix = (x_pix * 2.0 - 1.0).to(device)

            x_vae = sample_vae(vae, N_FID, device)                      # (N,1,28,28) in [0,1]
            x_vae = (x_vae * 2.0 - 1.0).to(device)

            x_lat = sample_latent_ddpm(lat_ddpm, vae, N_FID, device)    # (N,1,28,28) in [0,1]
            x_lat = (x_lat * 2.0 - 1.0).to(device)

            fid_pix = compute_fid(x_real, x_pix, device=str(device))
            fid_vae = compute_fid(x_real, x_vae, device=str(device))
            fid_lat = compute_fid(x_real, x_lat, device=str(device))

            print(f"FID: VAE={fid_vae:.2f}, DDPM={fid_pix:.2f}, latentDDPM={fid_lat:.2f}")
        except FileNotFoundError:
            print("FID classifier checkpoint 'mnist_classifier.pth' not found. Put it next to fid.py.")
if __name__ == "__main__":
    main()