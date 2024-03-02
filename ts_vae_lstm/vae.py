# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_vae.ipynb.

# %% auto 0
__all__ = ['num_workers', 'data', 'df', 'window_size', 'end_steps', 'data_windowed', 'val_data_idxs', 'trn_data_idxs', 'val_data',
           'trn_data', 'n_features', 'means', 'stds', 'slice_from', 'dset_trn', 'dset_val', 'batch_size', 'dl_trn',
           'dl_val', 'device', 'TSDataset', 'Encoder', 'StochasticSampler', 'Decoder', 'VAE', 'calculate_smape',
           'loss_func', 'get_similarity', 'validate_epoch']

# %% ../nbs/01_vae.ipynb 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
from .concepts import get_window
from scipy import signal
import os
import math


# %% ../nbs/01_vae.ipynb 4
num_workers = os.cpu_count()
num_workers


# %% ../nbs/01_vae.ipynb 6
data = np.load("../sample_data/nyc_taxi.npz")
for k in data.keys():
    print(k)

# %% ../nbs/01_vae.ipynb 14
df = pd.DataFrame(data["training"], index=data["t_train"], columns=["value"])
df.head(2)


# %% ../nbs/01_vae.ipynb 19
window_size = 48  # one window is a day
end_steps = [es for es in range(window_size, len(df), window_size)]
len(end_steps), end_steps[:3]

# %% ../nbs/01_vae.ipynb 20
data_windowed = [
    {
        "subset": get_window(
            df.values,
            window_size=window_size,
            end_step=end_step,
            indices=list(df.index),
            return_indices=False,
        ),
        "end_step": end_step,
        "start_step": end_step - window_size,
    }
    for end_step in end_steps
]


# %% ../nbs/01_vae.ipynb 23
val_data_idxs = np.random.choice(
    range(len(data_windowed)), size=int(0.1 * len(data_windowed)), replace=False
)
trn_data_idxs = [idx for idx in range(len(data_windowed)) if idx not in val_data_idxs]
len(val_data_idxs), len(trn_data_idxs)


# %% ../nbs/01_vae.ipynb 24
from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score


# %% ../nbs/01_vae.ipynb 25
val_data = [data_windowed[idx] for idx in val_data_idxs]
trn_data = [data_windowed[idx] for idx in trn_data_idxs]

# %% ../nbs/01_vae.ipynb 28
n_features = trn_data[0]["subset"].shape[1]  # - 1
n_features


# %% ../nbs/01_vae.ipynb 30
means = np.zeros((len(trn_data), n_features))  # ((len(trn_data), 4))
stds = np.zeros((len(trn_data), n_features))  # ((len(trn_data), 4))
slice_from = n_features - 1
"""
for i, _trn_data in enumerate(trn_data):
    means[i] = (np.mean(_trn_data["subset"][:, slice_from:], axis=0)).astype(np.float32)
    stds[i] = (np.var(_trn_data["subset"][:, slice_from:], axis=0) ** 0.5).astype(
        np.float32
    )
"""
means = means.mean(0)
stds = stds.mean(0)

means, stds


# %% ../nbs/01_vae.ipynb 33
class TSDataset(Dataset):
    def __init__(self, data, mean, std):
        self.data = data
        self.mean = mean
        self.std = std

    def __getitem__(self, idx):
        # output[channel] = (input[channel] - mean[channel]) / std[channel]
        # ignore the timestamp column
        x = self.data[idx]["subset"][:, slice_from:]  # 1024, 4
        normed_X = ((x - self.mean) / (self.std + 1e-8)).astype(np.float32)
        return torch.as_tensor(normed_X)

    def __len__(self):
        return len(self.data)

# %% ../nbs/01_vae.ipynb 34
dset_trn = TSDataset(trn_data, mean=means, std=stds)
dset_val = TSDataset(val_data, mean=means, std=stds)
# use same stats from training data


# %% ../nbs/01_vae.ipynb 37
batch_size = 8


# %% ../nbs/01_vae.ipynb 38
dl_trn = DataLoader(
    dataset=dset_trn,
    batch_size=batch_size,
    drop_last=True,
    shuffle=True,
    num_workers=num_workers,
)
dl_val = DataLoader(
    dataset=dset_val,
    batch_size=batch_size,
    drop_last=True,
    shuffle=False,
    num_workers=num_workers,
)


# %% ../nbs/01_vae.ipynb 44
# encoder
# l_win to 24, the model would consider each 24-hour period as one sequence.
# pad: if your array is [1, 2, 3] and you symmetrically pad it with 1 unit, the result would be [2, 1, 2, 3, 2].
# xavier_initializer()
# conv 1: num_hidden_units / 16
# conv 2: num_hidden_units / 8
# conv 3: num_hidden_units / 4
# conv 4: num_hidden_units / 1, kernel = 4, 1
# padding : same


class Encoder(nn.Module):
    def __init__(
        self,
        latent_dim=20,
        num_hidden_units=512,
        kernel_size=(3, 1),
        stride=(2, 1),
        act=F.mish,
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=num_hidden_units // 16,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.conv2 = nn.Conv2d(
            in_channels=num_hidden_units // 16,
            out_channels=num_hidden_units // 8,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.conv3 = nn.Conv2d(
            in_channels=num_hidden_units // 8,
            out_channels=num_hidden_units // 4,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.conv4 = nn.Conv2d(
            in_channels=num_hidden_units // 4,
            out_channels=num_hidden_units,
            kernel_size=(4, 1),
            stride=stride,
        )
        self.linear = nn.Linear(
            in_features=num_hidden_units, out_features=num_hidden_units, bias=True
        )
        self.linear_mean = nn.Linear(
            in_features=num_hidden_units, out_features=latent_dim, bias=True
        )
        self.linear_var = nn.Linear(
            in_features=num_hidden_units, out_features=latent_dim, bias=True
        )
        self.act = act
        self.init_weights()

    def forward(self, x):
        x = x.unsqueeze(1)  # 100, 1, 48, 1
        x = self.act(self.conv1(x))  # 100, 32, 23, 1
        x = self.act(self.conv2(x))  # 100, 64, 11, 1
        x = self.act(self.conv3(x))  # 100, 128, 5, 1
        x = self.act(self.conv4(x))  # 100, 512, 1, 1
        x = self.flatten(x)  # 100, 512
        x = self.act(self.linear(x))  # 100, 512
        z_mean = self.linear_mean(x)
        z_log_var = self.linear_var(x)
        return z_mean, z_log_var

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0)


# %% ../nbs/01_vae.ipynb 48
class StochasticSampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.sampler = torch.distributions.Normal(loc=0, scale=1)

    def forward(self, z_mean, z_log_var):
        """Return a normal sample value Z from the latent space given a mean and variance"""
        # z_mean and z_log_var are mean and log-var estimates of the latent space
        # under the assumption that the latent space is a gaussian normal
        device = z_mean.device
        eps = self.sampler.sample(z_mean.shape).squeeze().to(device)
        # print(eps.shape, z_log_var.shape, z_mean.shape)
        return z_mean + torch.exp(0.5 * z_log_var) * eps


# %% ../nbs/01_vae.ipynb 53
# l_win to 24, the model would consider each 24-hour period as one sequence.
# pad: if your array is [1, 2, 3] and you symmetrically pad it with 1 unit, the result would be [2, 1, 2, 3, 2].
# xavier_initializer()
# dense 1: num_hidden_units
# reshape: (bs, 1, 1, num_hidden_units)  -> this is tensorflow notation, channel at end so actually (bs, num_hidden_units, 1, 1)

# conv 2: num_hidden_units, kernel = 1
# reshape: (bs, 4, 1, num_hidden_units / 4)

# conv 3: num_hidden_units / 4, kernel = 3, 1, stride = 1
# permute depth to spatial tf
# reshape: (bs, 8, 1, num_hidden_units / 8),

# conv 4: num_hidden_units / 8,  kernel = 3, 1, stride = 1
# permute depth to spatial tf
# reshape: (bs, 16, 1, num_hidden_units / 16)

# conv 5: num_hidden_units / 16, kernel = 3, 1, stride = 1
# permute depth to spatial tf
# reshape: (bs, num_hidden_units /16, 1,  16)

# conv 6: num_channel, kernel = 9, 1, stride = 1
# reshape: (bs, l_win, num_channel)


class Decoder(nn.Module):
    def __init__(
        self,
        output_shape,
        latent_dim=20,
        num_hidden_units=512,
        kernel_size=(3, 1),
        stride=1,
        act=F.mish,
    ):
        super().__init__()
        self.output_shape = output_shape
        self.linear = nn.Linear(
            in_features=latent_dim, out_features=num_hidden_units, bias=True
        )
        self.dconv1 = nn.ConvTranspose2d(
            in_channels=num_hidden_units,
            out_channels=num_hidden_units // 4,
            kernel_size=(4, 1),
            stride=stride,
        )
        self.dconv2 = nn.ConvTranspose2d(
            in_channels=num_hidden_units // 4,
            out_channels=num_hidden_units // 8,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.dconv3 = nn.ConvTranspose2d(
            in_channels=num_hidden_units // 8,
            out_channels=num_hidden_units // 16,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.dconv4 = nn.ConvTranspose2d(
            in_channels=num_hidden_units // 16,
            out_channels=num_hidden_units // 32,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.dconv5 = nn.ConvTranspose2d(
            in_channels=num_hidden_units // 32,
            out_channels=num_hidden_units // 64,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.flatten = nn.Flatten()
        self.linear_out = nn.Linear(
            in_features=96, out_features=math.prod(output_shape), bias=True
        )

        self.act = act

        self.init_weights()

    def forward(self, x):
        x = self.linear(x)
        x = x[:, :, None, None]
        x = self.act(self.dconv1(x))
        x = self.act(self.dconv2(x))
        x = self.act(self.dconv3(x))
        x = self.act(self.dconv4(x))
        x = self.act(self.dconv5(x))
        x = self.flatten(x)
        x = self.act(self.linear_out(x))
        return self.reshape_to_output(x)

    def reshape_to_output(self, x):
        bs = x.shape[0]
        return x.reshape(bs, *self.output_shape)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0)


# %% ../nbs/01_vae.ipynb 56
class VAE(nn.Module):
    def __init__(self, input_shape, latent_dim=20, act=F.leaky_relu):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim, act=act)
        self.decoder = Decoder(output_shape=input_shape, latent_dim=latent_dim, act=act)
        self.latent_sampler = StochasticSampler()
        self.act = act

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.latent_sampler(z_mean, z_log_var)
        out = self.decoder(z)
        # loss to enforce all possible values are sampled from latent space
        # should be of the size of the batch
        loss_kl = -0.5 * torch.sum(
            1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=-1
        )
        return out, loss_kl

# %% ../nbs/01_vae.ipynb 59
def calculate_smape(predicted, actual):
    with torch.no_grad():
        absolute_percentage_errors = (
            torch.abs(predicted - actual) / (torch.abs(predicted) + torch.abs(actual))
        ) * 100
        return absolute_percentage_errors.mean()

# %% ../nbs/01_vae.ipynb 60
def loss_func(inputs, targets, loss_kl, beta=0.5):
    # targets = torch.where(targets >= 0, 1., 0.)
    bs = inputs.shape[0]
    # loss_kl = loss_kl.unsqueeze(-1)  # add loss_kl per time step.
    loss_reconstruct_numerical = F.huber_loss(
        inputs,
        targets,
        delta=5,
        reduction="none",
    ).sum((1, 2))
    # loss only for the signal
    # loss_reconstruct = calculate_smape(inputs, targets).mean((1, 2))
    # should be of the size of the batch to add losses correctly
    # loss_kl of shape bs,
    # loss_reconstruct of shape bs,
    # print(loss_reconstruct_numerical.shape, loss_kl.shape)
    return loss_reconstruct_numerical + loss_kl

# %% ../nbs/01_vae.ipynb 62
def get_similarity(inputs, targets):
    func = F.mse_loss
    with torch.no_grad():
        loss_num = func(inputs.flatten(), targets.flatten())
        return loss_num


# %% ../nbs/01_vae.ipynb 66
device = "cuda" if torch.cuda.is_available() else "cpu"
device

# %% ../nbs/01_vae.ipynb 67
def validate_epoch(dls, criterion, scorer):
    """For the full dataloader, calculate the running loss and score"""
    model.eval()
    running_loss = 0.0
    running_score = 0.0
    with torch.no_grad():
        for batch_idx, xs in enumerate(dls):
            # move to device
            xs = xs.to(device)

            # Forward pass
            xs_gen, loss_kl = model(xs)

            loss = criterion(xs_gen, targets=xs, loss_kl=loss_kl)
            # print(loss - loss_kl, loss_kl)
            # calc score
            score = scorer(xs_gen, xs)

            running_loss += loss.item()
            running_score += score.item()
    return running_loss / len(dls), running_score / len(dls)


# %% ../nbs/01_vae.ipynb 68
from fastcore.xtras import partial
