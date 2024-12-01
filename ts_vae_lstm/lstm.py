"""Fill in a module description here"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/02_lstm.ipynb.

# %% auto 0
__all__ = ['get_activation', 'get_embeddings', 'TSLSTMDataset', 'LSTMModel', 'embedding_loss', 'embedding_metric',
           'validate_epoch', 'concat_first_emb']

# %% ../nbs/02_lstm.ipynb 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .concepts import get_window
import torch

torch.backends.mkldnn.enabled = False

# %% ../nbs/02_lstm.ipynb 12
from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader

# %% ../nbs/02_lstm.ipynb 13
from .vae import VAE, Encoder, Decoder, StochasticSampler
from fastcore.xtras import noop

# %% ../nbs/02_lstm.ipynb 14
def get_activation(act):
    activations = {
        "mish": F.mish,
        "silu": F.silu,
        "relu": F.relu,
        "tanh": F.tanh,
        "sigmoid": F.sigmoid,
    }
    try:
        return activations[act]
    except KeyError:
        raise NotImplementedError(f"Unknown activation: {act}")

# %% ../nbs/02_lstm.ipynb 25
@torch.no_grad()
def get_embeddings(
    x,
    vae_model,
    n_windows=1,
    latent_dim=32,
    seq_len=1,
    sampler_repeat=200,
    device="cpu",
):
    """
    _summary_

    Parameters
    ----------
    x : _type_
        _description_
    n_windows : int, optional
        _description_, by default 1
    latent_dim : int, optional
        _description_, by default 32
    seq_len : int, optional
        _description_, by default 1
    sampler_repeat : int, optional
        Number of times to repeatedly sample from the sampler to ensure we have enough variablity in the embedding, by default 10

    Returns
    -------
    _type_
        _description_
    """
    vae_model.eval()
    x = (
        torch.from_numpy(x.astype(np.float32)).view(n_windows, -1, seq_len).to(device)
    )  # p, seq -> n_windows, p, seq
    embedded_x = torch.zeros(n_windows, latent_dim, seq_len).to(device)
    for idx in range(n_windows):
        batched_x_window = x[idx].unsqueeze(0).to(device)
        # print(batched_x_window.shape)
        z_mean, z_log_var = vae_model.encoder.to(device)(batched_x_window)
        # print(z_mean.shape, z_log_var.shape)
        for _ in range(sampler_repeat):
            # explore multiple potential future embeddings by sampling from the latent space multiple times (Monte Carlo sampling).
            embedded_x[idx] += (
                vae_model.latent_sampler(z_mean, z_log_var).permute(1, 0)
                / sampler_repeat
            )
    # reshape
    embedded_x = embedded_x.reshape(latent_dim * n_windows, -1)
    return embedded_x  # is of shape (n_windows* latent_dim, seq_len)

# %% ../nbs/02_lstm.ipynb 47
class TSLSTMDataset(Dataset):
    def __init__(
        self,
        embeddings,  # full dataset (not separated as we need to index the next window)
        indices,
        # window_size=48,
        # latent_dim=32,
        # n_features=1,
        # n_prev_windows=1,
        mean=0,
        std=1,
    ):
        self.embeddings = embeddings
        self.indices = indices  # (idx, idx+1) pairs)
        self.mean = mean
        self.std = std
        # self.n_prev_windows = n_prev_windows
        # self.window_size = window_size
        # self.n_features = n_features
        # self.latent_dim = latent_dim

    def __getitem__(self, idx):
        xidx = self.indices[idx]
        # per-window statistics could be anopter way to do this.
        seq_emb = (self.embeddings[xidx]["subset"] - self.mean) / self.std
        x_emb = seq_emb[:-1]
        y_emb = seq_emb[1:]
        assert len(x_emb) == len(y_emb)
        return x_emb, y_emb  # latent_dim, seq_len

    def __len__(self):
        return len(self.indices)

# %% ../nbs/02_lstm.ipynb 59
class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size=32,
        hidden_size=128,
        output_size=32,
        activation=F.mish,
        bidirectional=True,
        num_layers=(1, 1, 1),
        dropout_p=0.5,
    ):
        super(LSTMModel, self).__init__()
        self.isz = input_size
        self.hsz = hidden_size
        self.osz = output_size

        self.lstm_input = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,  # output_size,
            batch_first=True,
            bidirectional=bidirectional,
            num_layers=num_layers[0],
        )
        self.lstm_hidden = nn.LSTM(
            input_size=hidden_size * 2
            if bidirectional
            else hidden_size,  # output_size,
            hidden_size=hidden_size * 2 if bidirectional else hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
            num_layers=num_layers[1],
        )
        self.lstm_output = nn.LSTM(
            input_size=hidden_size * 4 if bidirectional else hidden_size,
            hidden_size=output_size,
            num_layers=num_layers[2],
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(hidden_size * 2 if bidirectional else hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size * 4 if bidirectional else hidden_size)
        self.ln3 = nn.LayerNorm(output_size)

        self.activation = activation
        self.dropout = nn.Dropout(dropout_p)
        # self._init_lstm_weights()

    def forward(self, x):
        # x has shape bs, emb_dim, seq_len (emb_dim=latent_dim, seq_len=1)
        x = x.permute(0, 2, 1)  # bs, seq_len, emb_dim
        # LSTM1
        lstm_out1, _ = self.lstm_input(x)
        lstm_out1 = self.dropout(self.activation(self.ln1(lstm_out1)))
        # print(lstm_out1.shape)
        # LSTM2
        lstm_out2, _ = self.lstm_hidden(lstm_out1)
        lstm_out2 = self.dropout(self.activation(self.ln2(lstm_out2)))
        # LSTM3
        lstm_out3, _ = self.lstm_output(lstm_out2)
        lstm_out3 = self.ln3(lstm_out3)
        return lstm_out3.permute(0, 2, 1)  # bs, output_size, seq_len

    # def _init_lstm_weights(self):
    #     for layer in [self.lstm_input, self.lstm_hidden, self.lstm_output]:
    #         for name, param in layer.named_parameters():
    #             if "weight" in name:
    #                 nn.init.xavier_normal_(param)
    #             elif "bias" in name:
    #                 nn.init.constant_(param, 0.0)

# %% ../nbs/02_lstm.ipynb 64
def embedding_loss(emb1, emb2, dim=1):
    """
    Calculate the embedding loss as the MAE.

    Parameters
    ----------
    emb1 : _type_
        _description_
    emb2 : _type_
        _description_
    """
    return torch.mean(torch.mean(torch.abs(emb1.cpu() - emb2.cpu()), dim=dim))

# %% ../nbs/02_lstm.ipynb 66
@torch.no_grad()
def embedding_metric(emb1, emb2, reduce=True, dim=1):
    # Jaccard Similarity or Sørensen–Dice coefficient ?
    # avg per embedding dimension the metric, ie: [4, emb, 1] -> [4, 1]
    scores = {
        "mae": torch.mean(torch.abs(emb1.cpu() - emb2.cpu()), dim=dim),
        "cos_sim": F.cosine_similarity(emb1.cpu(), emb2.cpu(), dim=dim),
        "mse": torch.mean((emb1.cpu() - emb2.cpu()) ** 2, dim=dim),
    }

    if reduce:
        for k, v in scores.items():
            scores[k] = torch.mean(v).item()  # mean loss per batch
    return scores

# %% ../nbs/02_lstm.ipynb 71
from fastcore.xtras import partial

# %% ../nbs/02_lstm.ipynb 72
def validate_epoch(
    model, dls, criterion, scorer, metric_name="cos_sim", device="cpu", show=False
):
    model.eval()
    running_loss = 0.0
    running_score = 0.0
    n_dls = 0  # due to droplast, make sure to divide by correct batch_size
    with torch.no_grad():
        for batch_idx, (xs, ys) in enumerate(dls):
            # move to device
            xs = xs.to(device)
            ys = ys.to(device)

            # Forward pass
            pred_ys = model(xs)

            loss = criterion(pred_ys, ys)
            # calc score
            score = scorer(pred_ys, ys)[metric_name]

            running_loss += loss.item()
            running_score += score
            n_dls += 1
        # """
        if show:
            for idx in range(n_dls):
                ax = plt.subplot(4, 2, idx + 1)
                pred_ys_idx, ys_idx = (
                    pred_ys[idx].detach().cpu().squeeze(),
                    ys[idx].detach().cpu().squeeze(),
                )
                sns.lineplot(ys_idx.numpy(), alpha=0.5, linestyle="-.", ax=ax)
                sns.lineplot(pred_ys_idx.numpy(), ax=ax)
                score_idx = scorer(pred_ys, ys, reduce=False)[metric_name][idx].item()
                plt.title(f"{metric_name} = {score_idx:.4f}")
            plt.tight_layout()
            plt.show()
        # """
    return running_loss / n_dls, running_score / n_dls


# %% ../nbs/02_lstm.ipynb 82
def concat_first_emb(y_emb, first_emb, dim=0):
    """Concat first_emb to y_emb"""
    return torch.concat([first_emb, y_emb], dim=dim)
