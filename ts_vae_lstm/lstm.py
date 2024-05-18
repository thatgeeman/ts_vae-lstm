# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/02_lstm.ipynb.

# %% auto 0
__all__ = ['TSLSTMDataset', 'LSTMModel', 'loss_func_lstm', 'huber_loss', 'MAELoss', 'mean_squared_error', 'error_weighted',
           'mean_absolute_error', 'calculate_smape', 'scorer_lstm']

# %% ../nbs/02_lstm.ipynb 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
from .concepts import get_window
from scipy import signal
import os
import math
import torch

# %% ../nbs/02_lstm.ipynb 7
from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

# %% ../nbs/02_lstm.ipynb 8
from .vae import VAE, Encoder, Decoder, StochasticSampler
from fastcore.xtras import noop

# %% ../nbs/02_lstm.ipynb 23
class TSLSTMDataset(Dataset):
    def __init__(
        self,
        embeddings,  # full dataset (not separated as we need to index the next window)
        indices,
        window_size=48,
        latent_dim=32,
        n_features=1,
        n_prev_windows=1,
        mean=0,
        std=1,
    ):
        self.embeddings = embeddings
        self.indices = indices
        self.mean = mean
        self.std = std
        self.n_prev_windows = n_prev_windows
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.n_features = n_features

    def __getitem__(self, idx):
        true_idx = self.indices[idx]
        true_idx_next = true_idx + 1
        x_emb = (self.embeddings[true_idx]["subset"] - self.mean) / self.std
        y_emb = (self.embeddings[true_idx_next]["subset"] - self.mean) / self.std
        return x_emb, y_emb  # latent_dim, seq_len

    def __len__(self):
        return len(self.indices) - 1

# %% ../nbs/02_lstm.ipynb 35
class LSTMModel(nn.Module):
    def __init__(
        self, input_size=32, hidden_size=128, output_size=32, activation=F.tanh
    ):
        super(LSTMModel, self).__init__()
        self.isz = input_size
        self.hsz = hidden_size
        self.osz = output_size
        self.lstm_input = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,  # output_size,
            num_layers=1,
        )
        self.lstm_hidden = nn.LSTM(
            input_size=hidden_size,  # output_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_output = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=output_size,
            num_layers=1,
            batch_first=True,
        )
        self.activation = activation
        self.dropout = nn.Dropout(0.5)
        # F.tanh  # F.relu, F.tanh, F.sigmoid, F.leaky_relu, F.elu, F.selu, F.softplus, F.softmax
        self.init_lstm_weights()

    def forward(self, x):
        # x has shape bs, emb_dim, seq_len (emb_dim=latent_dim, seq_len=1)
        x = x.permute(0, 2, 1)  # bs, seq_len, emb_dim
        # LSTM1
        lstm_out1, (h1, c1) = self.lstm_input(x)
        lstm_out1 = self.activation(self.dropout(lstm_out1))
        # LSTM2
        lstm_out2, _ = self.lstm_hidden(lstm_out1)
        lstm_out2 = self.activation(self.dropout(lstm_out2))
        # LSTM3
        lstm_out3, _ = self.lstm_output(lstm_out2)

        lstm_out3 = self.activation(self.dropout(lstm_out3))
        return lstm_out3.permute(0, 2, 1)  # bs, output_size, seq_len

    def init_lstm_weights(self):
        for layer in [self.lstm_input, self.lstm_hidden, self.lstm_output]:
            for name, param in layer.named_parameters():
                if "weight" in name:
                    nn.init.xavier_normal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)

# %% ../nbs/02_lstm.ipynb 41
def loss_func_lstm(inputs, targets):
    bs = inputs.shape[0]
    distance = torch.norm(targets - inputs, dim=2)
    loss = torch.where(distance > 1, distance.pow(2), torch.clamp(1 - distance, min=0))
    return loss.sum((-1)).mean()
    # return loss


def huber_loss(inputs, targets):
    bs = inputs.shape[0]
    loss = F.huber_loss(inputs, targets, reduction="none", delta=1)
    # delta=0.05 cos the error in embedding space is really small
    return loss.sum((-1, -2)).mean()  # sum((-1, -2)).mean()
    # return loss


def MAELoss(predictions, targets):
    error = torch.abs(predictions - targets)
    mae = torch.sum(error, dim=(-1, -2))
    return torch.mean(mae)


def mean_squared_error(predictions, targets):
    error = (predictions - targets) ** 2
    mse = torch.sum(error, dim=(-1, -2))
    return torch.mean(mse)


def error_weighted(predictions, targets):
    weights = torch.abs(targets) + 1
    error = torch.abs(predictions - targets) * weights
    mse = torch.sum(error, dim=(-1, -2))
    return torch.mean(mse)


@torch.no_grad()
def mean_absolute_error(predictions, targets):
    error = torch.abs(predictions - targets)
    mae = torch.sum(error, dim=(-1, -2))
    return torch.mean(mae)


def calculate_smape(predicted, actual):
    absolute_percentage_errors = torch.abs(predicted - actual) / (
        torch.abs(predicted) + torch.abs(actual)
    )
    return absolute_percentage_errors.sum((-1, -2)).mean()


def scorer_lstm(inputs, targets):
    with torch.no_grad():
        return torch.pow(inputs.squeeze() - targets.squeeze(), 2).mean()

# %% ../nbs/02_lstm.ipynb 45
from fastcore.xtras import partial
