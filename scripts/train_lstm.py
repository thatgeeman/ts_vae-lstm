import copy
import glob
import logging
import os
import time

import hydra
import joblib
import numpy as np
import torch
import torch.functional as F  # noqa
from torch.utils.data import DataLoader

from scripts.train_vae import get_windows, load_data, set_device
from ts_vae_lstm.concepts import compare
from ts_vae_lstm.lstm import (
    LSTMModel,
    TSLSTMDataset,
    embedding_loss,
    embedding_metric,
    get_activation,
    get_embeddings,
    validate_epoch,
)
from ts_vae_lstm.vae import VAE

EPSILON = 1e-100
train_dataloader = None
valid_dataloader = None
test_dataloader = None
device = None
vae_device = "cpu"  # keep embeddings on cpu
params = dict()


def parse_path(cfg, attribute="vae_path"):
    if cfg.pattern:
        paths = glob.glob(cfg[attribute])
        logging.info(f"Will select latest file from {paths}")
        assert len(paths) > 0, f"No paths matching pattern {cfg[attribute]} found!"
        latest_path = paths[0]
        latest_time = 0
        for path in paths:
            if os.path.getmtime(path) > latest_time:
                latest_path = path
                latest_time = os.path.getmtime(path)
        return latest_path
    else:
        return cfg[attribute]


def get_embedding_windows(data, vae, cfg):
    data_windowed = get_windows(data, cfg.n_lag)
    logging.info(f"Got the windowed dataset: {len(data_windowed)}")
    # to apply standardization to the raw data first
    logging.info("Loading parameters of VAE model")
    vae_params_path = parse_path(cfg, attribute="vae_params_path")
    vae_params = joblib.load(vae_params_path)
    means = vae_params["means"]
    stds = vae_params["stds"]
    slice_from = 1 if cfg.dataset.time_index else 0
    # now run standarzation and pass to get-embeddings
    embeddings = [
        {
            "subset": get_embeddings(
                x=(data_windowed[i]["subset"][:, slice_from:] - means)
                / (stds + EPSILON),
                vae_model=vae,
                latent_dim=cfg.vae.latent_dim,
                n_windows=cfg.n_windows,
                seq_len=cfg.n_signals,
                sampler_repeat=cfg.sampler_repeat,  # large number would mean we get the average embedding of the latent space
                device=vae_device,  # embeddings on cpu
            ),
            "end_step": data_windowed[i]["end_step"],
            "start_step": data_windowed[i]["start_step"],
        }
        for i in range(len(data_windowed))
    ]
    logging.info(
        f"From {len(data_windowed)} windows, {len(embeddings)} Embeddings generated using VAE."
    )
    return embeddings


def freeze(model):
    for param in model.parameters():
        try:
            param.requires_grad = False
        except Exception:
            logging.error(f"Could not set requires_grad=False in param: {param}")
    logging.info("VAE model parameters frozen.")


def load_model_path(cfg):
    path = parse_path(cfg)
    logging.info(f"Got VAE model path {path}")
    model = VAE(
        input_shape=(cfg.n_lag, cfg.n_signals),
        latent_dim=cfg.vae.latent_dim,
        num_hidden_units=cfg.vae.num_hidden_units,
        kernel_size=(cfg.vae.kernel_size, cfg.n_signals),
        stride=(cfg.vae.stride, cfg.n_signals),
    ).to(vae_device)
    checkpoint = torch.load(path, map_location=vae_device)
    model.load_state_dict(checkpoint)
    logging.info(f"VAE model loaded to {vae_device}.")
    freeze(model)
    return model


def train(cfg, print_every=1):
    model = LSTMModel(
        input_size=cfg.vae.latent_dim - 1,  # x[:-1]
        output_size=cfg.vae.latent_dim - 1,  # x[1:]
        hidden_size=cfg.lstm.hidden_size,
        activation=get_activation(cfg.lstm.activation),
        num_layers=cfg.lstm.num_layers,
        dropout_p=cfg.lstm.dropout_p,
    ).to(device)
    logging.info(f"Model:\n{str(model)}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lstm.lr,
        weight_decay=cfg.lstm.wd,
        maximize=cfg.lstm.optimizer_maximize,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=cfg.lstm.factor,
        patience=cfg.lstm.patience,
        min_lr=1e-8,
        verbose=True,
    )
    # Training loop
    logging.info(
        f"Metric: {cfg.lstm.metric} and larger-is-better: {cfg.lstm.metric_maximize}"
    )
    best_model = None
    best_score = -np.inf if cfg.lstm.metric_maximize else np.inf
    best_epoch = 0
    for epoch in range(cfg.lstm.epochs):
        model.train()
        running_loss = 0.0
        running_score = 0.0
        n_dls = 0  # due to droplast, make sure to divide by correct batch_size
        for batch_idx, (xs, ys) in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()

            # move to device
            xs = xs.to(device)
            ys = ys.to(device)

            # Forward pass
            pred_ys = model(xs)

            loss = embedding_loss(pred_ys, ys)
            # calc score
            score = embedding_metric(pred_ys, ys)[cfg.lstm.metric]

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_score += score
            n_dls += 1
        # epoch ended
        if (batch_idx + 1) % print_every == 0:
            # calculate loss for valid_dataloader
            # n_dls = len(train_dataloader)
            val_loss, val_score = validate_epoch(
                model,
                valid_dataloader,
                embedding_loss,
                embedding_metric,
                metric_name=cfg.lstm.metric,
                device=device,
            )
            print(
                "Epoch [{}/{}], Batch [{}/{}], Loss: [{:.3f}, {:.3f}], Score: [{:.3f}, {:.3f}]".format(
                    epoch + 1,
                    cfg.lstm.epochs,
                    batch_idx + 1,
                    n_dls,
                    running_loss / n_dls,
                    val_loss,
                    running_score / n_dls,
                    val_score,
                )
            )
        # Step the LR scheduler
        scheduler.step(val_loss)  # min the running_loss
        # keep best model
        if compare(running_score, best_score, cfg.lstm.metric_maximize):
            best_model = copy.deepcopy(model.state_dict())
            best_score = running_score
            best_epoch = epoch + 1
            logging.info(
                f"New best model found at epoch {best_epoch} with score {best_score}"
            )
        # reset at end of epoch
        running_loss = 0.0
        running_score = 0.0
    time_slug = int(time.time())
    model_pth = (
        f"{cfg.model_dir}/lstm_{cfg.lstm.epochs}_z{cfg.vae.latent_dim}_{time_slug}.pth"
    )
    best_model_pth = (
        f"{cfg.model_dir}/best_lstm_{best_epoch}_z{cfg.vae.latent_dim}_{time_slug}.pth"
    )

    logging.info(f"Last model saved to {model_pth}")
    torch.save(model.state_dict(), model_pth)
    logging.info(f"Best model saved to {best_model_pth}")
    torch.save(best_model, best_model_pth)
    # params to file
    params.update(dict(cfg))
    params_pth = f"{cfg.model_dir}/lstm_z{cfg.vae.latent_dim}_{time_slug}.params"
    joblib.dump(params, params_pth)
    logging.info(f"Params saved to {params_pth}")


@hydra.main("../config", "config.yaml", version_base="1.2")
def main(cfg):
    global device
    device = set_device(cfg)
    data = load_data(cfg)
    data_train, _ = (
        data[cfg.dataset.signal][: cfg.dataset.idx_split],
        data[cfg.dataset.signal][cfg.dataset.idx_split :],
    )

    vae = load_model_path(cfg)
    emb = get_embedding_windows(data_train, vae, cfg)  # store the embeddings in advance

    # split the embeddings in the order.
    # we provide as input x[:-1] and make it predict x[1:] (one step forward)
    range_choices = range(len(emb) - 1)
    np.random.seed(cfg.random_seed)
    val_data_idxs = np.random.choice(
        range_choices, size=int(cfg.test_split * len(emb)), replace=False
    )
    trn_data_idxs = [idx for idx in range_choices if idx not in val_data_idxs]
    np.random.shuffle(trn_data_idxs)
    logging.info(f"Train embedding indices: {len(trn_data_idxs)}")
    # calculate mean and std of embeddings, should be very close to 0, 1 as sampler of VAE is Normal
    window_means = np.asarray([emb[i]["subset"].mean().item() for i in trn_data_idxs])
    window_stds = np.asarray([emb[i]["subset"].std().item() for i in trn_data_idxs])
    emb_mean, emb_std = window_means.mean(), window_stds.mean()
    logging.info(f"Embedding mean and std of train: {emb_mean} ({emb_std})")

    dset_trn = TSLSTMDataset(
        emb,
        indices=trn_data_idxs,
        mean=emb_mean,
        std=emb_std,
    )

    dset_val = TSLSTMDataset(
        emb,
        indices=val_data_idxs,
        mean=emb_mean,
        std=emb_std,
    )  # use same stats from training data
    global train_dataloader
    train_dataloader = DataLoader(
        dataset=dset_trn,
        batch_size=cfg.batch_sz,
        drop_last=cfg.drop_last,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    global valid_dataloader
    valid_dataloader = DataLoader(
        dataset=dset_val,
        batch_size=cfg.batch_sz,
        drop_last=cfg.drop_last,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    print_every = len(train_dataloader)
    logging.info("Data preparation complete. Training.")
    train(cfg, print_every=print_every)


if __name__ == "__main__":
    main()
