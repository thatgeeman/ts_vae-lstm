import copy
import logging
import time

import hydra
import joblib
import numpy as np
import pandas as pd
import torch
from fastcore.xtras import Path
from omegaconf.listconfig import ListConfig
from torch.utils.data import DataLoader

from ts_vae_lstm.concepts import compare, get_window
from ts_vae_lstm.vae import VAE, TSDataset, evaluate_reconstruction, validate_epoch

EPSILON = 1e-100
train_dataloader = None
valid_dataloader = None
test_dataloader = None
device = None
params = dict()


def set_device(cfg):
    global device
    device = cfg.device if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    return device


def load_data(cfg):
    DATAPATH = Path(cfg.dataset.path).resolve()
    logging.info(f"Dataset is {DATAPATH}")
    if DATAPATH.name.endswith(".npz"):
        data = np.load(DATAPATH)
        logging.info(data.keys())
    elif DATAPATH.name.endswith(".csv"):
        data = pd.read_csv(DATAPATH)
        logging.info(data.columns)
    elif DATAPATH.name.endswith(".parquet"):
        data = pd.read_parquet(DATAPATH)
        logging.info(data.columns)
    else:
        raise NotImplementedError(
            f"Cannot read {DATAPATH}. Please raise an issue on gh."
        )
    return data


def check_split_config(cfg):
    try:
        assert isinstance(cfg.dataset.idx_anomaly, ListConfig)
    except AssertionError:
        logging.info("Anomaly idxs are not provided as a list")
    except AttributeError:
        logging.info("No idx_anomaly set in config")
    idx_anomaly = sorted(cfg.dataset.idx_anomaly)
    assert (
        idx_anomaly[0] > cfg.dataset.idx_split
    ), f"First known anomaly should be after idx_split={cfg.dataset.idx_split}"

    logging.info("Split is OK.")


def get_windows(data, window_size):
    end_steps = [es for es in range(window_size, len(data), 1)]

    data_windowed = [
        {
            "subset": get_window(
                data,
                window_size=window_size,
                end_step=t,
                return_indices=False,
            ),
            "end_step": t,  # the time we want to predict for. For vae, we reconstruct the window.
            "start_step": t - window_size,
        }
        for t in end_steps
    ]
    logging.info(f"Dataset transformed to {len(data_windowed)} windows.")
    return data_windowed


def train(cfg, print_every=1):
    model = VAE(
        input_shape=(cfg.n_lag, cfg.n_signals),
        latent_dim=cfg.vae.latent_dim,
        num_hidden_units=cfg.vae.num_hidden_units,
        kernel_size=(cfg.vae.kernel_size, cfg.n_signals),
        stride=(cfg.vae.stride, cfg.n_signals),
    ).to(device)

    logging.info(f"Model:\n{str(model)}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.vae.lr, weight_decay=cfg.vae.wd
    )
    scorer = evaluate_reconstruction  # calculate_smape
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=cfg.vae.factor,
        patience=cfg.vae.patience,
        min_lr=1e-8,
        verbose=True,
    )
    # Training loop
    logging.info(f"Metric larger-is-better: {cfg.vae.metric_maximize}")
    best_model = None
    best_score = -np.inf if cfg.vae.metric_maximize else np.inf
    best_epoch = 0
    for epoch in range(cfg.vae.epochs):
        model.train()
        running_loss = 0.0
        running_score = 0.0
        n_dls = 0  # due to droplast, make sure to divide by correct batch_size
        for batch_idx, xs in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()

            # move to device
            xs = xs.to(device)

            # Forward pass
            xs_gen, loss = model(xs)
            # calc score
            score = scorer(xs_gen, xs)["mse"]

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
                model, valid_dataloader, scorer, device=device
            )
            logging.info(
                "Epoch [{}/{}], Batch [{}/{}], Loss: [{:.3f}, {:.3f}], Score: [{:.3f}, {:.3f}]".format(
                    epoch + 1,
                    cfg.vae.epochs,
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

        if compare(running_score, best_score, cfg.vae.metric_maximize):
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
        f"{cfg.model_dir}/vae_{cfg.vae.epochs}_z{cfg.vae.latent_dim}_{time_slug}.pth"
    )
    best_model_pth = (
        f"{cfg.model_dir}/best_vae_{best_epoch}_z{cfg.vae.latent_dim}_{time_slug}.pth"
    )

    logging.info(f"Last model saved to {model_pth}")
    torch.save(model.state_dict(), model_pth)
    logging.info(f"Best model saved to {best_model_pth}")
    torch.save(best_model, best_model_pth)
    # params to file
    params.update(dict(cfg))
    params_pth = f"{cfg.model_dir}/vae_z{cfg.vae.latent_dim}_{time_slug}.params"
    joblib.dump(params, params_pth)
    logging.info(f"Params saved to {params_pth}")


@hydra.main("../config", "config.yaml", version_base="1.2")
def main(cfg):
    set_device(cfg)
    check_split_config(cfg)
    data = load_data(cfg)

    data_train, _ = (
        data[cfg.dataset.signal][: cfg.dataset.idx_split],
        data[cfg.dataset.signal][cfg.dataset.idx_split :],
    )
    # train_m, train_std = data_train.mean(), data_train.std()
    data_windowed = get_windows(data_train, window_size=cfg.n_lag)

    val_data_idxs = np.random.choice(
        range(len(data_windowed)),
        size=int(cfg.test_split * len(data_windowed)),
        replace=False,
    )
    trn_data_idxs = [
        idx for idx in range(len(data_windowed)) if idx not in val_data_idxs
    ]

    val_data = [data_windowed[idx] for idx in val_data_idxs]
    trn_data = [data_windowed[idx] for idx in trn_data_idxs]

    assert (
        cfg.n_signals == trn_data[0]["subset"].shape[1]
    ), f"Got {cfg.n_signals} in config but {trn_data[0]['subset'].shape[1]} features in data!"

    means = np.zeros((len(trn_data), cfg.n_signals))  # ((len(trn_data), 4))
    stds = np.zeros((len(trn_data), cfg.n_signals))  # ((len(trn_data), 4))
    slice_from = 1 if cfg.dataset.time_index else 0
    logging.info(
        f"Time column in dataset: {cfg.dataset.time_index}. Slicing columns from index {slice_from}"
    )
    for i, _trn_data in enumerate(trn_data):
        means[i] = (np.mean(_trn_data["subset"][:, slice_from:], axis=0)).astype(
            np.float32
        )
        stds[i] = (np.var(_trn_data["subset"][:, slice_from:], axis=0) ** 0.5).astype(
            np.float32
        )
    means = means.mean(0)
    stds = stds.mean(0)

    params.update({"means": means, "stds": stds})

    dset_trn = TSDataset(trn_data, mean=means, std=stds, slice_from=slice_from)
    dset_val = TSDataset(val_data, mean=means, std=stds, slice_from=slice_from)

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
        drop_last=False,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    print_every = len(train_dataloader)
    logging.info("Data preparation complete. Training.")

    train(cfg, print_every)


if __name__ == "__main__":
    main()
