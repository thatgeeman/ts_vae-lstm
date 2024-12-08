import logging
import time

import hydra
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import torch.functional as F  # noqa
from fastcore.xtras import dict2obj

from scripts.train_lstm import load_lstm_model, load_vae_model, parse_path
from scripts.train_vae import load_data, set_device
from ts_vae_lstm.ad_complete import AD

EPSILON = 1e-100


def plot_anomalies(cfg, pred, true_anomalies=None):
    pred_anomalies = pred["status_granular"]
    if true_anomalies is not None:
        true_anomalies = [v - cfg.dataset.idx_split for v in cfg.dataset.idx_anomaly]
    else:
        true_anomalies = [0]
    logging.info(f"True anomalies shifted by the idx_split: {cfg.dataset.idx_split}")
    quantile_thresh = cfg.infer.quantile_thresh
    logging.info(
        f"Value ranges: \nActual: {pred['actual'].reshape(-1)[:5]}\nReconstructed: {pred['reconstructed'].reshape(-1)[:5]}"
    )
    min_x = min(min(pred["actual"].reshape(-1)), min(pred["reconstructed"].reshape(-1)))
    max_x = max(max(pred["actual"].reshape(-1)), max(pred["reconstructed"].reshape(-1)))

    logging.info(f"Predicted steps: {pred['steps'].shape}")
    logging.info(f"Actual shape: {pred['actual'].shape}")
    logging.info(f"Reconstructed shape: {pred['reconstructed'].shape}")
    logging.info("Shapes will be flattened for plotting")
    fig, ax = plt.subplots(figsize=(14, 3))

    pred_steps_ = pred["steps"].reshape(-1)

    ax = sns.lineplot(
        x=pred_steps_,
        y=pred["actual"].reshape(-1),
        alpha=0.6,
        linestyle="-",
        label="data",
        ax=ax,
    )
    ax = sns.lineplot(
        x=pred_steps_,
        y=pred["reconstructed"].reshape(-1),
        label="reconstructed",
        alpha=0.3,
        ax=ax,
    )

    # sns.lineplot(pred["status"].reshape(-1), label="pred anomaly")
    for idx, val in enumerate(pred["status"].mean(1)):
        if val == 1:
            ax.axvspan(
                pred["steps"][idx][0],
                pred["steps"][idx][-1],
                facecolor="green",
                alpha=0.1,
            )
    for idx, val in enumerate(true_anomalies):
        ax.vlines(
            val,
            min_x,
            max_x,
            linestyles="--",
            linewidth=1.5,
            colors="red",
            label="actual" if idx == 0 else None,
        )
    for idx, val in enumerate(pred_anomalies):
        ax.vlines(
            val,
            min_x,
            max_x,
            linestyles="-",
            linewidth=1.2,
            colors="green",
            label="predicted" if idx == 0 else None,
            alpha=0.8,
        )
    plt.xlim(-30, pred_steps_[-1] + 30)
    # plt.ylim(-3, 3)
    plt.legend(bbox_to_anchor=(1.0, 1.05))
    plt.xticks(ticks=[x for x in pred_steps_ if x % 500 == 0], rotation=0)
    plt.ylabel("Taxi rides (normalized)")
    plt.xlabel("time (30 minute step)")
    plt.title(
        f"Number of previous steps: {len(pred_steps_)}; L2-threshold at Q{quantile_thresh}: {pred['threshold']:.2f}"
    )
    plot_path = (
        f"{cfg.infer.plot_path}/ad_result_z{cfg.vae.latent_dim}_lstm_{time.time()}.png"
    )
    logging.info(f"Plotting figure to {plot_path}")
    plt.tight_layout()

    fig.savefig(f"{plot_path}")


@hydra.main("../config", "config.yaml", version_base="1.2")
def main(cfg):
    global device
    device = set_device(cfg)
    data = load_data(cfg)
    _, data_test = (
        data[cfg.dataset.signal][: cfg.dataset.idx_split],
        data[cfg.dataset.signal][cfg.dataset.idx_split :],
    )

    vae = load_vae_model(cfg, device=cfg.infer.device)
    lstm = load_lstm_model(cfg, device=cfg.infer.device)

    # load stats for vae
    best_cfg_path = parse_path(cfg, attribute="vae_params_path")
    logging.info(f"Got VAE params path {best_cfg_path}")
    best_cfg = dict2obj(joblib.load(best_cfg_path))
    training_stats = {"vae": (best_cfg.means, best_cfg.stds)}
    # load stats for lstm
    best_cfg_path = parse_path(cfg, attribute="lstm_params_path")
    logging.info(f"Got LSTM params path {best_cfg_path}")
    best_cfg = dict2obj(joblib.load(best_cfg_path))
    training_stats.update({"lstm": (best_cfg.means, best_cfg.stds)})

    logging.info(f"Got stats for VAE and LSTM: {training_stats}")

    pred = AD(
        x=list(range(len(data_test))),
        y=data_test,
        vae_model=vae,
        lstm_model=lstm,
        window_size=cfg.n_lag,
        stats=training_stats,
        # threshold=threshold,
        quantile_thresh=cfg.infer.quantile_thresh,
        reconstruct_with_true=False,
    )
    logging.info("Prediction complete!")
    logging.info(f"Anomalies at indices: {pred['status_granular']}")
    # visualize
    try:
        true_anomalies = cfg.dataset.idx_anomaly
    except Exception as e:
        logging.info(f"True anomalies cannot be read in dataset config! {e}")
        true_anomalies = None

    plot_anomalies(cfg, pred, true_anomalies)


if __name__ == "__main__":
    main()
