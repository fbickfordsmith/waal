import logging
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.random import Generator

from src.data import dataset_configs, get_data
from src.models import get_models
from src.training import Trainer


def get_config() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--results_dir", type=str, default="results")

    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--n_labels_start", type=int, default=2_000)
    parser.add_argument("--n_labels_end", type=int, default=12_000)
    parser.add_argument("--n_labels_step", type=int, default=2_000)

    parser.add_argument("--use_resnet", type=bool, default=False)

    parser.add_argument("--alpha", type=float, default=2e-3)
    parser.add_argument("--grad_penalty_weight", type=int, default=5)
    parser.add_argument("--n_epochs", type=int, default=80)
    parser.add_argument("--selection_coef", type=int, default=10)

    return parser.parse_args()


def get_rng(seed: int = -1) -> Generator:
    """
    References:
        https://pytorch.org/docs/stable/notes/randomness.html
    """
    if seed == -1:
        seed = random.randint(0, int(1e6))

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    return np.random.default_rng(seed)


def get_device(use_gpu: bool = True, use_deterministic_ops: bool = False) -> str:
    """
    References:
        https://pytorch.org/docs/stable/notes/mps.html
        https://pytorch.org/docs/stable/notes/randomness.html
    """
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
    elif use_gpu and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if use_deterministic_ops:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False

    return device


def set_up_logging() -> None:
    """
    References:
        https://stackoverflow.com/a/44175370
    """
    logging.basicConfig(
        format="[%(asctime)s][%(levelname)s] - %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(cfg: Namespace) -> None:
    logging.info("Setting up")
    logging.info(f"Seed: {cfg.seed}")

    rng = get_rng(cfg.seed)
    device = get_device(cfg.use_gpu)

    if cfg.use_gpu and (device not in {"cuda", "mps"}):
        logging.warning(f"Device: {device}")
    else:
        logging.info(f"Device: {device}")

    experiment_name = f"{cfg.dataset}_{cfg.n_labels_start}_to_{cfg.n_labels_end}_labels"

    if cfg.use_resnet:
        experiment_name += "_resnet"

    results_dir = Path(cfg.results_dir) / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    train_inputs, train_labels, test_inputs, test_labels = get_data(cfg.data_dir, cfg.dataset)

    is_labelled = np.zeros(len(train_inputs), dtype=bool)
    is_labelled[: cfg.n_labels_start] = True
    is_labelled = rng.permutation(is_labelled)

    feat_model, task_model, pool_model = get_models(cfg.dataset, cfg.use_resnet)

    trainer = Trainer(
        train_inputs,
        train_labels,
        is_labelled,
        feat_model,
        task_model,
        pool_model,
        cfg.selection_coef,
        device,
        dataset_configs[cfg.dataset],
    )

    logging.info("Starting active learning")

    test_log = {"n_labels": [], "test_acc": []}

    while True:
        n_labels = np.sum(trainer.is_labelled)

        logging.info(f"Number of labels: {n_labels}")

        trainer.train(cfg.alpha, cfg.grad_penalty_weight, cfg.n_epochs)

        predictions = trainer.predict_labels_argmax(test_inputs, test_labels)
        test_acc = torch.sum(test_labels == predictions).item() / len(test_labels)

        logging.info(f"Testing: acc={test_acc:.3f}")

        test_log["n_labels"].append(n_labels)
        test_log["test_acc"].append(test_acc)

        pd.DataFrame(test_log).to_csv(results_dir / f"seed{cfg.seed}.csv", index=False)

        if n_labels >= cfg.n_labels_end:
            logging.info("Stopping active learning")
            break

        selected_inds = trainer.select_queries(cfg.n_labels_step)
        trainer.is_labelled[selected_inds] = True


if __name__ == "__main__":
    set_up_logging()
    cfg = get_config()
    main(cfg)
