from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import StrMethodFormatter


def set_font_size(font_size: int) -> None:
    # https://stackoverflow.com/a/39566040
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "figure.titlesize": font_size,
        }
    )


def plot(axes: Axes, results: pd.DataFrame, label: str = None) -> None:
    axes.plot(results["n_labels"], results["test_acc_mean"], label=label)
    axes.fill_between(
        results["n_labels"],
        results["test_acc_mean"] + results["test_acc_sem"],
        results["test_acc_mean"] - results["test_acc_sem"],
        alpha=0.3,
    )
    axes.grid(visible=True, axis="y")


def main() -> None:
    results_dir = Path("results")

    results = {
        "CIFAR10_20_to_300_labels_resnet": None,
        "CIFAR10_2000_to_12000_labels": None,
        "CIFAR10_2000_to_12000_labels_resnet": None,
    }

    for experiment_name in results:
        filepaths = sorted(results_dir.glob(f"{experiment_name}/*.csv"))

        for i, filepath in enumerate(filepaths):
            print(filepath)

            run_results = pd.read_csv(filepath)
            run_results = run_results.rename(columns={"test_acc": f"test_acc_{filepath.stem}"})

            if i == 0:
                experiment_results = run_results
            else:
                experiment_results = experiment_results.merge(run_results, on="n_labels")

        experiment_results["test_acc_mean"] = experiment_results.filter(regex="acc").mean(axis=1)
        experiment_results["test_acc_sem"] = experiment_results.filter(regex="acc").sem(axis=1)
        experiment_results[experiment_results.filter(regex="acc").columns] *= 100

        results[experiment_name] = experiment_results

    set_font_size(11)

    figure, axes = plt.subplots(ncols=2, sharey=True, figsize=(8, 3))

    plot(axes[0], results["CIFAR10_20_to_300_labels_resnet"])
    plot(axes[1], results["CIFAR10_2000_to_12000_labels_resnet"], label="ResNet18")
    plot(axes[1], results["CIFAR10_2000_to_12000_labels"], label="VGG16")

    axes[0].set(title="Low label budget", xlabel="Number of labels", ylabel="Test accuracy (%)")
    axes[1].set(title="High label budget", xlabel="Number of labels", yticks=(20, 40, 60, 80))
    axes[1].xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes[1].legend(loc="lower right", borderpad=0.5)

    figure.tight_layout(w_pad=2)
    figure.savefig(results_dir / "plot.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
