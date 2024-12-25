import argparse
import os
import pathlib
import re

import matplotlib.pyplot as plt
import pandas as pd


def _get_losses(log_data: str) -> list[dict[str, int | float]]:
    LOSS_PATTERN = r"Epoch \[(\d+)/\d+\], Step \[(\d+)/\d+\], Loss: (\d+\.\d+)"
    # Extract loss data
    lines: list[dict[str, int | float]] = []

    for match in re.finditer(LOSS_PATTERN, log_data):
        lines.append(
            dict(
                epoch=int(match.group(1)),
                step=int(match.group(2)),
                loss=float(match.group(3)),
            )
        )

    return lines


def _get_avg_losses(log_data: str) -> list[dict[str, int | float]]:
    AVG_LOSS_PATTERN = r"Epoch \[(\d+)/\d+\], Average Loss: (\d+\.\d+)"
    lines: list[dict[str, int | float]] = []

    for match in re.finditer(AVG_LOSS_PATTERN, log_data):
        lines.append(dict(epoch=int(match.group(1)), loss=float(match.group(2))))

    return lines


metric_pattern = r"(\w+)=([\d\.]+)(?:, )?"


def _get_metrics(log_data: str) -> list[dict[str, float]]:
    METRIC_PATTERN = r"Metrics: (\w+=[\d\.]+(?:, \w+=[\d\.]+)*)"
    metric_pattern = r"(\w+)=([\d\.]+)(?:, )?"
    metrics: list[dict[str, float]] = []

    for match in re.findall(METRIC_PATTERN, log_data):
        keys, items = zip(*re.findall(metric_pattern, match))
        metrics.append(dict(zip(keys, map(float, items))))

    return metrics


# Plot the pd.DataFrame
def _plot_dataframe(
    df: pd.DataFrame, ax: plt.Axes, xlabel: str, ylabel: str, title: str
) -> None:
    for key in df.columns:
        ax.semilogy(range(len(df)), df[key], label=key)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))


def parse_log_file(prefix: pathlib.Path, fname: pathlib.Path) -> None:
    assert str(fname).endswith(".log")
    log_file_name = str(fname)[:4]
    avg_losses = []
    step_losses = []
    metrics = []
    with open(prefix / fname) as f:
        for log_line in f.readlines():
            avg_losses.extend(_get_avg_losses(log_line))
            step_losses.extend(_get_losses(log_line))
            metrics.extend(_get_metrics(log_line))

    _plot_dataframe(
        pd.DataFrame(step_losses)[["loss"]],
        ax=plt.subplot(),
        xlabel="Steps",
        ylabel="Train Loss",
        title="Training",
    )
    plt.savefig(prefix / f"{log_file_name}-loss.png")
    plt.show()
    plt.cla()
    _plot_dataframe(
        pd.DataFrame(
            metrics
        ),  # pd.DataFrame(map(lambda a, b: {"loss": a["loss"], **b}, avg_losses, metrics)),
        ax=plt.subplot(),
        xlabel="Valid Epoch",
        ylabel="Validation Loss",
        title="Validating",
    )
    plt.savefig(prefix / f"{log_file_name}-metrics.png")
    plt.show()
    plt.cla()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a log file for metrics.")
    parser.add_argument(
        "log_path_prefix",
        type=pathlib.Path,
        help="Parent Path to the log file to be parsed.",
    )
    parser.add_argument("--log-fname", type=pathlib.Path, default="training.log")

    # Parse the arguments
    args = parser.parse_args()

    # Process the log file
    assert os.path.isdir(
        args.log_path_prefix
    ), f"{args.log_file} must be a valid folder"
    assert str(args.log_fname).endswith(".log"), f"{args.log_file} must end with '.log'"
    parse_log_file(args.log_path_prefix, args.log_fname)
