import matplotlib.pyplot as plt
import numpy as np
from pyutils.general import ensure_dir
from pyutils.plot import pdf_crop, set_ms

__all__ = [
    "prepare_plot_data",
    "plot_arch_breakdown",
]


set_ms()
color_dict = {
    "black": "#000000",
    "red": "#de425b",  # red
    "blue": "#1F77B4",  # blue
    "orange": "#f58055",  # orange
    "yellow": "#f6df7f",  # yellow
    "green": "#2a9a2a",  # green
    "grey": "#979797",  # grey
    "purple": "#AF69C5",  # purple,
    "mitred": "#A31F34",  # mit red
    "pink": "#CDA2BE",
}


def prepare_plot_data(
    data: dict,
    keys: list,
) -> tuple:
    labels = list(data.keys())
    values = {key: [] for key in keys}

    for component, metrics in data.items():
        for key in keys:
            values[key].append(metrics[key])

    return labels, values


def plot_arch_breakdown(
    arch_breakdown: dict,
    plot_attr: str,
    save_path: str,
) -> None:
    labels, values = prepare_plot_data(arch_breakdown, plot_attr)

    fig, ax = plt.subplots()
    bottom = np.zeros(len(labels))

    for key, color in zip(values.keys(), plt.cm.Paired.colors):
        ax.bar(labels, values[key], label=key, bottom=bottom, color=color)
        bottom += values[key]

    # ax.set_ylabel(ylabel)
    # ax.set_title(title)
    ax.legend()

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    set_ms()
    ensure_dir(f"./{save_path}")
    fig.savefig(f"./{save_path}/{plot_attr}.png", format="png")
    fig.savefig(f"./{save_path}/{plot_attr}.pdf", format="pdf")
    pdf_crop(f"./{save_path}/{plot_attr}.pdf", f"./{save_path}/{plot_attr}.pdf")


