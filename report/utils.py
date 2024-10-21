import matplotlib.pyplot as plt
import numpy as np
import os
import re
import json
import types
import inspect
import importlib


def build_metrics_dict(metrics):
    metrics_dict = {}

    for key, value in metrics.items():
        if isinstance(value, dict):
            metrics_dict[key] = {
                graph: {line: [] for line in value[graph]} for graph in value.keys()
            }

        elif isinstance(value, list):
            metrics_dict[key] = {key: {line: [] for line in value}}

        else:
            raise ValueError("Invalid structure for metrics dict.")

    return metrics_dict


def split_metric_path(path, n_parts=3):
    parts = re.split(r"[\\/]", path)

    if (len(parts) == 0) or (len(parts) > n_parts):
        raise ValueError("Invalid path for metric.")

    # Repeats the first part of the path until it matches the pattern (key/graph/line)
    while len(parts) < n_parts:
        parts.insert(0, parts[0])

    return parts


def plot_report_metric(
    metric_key,
    metric_value,
    save_path,
    max_cols=3,
    fig_size=(10, 6),
    x_axis="Epochs",
):
    # Use a light style for better readability
    plt.style.use("seaborn-v0_8-whitegrid")

    # Using a colormap for dynamic color selection
    colormap = plt.cm.get_cmap("tab10")

    n_graphs = len(metric_value)
    n_cols = min(n_graphs, max_cols)
    n_rows = (n_graphs + n_cols - 1) // n_cols  # Use ceiling division to get rows

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_size[0] * n_cols, fig_size[1] * n_rows),
    )

    # Ensure axs is always a 1D array for consistent indexing
    if n_rows == 1 and n_cols == 1:
        axs = np.array([axs])  # If only one subplot, make it iterable

    axs = axs.flatten()  # Ensure axs is a flat array for indexing

    for i, (graph, lines) in enumerate(metric_value.items()):
        ax = axs[i]  # Directly access the correct axis

        line_count = len(lines.keys())
        labels_added = False  # Track if any labels were added

        for j, (line, values) in enumerate(lines.items()):
            epochs = range(1, len(values) + 1)
            label = (
                line if line_count > 1 else None
            )  # Set label only if more than one line
            ax.plot(
                epochs,
                values,
                label=label,
                linewidth=2,
                color=colormap(j % colormap.N),
                marker="o",
                markersize=4,
                markerfacecolor="white",
            )
            # If a label is added, update the flag
            if label is not None:
                labels_added = True

        # Add title and axis labels with better styling
        ax.set_title(graph, fontsize=16, weight="bold")
        ax.set_xlabel(x_axis, fontsize=14, labelpad=10)
        ax.set_ylabel(graph, fontsize=14, labelpad=10)

        # Add legend only if labels were added
        if labels_added:
            ax.legend(
                fontsize=12, loc="upper left", bbox_to_anchor=(1, 1), fancybox=True
            )

        # Improve grid visibility
        ax.grid(axis="both", linestyle="--", color="gray", alpha=0.7, linewidth=0.7)

    # Beautify the plot and remove unnecessary spines
    for ax in axs:
        ax.set_facecolor("white")  # Set the background of each subplot to white
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(1)
        ax.spines["left"].set_linewidth(1)
        ax.tick_params(width=1, labelsize=12)

    # Hide any unused subplots if there are fewer graphs than subplots
    for j in range(n_graphs, len(axs)):
        fig.delaxes(axs[j])  # Remove unused axes

    # Adjust layout to ensure everything fits well
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save the plot as a high-resolution image
    save_name = os.path.join(save_path, f"{metric_key}.png")
    plt.savefig(save_name, dpi=300, bbox_inches="tight", transparent=False)
    plt.close()


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle class objects (not instances)
        if isinstance(obj, type):
            return {
                "__type__": "class",
                "name": obj.__name__,
                "module": obj.__module__,
                "file": inspect.getfile(obj),  # Get file where the class is defined
            }

        # Handle function objects
        elif isinstance(obj, types.FunctionType):
            return {
                "__type__": "function",
                "name": obj.__name__,
                "module": obj.__module__,
                "file": inspect.getfile(obj),  # Get file where the function is defined
            }

        # Handle class instances
        elif hasattr(obj, "__class__") and not isinstance(obj, type):
            cls = obj.__class__
            return {
                "__type__": "instance",
                "class_name": cls.__name__,
                "module": cls.__module__,
                "file": inspect.getfile(cls),  # Get file where the class is defined
            }

        # Failsafe: Handle any object by serializing it as a string with its type information
        elif isinstance(obj, object):
            return {
                "__type__": "object",
                "class_name": obj.__class__.__name__,
                "module": obj.__class__.__module__,
                "file": inspect.getfile(
                    obj.__class__
                ),  # Get file where the class is defined
                "repr": repr(obj),  # Fallback representation
            }

        # Default behavior for other types
        return super().default(obj)
