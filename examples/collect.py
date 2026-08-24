import json
import os
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter
from tqdm.contrib.concurrent import process_map

from vtcbench.metric import calc_metrics

sns.set_theme(
    context=None,
    style=None,
    palette="husl",
    font="serif",
    rc={
        "axes.spines.top": False,
        "axes.spines.right": False,
        "mathtext.fontset": "cm",
        "ytick.labelsize": 8,
        "xtick.labelsize": 8,
        "xtick.major.pad": 1,
    },
)

__doc__ = """
Collect results from multiple json files and summarize the results by metadata,
such as VLM model name, data info, render args, shown as a table.
"""

IS_WILD: bool = os.getenv("WILD", "0") == "1"
PLOT_TRADEOFF: bool = os.getenv("PLOT_TRADEOFF", "0") == "1"
PLOT_HEATMAP: bool = os.getenv("PLOT_HEATMAP", "0") == "1"
PLOT_OUTDIR: Path = Path(os.getenv("PLOT_OUTDIR", "fig"))

GROUP_BY_COLS: list[str] = [
    "model_id",
    "render_css",
    "render_css_path",
    "needle_set_path",
    "pure_text",
]
# wild does not need collection_id since it is a dataset-level evaluation
if not IS_WILD:
    GROUP_BY_COLS.append("collection_id")

AGG_MAPPING: dict[str, str] = {
    "contains_all": "mean",
    "ROUGE-L": "mean",
    "json_id": "count",
    "context_length": "mean",
}
METRIC_COLUMNS = [
    # "contains",
    "contains_all",
    "ROUGE-L",
]


def _safe_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [k for k in columns if k in df.columns]


def read_worker(fp: Path) -> list[dict]:
    assert fp.suffix == ".json"

    json_data: dict = json.load(open(fp))
    results: list[dict] = json_data["results"]
    model_id: str = json_data["model_args"]["model"]

    # optional args, may be missing for static dataset
    data_args: dict = json_data.get("data_args", {})
    question_id: str = json_data.get("question_item", {}).get("question_id", "")
    render_css: str = json_data.get("render_args", {}).get("css", "")
    render_css_path: str = json_data.get("render_args", {}).get("css_path", "")
    # redo evaluation to ensure consistency
    return [
        calc_metrics(
            result["response"],
            result["gold_answers"],
            metric=METRIC_COLUMNS,  # ty:ignore[invalid-argument-type]
        )
        | data_args
        | {
            "depth": result["placement_metadata"]["depth"],
            "question_id": question_id[:8],
            "render_css": render_css,
            "render_css_path": render_css_path,
            "collection_id": fp.parent.as_posix(),
            "json_id": fp.stem,
            "model_id": model_id,
        }
        for result in results
    ]


def get_result_json() -> Iterable[Path]:
    if len(sys.argv) > 1:
        files: list[Path] = []
        for arg in sys.argv[1:]:
            p = Path(arg)
            if p.is_dir():
                files.extend(p.glob("**/*.json"))
            else:
                files.append(p)
        return files

    return Path("results").glob("**/*.json")


def plot_tradeoff(
    df: pd.DataFrame,
    output_dir: Path = PLOT_OUTDIR,
    ratio_col=r"$r_\mathtt{VTC}$",
    color_palette="hls",
):
    df["fontsize"] = df["render_css"].apply(
        lambda x: float(x.split("font-size:")[1].split("px")[0])
    )
    df[ratio_col] = (12**0.5 * 2 / df["fontsize"] ** 0.5).round(2)
    df = df[df["fontsize"].isin(list(range(11, 21, 1)))]
    df = df[df["context_length"].isin([1000, 2000, 4000, 8000, 32000])]

    groups = df.groupby("needle_set_path")
    fig, axes = plt.subplots(
        ncols=len(groups),
        figsize=(2.5 * len(groups), 3),
        sharex=True,
    )

    norm = Normalize(vmin=df[ratio_col].min(), vmax=df[ratio_col].max())
    cmap = sns.color_palette(color_palette, as_cmap=True)
    for (col, subtable), ax, idx, title in zip(
        groups,
        reversed(axes),
        range(100),
        ["Reasoning", "Retrieval (S-NIAH)"],
    ):
        sns.lineplot(
            data=subtable,
            x="context_length",
            y="contains_all",
            hue=ratio_col,
            size=1,
            style=ratio_col,
            dashes=False,
            palette=cmap,
            hue_norm=norm,
            markers=["P", "X", "*", "d"],
            markersize=5,
            markeredgewidth=0.2,
            ax=ax,
            legend=False,
            err_kws={"alpha": 0.05},
        )
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.yaxis.set_minor_formatter(ScalarFormatter())
        ax.yaxis.get_minor_formatter().set_scientific(False)
        ax.yaxis.get_minor_formatter().set_useOffset(False)
        if idx == 1:
            ax.set_ylim(50, 102)
        else:
            ax.set_ylim(1, 40)
        ax.tick_params(axis="y", pad=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.grid(which="major", axis="y", color="black", linestyle="--", linewidth=0.8)
        ax.grid(which="minor", axis="y", color="black", linestyle=":", linewidth=0.5)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axes.tolist(), shrink=0.8, fraction=0.02, aspect=40, pad=0.03
    )
    cbar.outline.set_visible(False)  # ty:ignore[call-non-callable]
    cbar.ax.tick_params(length=1, labelsize=6, pad=1, labelrotation=90)
    cbar.ax.set_title(ratio_col, fontsize=8)
    fig.supxlabel("Context Length (tokens)", fontsize=10)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "ratio_tradeoff.pdf", bbox_inches="tight")


def plot_heatmap(df: pd.DataFrame, output_dir: Path = PLOT_OUTDIR):
    # plot heatmap with x=contextlength, y=depth, z=contains_all
    # for each model_id and needle_set_path
    df: pd.DataFrame = df[
        (df["context_length"].notna())
        & (
            (df["render_css"].str.contains("12"))
            | (df["render_css"].str.contains("default") & df["render_css"].isna())
            | (df["pure_text"].isin(["True", True, "true"]))
        )
    ]
    assert not df.empty, f"No valid rows found for heatmap plotting in {df}"

    df = (
        df.groupby(_safe_columns(df, GROUP_BY_COLS + ["depth"]))
        .agg({k: v for k, v in AGG_MAPPING.items() if k in df.columns})
        .reset_index()
    )
    df["depth"] = pd.to_numeric(df["depth"])
    df["context_length"] = pd.to_numeric(
        df["context_length"] / 1000, downcast="integer"
    )

    for (model_id, needle_set_path), subdf in df.groupby(
        ["model_id", "needle_set_path"]
    ):
        save_to: Path = output_dir / model_id / f"{needle_set_path.split('/')[-1]}"
        save_to = save_to.with_suffix(".pdf")
        save_to.parent.mkdir(parents=True, exist_ok=True)
        pivot = (
            subdf.pivot_table(
                index="depth",
                columns="context_length",
                values="contains_all",
                aggfunc="mean",
            )
            .sort_index(ascending=False)
            .sort_index(axis=1)
        )
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(3, 2))
        sns.heatmap(pivot, ax=ax, cmap="Reds_r", vmin=0, vmax=100)
        ax.set_xlabel(r"Context Length ($10^3$ tokens)")
        ax.set_ylabel("Needle Depth")
        ax.tick_params(rotation=0, length=0)
        ax.set_yticks([0.5, len(pivot.index) - 0.5], labels=["1.0", "0.0"])
        ax.collections[0].colorbar.ax.tick_params(  # ty:ignore[unresolved-attribute]
            axis="y", rotation=90, length=0, pad=1
        )

        print(f"Saving heatmap to {save_to!r}. Sample{Counter(subdf['json_id'])}")
        fig.savefig(save_to, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    files = get_result_json()
    results: list[list[dict]] = process_map(
        read_worker,
        files,
        max_workers=16,
        chunksize=1,
        desc=f"Collecting {sys.argv[1:]}",
    )
    assert len(results) > 0, f"No results found in {sys.argv[1:]}"

    df = pd.DataFrame([item for sublist in results for item in sublist])
    # avoid hashing issue
    df[_safe_columns(df, GROUP_BY_COLS)] = df[_safe_columns(df, GROUP_BY_COLS)].astype(
        str
    )
    df[METRIC_COLUMNS] = (df[METRIC_COLUMNS] * 100).round(2)

    if PLOT_HEATMAP:
        plot_heatmap(df.reset_index())
        exit(0)
    if PLOT_TRADEOFF:
        plot_tradeoff(df.reset_index())
        exit(0)

    # aggregate results
    df = (
        df.groupby(_safe_columns(df, GROUP_BY_COLS))
        .agg({k: v for k, v in AGG_MAPPING.items() if k in df.columns})
        .reset_index()
    )

    if not IS_WILD:
        GROUP_BY_COLS.remove("collection_id")
    df = df.set_index(
        _safe_columns(df, GROUP_BY_COLS + ["context_length"])
    ).sort_index()
    df.to_csv("all_results.tsv", sep="\t")
    with pd.option_context("display.width", None, "display.max_colwidth", 30):
        print(df.head())
