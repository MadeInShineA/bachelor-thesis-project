import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")

with app.setup:
    import os
    from typing import Callable
    import marimo as mo
    from pathlib import Path
    import polars as pl
    from scipy.io import loadmat
    from collections import Counter
    import h5py
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pcafeat import (
        select_pca_features,
        select_ttest_features,
        select_optimal_pcs,
    )
    import scipy.io as sio
    from scipy.ndimage import center_of_mass
    from nilearn import plotting
    from matplotlib.lines import Line2D
    import json
    import nibabel as nib
    from nilearn.image import resample_to_img
    import pickle
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    from scipy import stats
    from typing import Tuple

    from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve
    from scipy.io import savemat
    from scipy.linalg import lu_factor, lu_solve
    import matplotlib.image as mpimg
    import subprocess
    from functools import partial
    import io

    from joblib import Parallel, delayed

    import re


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Fuzzy PCA Dimensionality Reduction Analysis

    The goals of this notebook are:
    - Reproduce the results of [this paper](https://direct.mit.edu/imag/article/doi/10.1162/IMAG.a.1121/134875/Extraction-of-robust-functional-connectivity) using the `SRPB` dataset.
    - Perturb the FC matrix extraction.
    - Assess how the perturbations affect the original results.
    - Do the same steps for the `BMB` dataset.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Reproduction of the paper results using the `SRPB` dataset
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Loading the harmonized FC matrices and their metadata
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We start by loading the harmonized FC matrices metadata.

    We use the Glasser parcellation (360 cortical + 54 subcortical + 32 cerebellar = 446 ROIs)
    with global signal regression (GSR) and filtering
    """)
    return


@app.cell
def _():
    harmonized_srpb_fc_matrices_metadata_path = Path(
        "/home/cbi-biomark03/ayumu/HARP/data/preproc_srpb_ts_harmonized/all_data_sub_Glasser_GSR1.csv"
    )
    return (harmonized_srpb_fc_matrices_metadata_path,)


@app.cell
def _(harmonized_srpb_fc_matrices_metadata_path):
    srpb_metadata_df = pl.read_csv(harmonized_srpb_fc_matrices_metadata_path)
    srpb_metadata_df.head()
    return (srpb_metadata_df,)


@app.cell
def _(srpb_metadata_df):
    print(srpb_metadata_df.columns)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And load the harmonized FC matrices themselves.
    """)
    return


@app.cell
def _():
    harmonized_srpb_fc_matrices = []


    with h5py.File(
        "/home/cbi-biomark03/ayumu/HARP/data/preproc_srpb_ts_harmonized/all_data_con_Glasser_GSR1_ortho1_harmonized.mat",
        "r",
    ) as _f:
        harmonized_srpb_fc_matrices = _f["X"][:]

    harmonized_srpb_fc_matrices
    return (harmonized_srpb_fc_matrices,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We now want to merge the metadata DF and the FC_matrices
    """)
    return


@app.cell
def _(harmonized_srpb_fc_matrices, srpb_metadata_df):
    harmonized_srpb_fc_matrices_df = srpb_metadata_df.with_columns(
        harmonized_fc_matrix=harmonized_srpb_fc_matrices
    )
    return (harmonized_srpb_fc_matrices_df,)


@app.cell
def _(harmonized_srpb_fc_matrices_df):
    harmonized_srpb_fc_matrices_df.select("sub_id", "harmonized_fc_matrix").head(1)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And only keep the HC and MDD subjects
    """)
    return


@app.cell
def _(harmonized_srpb_fc_matrices_df):
    harmonized_srpb_fc_matrices_hc_mdd_df = harmonized_srpb_fc_matrices_df.filter(
        pl.col("diag").is_in([0, 2])
    )
    return (harmonized_srpb_fc_matrices_hc_mdd_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Analyze the dataframe
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per disease
    """)
    return


@app.cell
def _(harmonized_srpb_fc_matrices_hc_mdd_df):
    harmonized_srpb_fc_matrices_hc_mdd_df.group_by("diag").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per sex
    """)
    return


@app.cell
def _(harmonized_srpb_fc_matrices_hc_mdd_df):
    harmonized_srpb_fc_matrices_hc_mdd_df.group_by("sex").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per age
    """)
    return


@app.cell
def _(harmonized_srpb_fc_matrices_hc_mdd_df):
    harmonized_srpb_fc_matrices_hc_mdd_df.group_by("age").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per BDI score
    """)
    return


@app.cell
def _(harmonized_srpb_fc_matrices_hc_mdd_df):
    harmonized_srpb_fc_matrices_hc_mdd_df.group_by("bdi").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per site
    """)
    return


@app.cell
def _(harmonized_srpb_fc_matrices_hc_mdd_df):
    harmonized_srpb_fc_matrices_hc_mdd_df.group_by("site").len()
    return


@app.cell
def _(harmonized_srpb_fc_matrices_hc_mdd_df):
    harmonized_srpb_fc_matrices_hc_mdd_df.group_by("meanFD").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Compute and plot the PCA feature selection
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Calculate the differente PCA and T test features for the following metrics:
            - Mental Depressive Disorder (MDD) vs Healthy Control (HC)
            - Age
            - Bdi
            - Sex
            - Site
            - Mean FD
    """)
    return


@app.function
def old_calculate_features(
    target_str: str,
    target_filters: Callable,
    harmonized_fc_matrices_df: pl.DataFrame,
    alpha_threshold: float,
    plot_dir: str,
    ttest_dir,
    ttest_output_prefix: str,
) -> dict:
    harmonized_fc_matrices_pandas_df = harmonized_fc_matrices_df.to_pandas()

    target_col = harmonized_fc_matrices_pandas_df[target_str]

    filtered_df = harmonized_fc_matrices_pandas_df[target_filters(target_col)]

    target = filtered_df[target_str]

    harmonized_fc_matrix_pandas_df = filtered_df[
        "harmonized_fc_matrix"
    ].to_frame()

    col_name = harmonized_fc_matrix_pandas_df.columns[0]

    df_X_train = pd.DataFrame(
        harmonized_fc_matrix_pandas_df[col_name].tolist(),
        index=harmonized_fc_matrix_pandas_df.index,
    )

    (cons, cons_pc) = select_pca_features(
        df_X_train=df_X_train,
        target=target,
        method_pick_pca="fdr_bh",
        alpha=alpha_threshold,
        method_pick_con="fdr_bh",
        fig_dir=plot_dir,
        fig_plot=True,
    )

    if target.nunique() == 2:
        (
            selected_indices,
            t_statistics,
            p_values,
        ) = select_ttest_features(
            df_X_train=df_X_train,
            target=target,
            output_dir=ttest_dir,
            output_prefix=ttest_output_prefix,
            save_results=True,
        )

        return {
            "cons": cons,
            "cons_pc": cons_pc,
            "selected_indices": selected_indices,
            "t_statistics": t_statistics,
            "p_values": p_values,
        }

    else:
        return {
            "cons": cons,
            "cons_pc": cons_pc,
        }


@app.function
def calculate_features(
    target_str: str,
    harmonized_fc_matrices_df: pl.DataFrame,
    alpha_threshold: float,
    n_pcs: int,
    plot_dir: str,
    ttest_dir,
    ttest_output_prefix: str,
) -> dict:
    harmonized_fc_matrices_pandas_df = harmonized_fc_matrices_df.to_pandas()
    target = harmonized_fc_matrices_pandas_df[target_str]
    harmonized_fc_matrix_pandas_df = harmonized_fc_matrices_pandas_df[
        "harmonized_fc_matrix"
    ].to_frame()
    col_name = harmonized_fc_matrix_pandas_df.columns[0]
    df_X_train = pd.DataFrame(
        harmonized_fc_matrix_pandas_df[col_name].tolist(),
        index=harmonized_fc_matrix_pandas_df.index,
    )
    cons, cons_pc, statistics = select_pca_features(
        df_X_train=df_X_train,
        target=target,
        n_pcs=n_pcs,
        return_statistics=True,
        alpha=alpha_threshold,
        method_pick_pca="fdr_bh",
        method_pick_con="fdr_bh",
        fig_dir=plot_dir,
        fig_plot=True,
    )
    top_pc_indices = np.argsort(np.abs(statistics))[::-1][:n_pcs]
    top_scores = np.abs(statistics[top_pc_indices])
    max_score = top_scores.max()
    if max_score > 0:
        normalized_scores = top_scores / max_score
    else:
        normalized_scores = top_scores
    selected_pcs_with_scores = [
        (int(pc), float(score))
        for pc, score in zip(top_pc_indices, normalized_scores)
    ]
    if target.nunique() == 2:
        selected_indices, t_statistics, p_values = select_ttest_features(
            df_X_train=df_X_train,
            target=target,
            output_dir=ttest_dir,
            output_prefix=ttest_output_prefix,
            save_results=True,
        )
        return {
            "cons": cons,
            "cons_pc": cons_pc,
            "statistics": statistics,
            "selected_pcs_with_scores": selected_pcs_with_scores,
            "selected_indices": selected_indices,
            "t_statistics": t_statistics,
            "p_values": p_values,
        }
    else:
        return {
            "cons": cons,
            "cons_pc": cons_pc,
            "statistics": statistics,
            "selected_pcs_with_scores": selected_pcs_with_scores,
        }


@app.cell
def _():
    calculate_features
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Define a function to plot the p values
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Please note that this plot code was generated by Qwen3.7-Plus
    """)
    return


@app.function
def plot_p_values(p_values, metric_name: str):
    # Combine both plots into a single figure (1 row, 2 columns)
    fig, (ax_hist, ax_qq) = plt.subplots(1, 2, figsize=(12, 5))

    # --- 1. Histogram ---
    ax_hist.hist(p_values, bins=50, color="skyblue", edgecolor="black")
    ax_hist.set_xlabel("Raw p-value")
    ax_hist.set_ylabel("Frequency")

    ax_hist.set_title(f"Distribution of raw p-values\n[{metric_name}]")
    ax_hist.axvline(0.05, color="red", linestyle="--", label="p=0.05")
    ax_hist.legend()

    # --- 2. QQ Plot ---
    sorted_p = np.sort(p_values)
    expected_p = np.linspace(1 / len(p_values), 1, len(p_values))

    ax_qq.scatter(
        -np.log10(expected_p),
        -np.log10(sorted_p + 1e-300),
        alpha=0.5,
        color="purple",
        s=10,
        label=f"Observed ({metric_name})",
    )
    max_val = max(-np.log10(expected_p))
    ax_qq.plot(
        [0, max_val], [0, max_val], "r--", label="Null hypothesis (Uniform)"
    )

    ax_qq.set_xlabel("Expected -log10(p-value)")
    ax_qq.set_ylabel("Observed -log10(p-value)")

    ax_qq.set_title(f"QQ Plot of p-values\n[{metric_name}]")
    ax_qq.legend()

    plt.tight_layout()

    return fig


@app.cell
def _():
    plot_p_values
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Define a function to calculate the PCA features extraction for the different metris
    """)
    return


@app.function
def old_calculate_metrics( df: pd.DataFrame,
    metric_dict: dict[str, dict],
    alpha_threshold: float,
    plot_dir: str,
    ttest_dir: str,
    cache_dir: str,
) -> dict:

    os.makedirs(cache_dir, exist_ok=True)

    res = {"results": {}}
    ui_elements = []

    for metric, config in metric_dict.items():
        cache_path = os.path.join(cache_dir, f"{metric}_results.pkl")

        if os.path.exists(cache_path):
            print(f"Loading cached results for metric: {metric}")
            with open(cache_path, "rb") as f:
                results = pickle.load(f)
        else:
            print(f"Calculating and caching results for metric: {metric}")
            results = old_calculate_features(
                target_str=metric,
                target_filters=config["filter_function"],
                harmonized_fc_matrices_df=df,
                alpha_threshold=alpha_threshold,
                plot_dir=plot_dir,
                ttest_dir=ttest_dir,
                ttest_output_prefix=config["prefix"],
            )

            with open(cache_path, "wb") as f:
                pickle.dump(results, f)

        print(results)
        cons = results.get("cons")
        cons_pc = results.get("cons_pc")

        is_binary_target = "p_values" in results

        if is_binary_target:
            selected_indices = results["selected_indices"]
            t_statistics = results["t_statistics"]
            p_values = results["p_values"]

        image_path = os.path.join(plot_dir, f"{metric}.png")
        metric_image = mo.image(src=image_path)

        if is_binary_target:
            plot_section = metric_image
            """
            p_value_plot = plot_p_values(p_values, metric_name=metric)
            plot_section = mo.hstack(
                [metric_image, p_value_plot],
                justify="center",
                align="center",
                gap=2,
                widths=[1, 2],
            )
            """
        else:
            plot_section = metric_image

        metric_ui = mo.vstack(
            [
                mo.md(f"#### Results for metric: `{metric}`"),
                plot_section,
                mo.md("---"),
            ]
        )

        ui_elements.append(metric_ui)

        res["results"][metric] = {
            "cons": cons,
            "cons_pc": cons_pc,
            "is_binary_target": is_binary_target,
        }

        if is_binary_target:
            res["results"][metric].update(
                {
                    "selected_indices": selected_indices,
                    "t_statistics": t_statistics,
                    "p_values": p_values,
                }
            )

    res["ui"] = mo.vstack(ui_elements, gap=3)
    return res


@app.function
def calculate_metrics(
    df: pd.DataFrame,
    metric_dict: dict[str, dict],
    alpha_threshold: float,
    n_pcs: int,
    plot_dir: str,
    ttest_dir: str,
    cache_dir: str,
) -> dict:
    os.makedirs(cache_dir, exist_ok=True)
    res = {"results": {}}
    ui_elements = []
    for metric, config in metric_dict.items():
        cache_path = os.path.join(cache_dir, f"{metric}_results.pkl")
        if os.path.exists(cache_path):
            print(f"Loading cached results for metric: {metric}")
            with open(cache_path, "rb") as f:
                results = pickle.load(f)
        else:
            print(f"Calculating and caching results for metric: {metric}")
            results = calculate_features(
                target_str=metric,
                harmonized_fc_matrices_df=df,
                alpha_threshold=alpha_threshold,
                n_pcs=n_pcs,
                plot_dir=plot_dir,
                ttest_dir=ttest_dir,
                ttest_output_prefix=config["prefix"],
            )
            with open(cache_path, "wb") as f:
                pickle.dump(results, f)
        cons = results.get("cons")
        cons_pc = results.get("cons_pc")
        statistics = results.get("statistics")
        selected_pcs_with_scores = results.get("selected_pcs_with_scores")
        is_binary_target = "p_values" in results
        if is_binary_target:
            selected_indices = results["selected_indices"]
            t_statistics = results["t_statistics"]
            p_values = results["p_values"]
        image_path = os.path.join(plot_dir, f"{metric}.png")
        metric_image = mo.image(src=Path(image_path).read_bytes())
        plot_section = metric_image
        metric_ui = mo.vstack(
            [
                mo.md(f"#### Results for metric: `{metric}`"),
                plot_section,
                mo.md("---"),
            ]
        )
        ui_elements.append(metric_ui)
        res["results"][metric] = {
            "cons": cons,
            "cons_pc": cons_pc,
            "statistics": statistics,
            "selected_pcs_with_scores": selected_pcs_with_scores,
            "is_binary_target": is_binary_target,
        }
        if is_binary_target:
            res["results"][metric].update(
                {
                    "selected_indices": selected_indices,
                    "t_statistics": t_statistics,
                    "p_values": p_values,
                }
            )
    # Build lists for select_optimal_pcs
    target_pcs_lists = []
    primary_target_pcs_lists = []
    noise_pcs_lists = []
    for metric, r in res["results"].items():
        kind = metric_dict[metric].get("kind", "target")
        if kind == "target":
            # Always add to target scoring
            target_pcs_lists.append(r["selected_pcs_with_scores"])
            # If primary, also restrict the candidate pool
            if metric_dict[metric].get("primary", False):
                primary_target_pcs_lists.append(r["selected_pcs_with_scores"])
        elif kind == "noise":
            noise_pcs_lists.append(r["selected_pcs_with_scores"])
    selected_pcs, info = select_optimal_pcs(
        target_pcs_lists,
        noise_pcs_lists,
        n_pcs=n_pcs,
        primary_target_pcs_lists=primary_target_pcs_lists,
    )
    res["selected_pcs"] = selected_pcs
    res["optimal_pc_info"] = info
    res["ui"] = mo.vstack(ui_elements, gap=3)
    return res


@app.cell
def _():
    calculate_metrics
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Calculate the differente PCA and T test features for the following metrics:
    - Mental Depressive Disorder (MDD) vs Healthy Control (HC)
    - Age
    - Bdi
    - Sex
    - Site
    - Mean FD
    """)
    return


@app.cell
def _():
    old_srpb_plot_dir = "./res/pca-dim-reduction/srpb/features-extraction/old/plots/"
    old_srpb_ttest_dir = "./res/pca-dim-reduction/srpb/features-extraction/old/t-tests/"
    old_srpb_cache_dir = "./res/pca-dim-reduction/srpb/features-extraction/old/cache/"
    old_srpb_metadata_dir = (
        "./res/pca-dim-reduction/srpb/features-extraction/old/metadatas/"
    )
    return old_srpb_cache_dir, old_srpb_plot_dir, old_srpb_ttest_dir


@app.cell
def _():
    srpb_plot_dir = "./res/pca-dim-reduction/srpb/features-extraction/plots/"
    srpb_ttest_dir = "./res/pca-dim-reduction/srpb/features-extraction/t-tests/"
    srpb_cache_dir = "./res/pca-dim-reduction/srpb/features-extraction/cache/"
    srpb_metadata_dir = (
        "./res/pca-dim-reduction/srpb/features-extraction/metadatas/"
    )
    return srpb_cache_dir, srpb_metadata_dir, srpb_plot_dir, srpb_ttest_dir


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Define the different metrics we want to extract and their filters
    """)
    return


@app.cell
def _():
    old_metric_dict = {
        "diag": {
            "prefix": "ttest_diag",
            "filter_function": lambda target_col: (
                (target_col != -1000) & pd.notna(target_col)
            ),
        },
        "bdi": {
            "prefix": "ttest_bdi",
            "filter_function": lambda target_col: (
                (target_col != -1000) & pd.notna(target_col)
            ),
        },
        "age": {
            "prefix": "ttest_age",
            "filter_function": lambda target_col: (
                (target_col != -1000) & pd.notna(target_col)
            ),
        },
        "sex": {
            "prefix": "ttest_sex",
            "filter_function": lambda target_col: (
                (target_col != -1000) & pd.notna(target_col)
            ),
        },
        "site": {
            "prefix": "ttest_site",
            "filter_function": lambda target_col: pd.notna(target_col),
        },
        "meanFD": {
            "prefix": "ttest_meanFD",
            "filter_function": lambda target_col: pd.notna(target_col),
        },
    }
    return (old_metric_dict,)


@app.cell
def _():
    metric_dict = {
        "diag": {
            "prefix": "ttest_diag",
            "kind": "target",
            "primary": True,
        },
        "bdi": {
            "prefix": "ttest_bdi",
            "kind": "target",
            "primary": False,
        },
        "age": {
            "prefix": "ttest_age",
            "kind": "noise",
        },
        "sex": {
            "prefix": "ttest_sex",
            "kind": "noise",
        },
        "site": {
            "prefix": "ttest_site",
            "kind": "noise",
        },
        "meanFD": {
            "prefix": "ttest_meanFD",
            "kind": "noise",
        },
    }
    return (metric_dict,)


@app.function
def srpb_filter(df: pl.DataFrame):
    return df.filter(
        (df["diag"] != -1000)
        & df["diag"].is_not_null()
        & (df["bdi"] != -1000)
        & df["bdi"].is_not_null()
        & (df["age"] != -1000)
        & df["age"].is_not_null()
        & (df["sex"] != -1000)
        & df["sex"].is_not_null()
        & df["site"].is_not_null()
        & df["meanFD"].is_not_null()
    )


@app.cell
def _(harmonized_srpb_fc_matrices_hc_mdd_df):
    harmonized_srpb_fc_matrices_hc_mdd_df.head()
    return


@app.cell
def _(
    harmonized_srpb_fc_matrices_hc_mdd_df,
    old_metric_dict,
    old_srpb_cache_dir,
    old_srpb_plot_dir,
    old_srpb_ttest_dir,
):
    old_srpb_metrics_dict = old_calculate_metrics(
        df=srpb_filter(harmonized_srpb_fc_matrices_hc_mdd_df),
        metric_dict=old_metric_dict,
        alpha_threshold=0.05,
        plot_dir=old_srpb_plot_dir,
        ttest_dir=old_srpb_ttest_dir,
        cache_dir=old_srpb_cache_dir,
    )

    old_srpb_results = old_srpb_metrics_dict["results"]
    old_srpb_ui = old_srpb_metrics_dict["ui"]
    return (old_srpb_results,)


@app.cell
def _(
    harmonized_srpb_fc_matrices_hc_mdd_df,
    metric_dict,
    srpb_cache_dir,
    srpb_plot_dir,
    srpb_ttest_dir,
):
    srpb_metrics_dict = calculate_metrics(
        df=srpb_filter(harmonized_srpb_fc_matrices_hc_mdd_df),
        metric_dict=metric_dict,
        alpha_threshold=0.05,
        n_pcs=5,
        plot_dir=srpb_plot_dir,
        ttest_dir=srpb_ttest_dir,
        cache_dir=srpb_cache_dir,
    )

    srpb_results = srpb_metrics_dict["results"]
    srpb_selected_pcs = srpb_metrics_dict["selected_pcs"]
    srpb_ui = srpb_metrics_dict["ui"]
    return srpb_results, srpb_selected_pcs, srpb_ui


@app.cell
def _(srpb_selected_pcs):
    srpb_selected_pcs
    return


@app.cell
def _(srpb_ui):
    srpb_ui
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Analyze the results of the PCA feature selection
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We first want to check wich PC was selected only in the BDI and diag metrics
    """)
    return


@app.function
def select_mdd_pc(
    result_dict: dict, metrics_to_accept_beside_diag: list
) -> dict:

    diag_pcs = set()
    accepted_pcs = set()
    confound_pcs = set()

    pc_to_confound_metrics = {}

    for metric, value in result_dict.items():
        pcs = set(value.get("cons_pc", []))

        if metric == "diag":
            diag_pcs = pcs
        elif metric in metrics_to_accept_beside_diag:
            accepted_pcs.update(pcs)
        else:
            confound_pcs.update(pcs)
            for pc in pcs:
                if pc not in pc_to_confound_metrics:
                    pc_to_confound_metrics[pc] = []
                pc_to_confound_metrics[pc].append(metric)

    in_diag_and_confounds = diag_pcs & confound_pcs

    in_diag_and_accepted = (diag_pcs & accepted_pcs) - confound_pcs

    only_in_diag = diag_pcs - accepted_pcs - confound_pcs

    detailed_confounds = {}
    for pc in sorted(list(in_diag_and_confounds)):
        detailed_confounds[pc] = pc_to_confound_metrics[pc]

    return {
        "only_in_diag": sorted(list(only_in_diag)),
        "in_diag_and_accepted": sorted(list(in_diag_and_accepted)),
        "in_diag_and_confounds": detailed_confounds,
    }


@app.cell
def _():
    select_mdd_pc
    return


@app.cell
def _(old_srpb_results):
    old_srpb_selected_mdd_pcs = select_mdd_pc(old_srpb_results, ["bdi"])
    old_srpb_selected_mdd_pcs
    return


@app.cell
def _(srpb_results):
    srpb_selected_mdd_pcs = select_mdd_pc(srpb_results, ["bdi"])
    srpb_selected_mdd_pcs
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We now want to plot the different FC of the desired PC
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We start by obtaining the brain parcellation
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Please note that all the code of this cell has been written by Qwen3.7-Plus and needs to be double-checked
    """)
    return


@app.cell
def _():
    parcel_folder = "/home/cbi-data34/ayumu/tools/Parcellations/"
    cortex_file = (
        f"{parcel_folder}/GlasserYeo/HCP-MMP1_on_MNI152_ICBM2009a_nlin.nii.gz"
    )
    subcortex_file = f"{parcel_folder}/Tian-Subcortex-Only/Tian_Subcortex_S4_3T_2009cAsym.nii.gz"
    cerebellum_file = f"{parcel_folder}/Nettekoven_2023_cerebellar/atl-NettekovenAsym32_space-MNI152NLin6AsymC_dseg.nii"

    subcortex_img = nib.load(subcortex_file)
    cortex_img = nib.load(cortex_file)
    cerebellum_img = nib.load(cerebellum_file)

    # Resample to subcortex space
    cortex_resampled = resample_to_img(
        cortex_img, subcortex_img, interpolation="nearest"
    )
    cortex_volume = np.round(cortex_resampled.get_fdata()).astype(int)
    cerebellum_resampled = resample_to_img(
        cerebellum_img, subcortex_img, interpolation="nearest"
    )
    cerebellum_volume = np.round(cerebellum_resampled.get_fdata()).astype(int)
    subcortex_volume = subcortex_img.get_fdata().astype(int)

    # Split left/right hemisphere for Cortex (adds 180 to left side)
    affine = cortex_resampled.affine
    shape = cortex_volume.shape
    i_coords = np.arange(shape[0])
    mni_x = affine[0, 0] * i_coords + affine[0, 3]
    left_hemisphere_mask = mni_x < 0
    left_mask_3d = left_hemisphere_mask[:, np.newaxis, np.newaxis]

    cortex_volume_processed = cortex_volume.copy()
    cortex_mask = cortex_volume > 0
    left_cortex_mask = cortex_mask & left_mask_3d
    cortex_volume_processed[left_cortex_mask] = (
        cortex_volume[left_cortex_mask] + 180
    )

    # Combine into single 446-label volume
    roi = np.zeros(np.shape(subcortex_volume), dtype=np.int32)
    roi[cortex_mask] = cortex_volume_processed[cortex_mask]
    subcortex_mask = subcortex_volume > 0
    roi[subcortex_mask] = subcortex_volume[subcortex_mask] + np.max(roi)
    cerebellum_mask = cerebellum_volume > 0
    roi[cerebellum_mask] = cerebellum_volume[cerebellum_mask] + np.max(roi)

    # ==========================================
    # 2. Calculate 3D MNI Coordinates
    # ==========================================
    # We use the affine of the subcortex_img because that's the space we resampled everything to
    target_affine = subcortex_img.affine

    labels = np.arange(1, 447)
    coords_voxel = np.array(center_of_mass(roi, labels=roi, index=labels))

    # Convert voxel indices to MNI coordinates (mm)
    coords_hom = np.hstack([coords_voxel, np.ones((446, 1))])
    coords_mni = (target_affine @ coords_hom.T).T[:, :3]

    # ==========================================
    # 3. Get Network Assignments for the Legend
    # ==========================================
    # Load the Glasser to Yeo mapping
    glasser_yeo_file = f"{parcel_folder}/GlasserYeo/Glasser2Yeo.csv"
    glasser_yeo = pd.read_csv(glasser_yeo_file)

    network_col = (
        "Yeo7" if "Yeo7" in glasser_yeo.columns else glasser_yeo.columns[1]
    )

    # ==========================================
    # MAP NUMBERS TO NETWORK NAMES
    # ==========================================
    # Standard Yeo 7-network mapping (adjust if your CSV uses different numbering)
    yeo_number_to_name = {
        0: "Visual",
        1: "SomatoMotor",
        2: "DorsalAttention",
        3: "DefaultMode",
        4: "Limbic",
        5: "Salience",
        6: "PrefrontalControlA",
        7: "PrefrontalControlB",
    }

    # Convert the numbers to names
    cortex_networks_raw = glasser_yeo[network_col].tolist()

    cortex_networks_clean = [
        yeo_number_to_name.get(int(val), "Unknown") for val in cortex_networks_raw
    ]

    # Build the full 446-length list
    network_assignments = []
    network_assignments.extend(
        cortex_networks_clean
    )  # 1-360: Glasser Yeo Networks
    network_assignments.extend(["Subcortical"] * 54)  # 361-414
    network_assignments.extend(["Cerebellum"] * 32)  # 415-446

    print(
        f"Network assignments created. Unique networks: {set(network_assignments)}"
    )

    # ==========================================
    # 4. Save to Disk
    # ==========================================
    out_dir = "./res/pca-dim-reduction/"
    os.makedirs(out_dir, exist_ok=True)  # Ensure the folder exists

    np.save(os.path.join(out_dir, "node_coords_446.npy"), coords_mni)

    # Save network assignments as a simple text file (one per line)
    with open(os.path.join(out_dir, "network_assignments_446.txt"), "w") as _f:
        for net in network_assignments:
            _f.write(f"{net}\n")
    return coords_mni, network_assignments


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And define the plotting function
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Please note that all the code of this cell has been written by Qwen3.7-Plus and needs to be double-checked
    """)
    return


@app.function(hide_code=True)
def plot_pcs(
    results: dict,
    fc_matrices_df: pl.DataFrame,
    node_coords: np.ndarray,
    pcs_to_plot: list,
    plot_dir: str,
    metadata_dir: str,
    node_colors=None,
    show_legend=True,
    network_assignments=None,
    skip_existing: bool = True,
) -> dict:
    """
    Generates plot_pcs brain plots for a list of Principal Components.
    Returns a dictionary with:
      - 'results': adjacency matrices keyed by PC index
      - 'ui_elements': individual Marimo UI elements keyed by PC index
    """
    res = {"results": {}, "ui_elements": {}}

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    glasser_network_colors = {
        "Visual": "#00BFC4",
        "SomatoMotor": "#F8766D",
        "DorsalAttention": "#7CAE00",
        "DefaultMode": "#FFC300",
        "Limbic": "#FF61C3",
        "Salience": "#C77CFF",
        "PrefrontalControlA": "#00BAFF",
        "PrefrontalControlB": "#0097A7",
        "Subcortical": "#333333",
        "Cerebellum": "#D2691E",
        "Unknown": "#000000",
    }

    # 1. Extract Data (once for all PCs)
    target_col = fc_matrices_df["diag"]
    mask = (
        (target_col != -1000)
        & (target_col.is_not_null())
        & ((target_col == 0) | (target_col == 2))
    )
    filtered_df = fc_matrices_df.filter(mask)
    group0_fc = filtered_df.filter(pl.col("diag") == 0)[
        "harmonized_fc_matrix"
    ].to_list()
    group2_fc = filtered_df.filter(pl.col("diag") == 2)[
        "harmonized_fc_matrix"
    ].to_list()
    mean_diff = np.mean(np.array(group2_fc), axis=0) - np.mean(
        np.array(group0_fc), axis=0
    )
    n_nodes = int((1 + np.sqrt(1 + 8 * len(mean_diff))) / 2)
    max_edge_idx = n_nodes * (n_nodes - 1) // 2 - 1

    if n_nodes > len(node_coords):
        padding = np.zeros((n_nodes - len(node_coords), 3))
        node_coords = np.vstack([node_coords, padding])
        if network_assignments is not None:
            network_assignments = network_assignments + ["Unknown"] * (
                n_nodes - len(network_assignments)
            )

    if node_colors is None:
        if network_assignments is not None:
            node_colors = [
                glasser_network_colors.get(str(net), "#000000")
                for net in network_assignments
            ]
        else:
            node_colors = [
                "#a6cee3" if coord[0] < 0 else "#fb9a99"
                for coord in node_coords
            ]
    elif isinstance(node_colors, list) and len(node_colors) < n_nodes:
        missing = n_nodes - len(node_colors)
        try:
            colors_arr = np.array(node_colors)
            if colors_arr.ndim == 2 and colors_arr.shape[1] in [3, 4]:
                gray_val = (
                    [0.8, 0.8, 0.8, 1.0]
                    if colors_arr.shape[1] == 4
                    else [0.8, 0.8, 0.8]
                )
                padding_colors = np.tile(gray_val, (missing, 1))
                node_colors = np.vstack([colors_arr, padding_colors])
            else:
                node_colors = node_colors + ["#000000"] * missing
        except Exception:
            node_colors = node_colors + ["#000000"] * missing

    # 2. Loop through each requested PC
    for pc_idx in pcs_to_plot:
        image_path = os.path.join(plot_dir, f"brain_pc_{pc_idx + 1}fc.png")
        json_path = os.path.join(
            metadata_dir, f"brain_pc_{pc_idx + 1}_metadata.json"
        )

        plot_exists = os.path.exists(image_path) and os.path.exists(json_path)
        cache_valid = False

        if skip_existing and plot_exists:
            try:
                with open(json_path, "r") as f:
                    metadata = json.load(f)
                saved_edges = metadata.get("edge_indices", [])
                if saved_edges:
                    max_saved = max(int(float(e)) for e in saved_edges)
                    if max_saved <= max_edge_idx:
                        cache_valid = True
                    else:
                        print(
                            f"Cache incompatible for PC {pc_idx + 1}, regenerating..."
                        )
                else:
                    cache_valid = True
            except Exception as e:
                print(f"Cache corrupt for PC {pc_idx + 1}, regenerating: {e}")
                cache_valid = False

        if skip_existing and cache_valid:
            print(
                f"PC {pc_idx + 1} (Index {pc_idx}) already exists, loading from cache: {image_path}"
            )

            adj_matrix = np.zeros((n_nodes, n_nodes))
            i_idx, j_idx = np.tril_indices(n_nodes, k=1)
            for edge_idx in metadata["edge_indices"]:
                edge_idx = int(float(edge_idx))
                i, j = i_idx[edge_idx], j_idx[edge_idx]
                adj_matrix[i, j] = mean_diff[edge_idx]
                adj_matrix[j, i] = mean_diff[edge_idx]

            res["results"][pc_idx] = {
                "adj_matrix": adj_matrix,
                "edges": metadata["edge_indices"],
                "num_edges": metadata["num_edges"],
            }

            pc_image = mo.image(src=Path(image_path).read_bytes())
            vmin = metadata["color_range"]["vmin"]
            vmax = metadata["color_range"]["vmax"]
            description_md = f"""
            #### Plot Description for PC {pc_idx + 1}
            - **Total unique edges displayed:** {metadata["num_edges"]}
            - **Range:** `[{vmin:.3f}, {vmax:.3f}]`
            - **RED:** Over-connectivity in MDD (MDD > HC)
            - **BLUE:** Under-connectivity in MDD (MDD < HC)
            """

            network_stats_md = ""
            if "network_statistics" in metadata:
                net_stats = metadata["network_statistics"]
                network_stats_md = "#### Participating Nodes per Network\n\n| Network | Node Count |\n| :--- | :---: |\n"
                for net_name, node_count in net_stats.get(
                    "nodes_per_network", {}
                ).items():
                    network_stats_md += f"| {net_name} | {node_count} |\n"

                network_stats_md += "\n#### FC Edges per Network\n\n| Network | Total Edges | Intra-Network | % of Total |\n| :--- | :---: | :---: | :---: |\n"
                per_network = net_stats.get("per_network", {})
                intra_network = net_stats.get("intra_network", {})
                for net_name, total_count in per_network.items():
                    intra_count = intra_network.get(net_name, 0)
                    percentage = (
                        total_count / (2 * metadata["num_edges"])
                    ) * 100
                    network_stats_md += f"| {net_name} | {total_count} | {intra_count} | {percentage:.1f}% |\n"

                network_stats_md += "\n#### Top Inter-Network Connections\n\n| Connection Pair | Count |\n| :--- | :---: |\n"
                for pair_name, count in list(
                    net_stats.get("inter_network", {}).items()
                )[:10]:
                    network_stats_md += f"| {pair_name} | {count} |\n"
                network_stats_md += "\n---\n\n"

            pc_ui_elements = [mo.md(description_md)]
            if network_stats_md:
                pc_ui_elements.append(mo.md(network_stats_md))
            pc_ui_elements.append(pc_image)

            pc_ui = mo.vstack(
                [mo.vstack(pc_ui_elements, gap=2), mo.md("---")], gap=2
            )
            res["ui_elements"][pc_idx] = pc_ui
            continue

        # GENERATION BRANCH
        print(f"Generating plot for PC {pc_idx + 1} (Index {pc_idx})...")

        diag_data = results["diag"]
        mask_pc = np.isclose(diag_data["cons_pc"], pc_idx)
        edges_to_plot = list(diag_data["cons"][mask_pc])

        if not edges_to_plot:
            print(
                f"  PC {pc_idx + 1} has no edges in results — creating empty placeholder."
            )

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(
                0.5,
                0.5,
                f"PC {pc_idx + 1}\nNo significant edges",
                ha="center",
                va="center",
                fontsize=16,
                transform=ax.transAxes,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            fig.patch.set_facecolor("white")
            fig.savefig(
                image_path, dpi=150, bbox_inches="tight", facecolor="white"
            )
            plt.close(fig)

            metadata = {
                "pc_index_0_based": float(pc_idx),
                "pc_number_1_based": float(pc_idx + 1),
                "num_edges": 0,
                "color_range": {"vmin": 0.0, "vmax": 0.0},
                "edge_indices": [],
                "network_statistics": {
                    "nodes_per_network": {},
                    "per_network": {},
                    "intra_network": {},
                    "inter_network": {},
                },
            }
            with open(json_path, "w") as f:
                json.dump(metadata, f, indent=4)

            pc_image = mo.image(src=Path(image_path).read_bytes())
            description_md = f"#### Plot Description for PC {pc_idx + 1}\n- **No edges found** — this PC was not selected."
            pc_ui = mo.vstack(
                [mo.md(description_md), pc_image, mo.md("---")], gap=2
            )
            res["ui_elements"][pc_idx] = pc_ui

            res["results"][pc_idx] = {
                "adj_matrix": np.zeros((n_nodes, n_nodes)),
                "edges": [],
                "num_edges": 0,
            }
            continue

        if max(int(float(e)) for e in edges_to_plot) > max_edge_idx:
            raise ValueError(
                f"Data mismatch! PC {pc_idx + 1} has edge indices that don't fit "
                f"current n_nodes={n_nodes}. Max allowed: {max_edge_idx}, got {max(edges_to_plot)}."
            )

        adj_matrix = np.zeros((n_nodes, n_nodes))
        i_idx, j_idx = np.tril_indices(n_nodes, k=1)
        for edge_idx in edges_to_plot:
            edge_idx = int(float(edge_idx))
            i, j = i_idx[edge_idx], j_idx[edge_idx]
            adj_matrix[i, j] = mean_diff[edge_idx]
            adj_matrix[j, i] = mean_diff[edge_idx]

        participating = np.zeros(n_nodes, dtype=bool)
        for edge_idx in edges_to_plot:
            edge_idx = int(float(edge_idx))
            participating[i_idx[edge_idx]] = True
            participating[j_idx[edge_idx]] = True
        node_sizes = np.where(participating, 20, 0)

        network_node_counts = {}
        network_edge_counts = {}
        intra_network_edges = {}
        inter_network_edges = {}

        network_stats_md = ""
        if network_assignments is not None:
            for node_idx in range(n_nodes):
                if participating[node_idx]:
                    net = (
                        network_assignments[node_idx]
                        if node_idx < len(network_assignments)
                        else "Unknown"
                    )
                    network_node_counts[net] = (
                        network_node_counts.get(net, 0) + 1
                    )

            for edge_idx in edges_to_plot:
                edge_idx = int(float(edge_idx))
                i, j = i_idx[edge_idx], j_idx[edge_idx]
                net_i = (
                    network_assignments[i]
                    if i < len(network_assignments)
                    else "Unknown"
                )
                net_j = (
                    network_assignments[j]
                    if j < len(network_assignments)
                    else "Unknown"
                )
                network_edge_counts[net_i] = (
                    network_edge_counts.get(net_i, 0) + 1
                )
                network_edge_counts[net_j] = (
                    network_edge_counts.get(net_j, 0) + 1
                )
                if net_i == net_j:
                    intra_network_edges[net_i] = (
                        intra_network_edges.get(net_i, 0) + 1
                    )
                else:
                    str_i, str_j = str(net_i), str(net_j)
                    pair_name = (
                        f"{str_i} ↔ {str_j}"
                        if str_i < str_j
                        else f"{str_j} ↔ {str_i}"
                    )
                    inter_network_edges[pair_name] = (
                        inter_network_edges.get(pair_name, 0) + 1
                    )

            network_stats_md = "#### Participating Nodes per Network\n\n| Network | Node Count |\n| :--- | :---: |\n"
            for net_name, node_count in sorted(
                network_node_counts.items(), key=lambda x: x[1], reverse=True
            ):
                network_stats_md += f"| {net_name} | {node_count} |\n"

            network_stats_md += "\n#### FC Edges per Network\n\n| Network | Total Edges | Intra-Network | % of Total |\n| :--- | :---: | :---: | :---: |\n"
            for net_name, total_count in sorted(
                network_edge_counts.items(), key=lambda x: x[1], reverse=True
            ):
                intra_count = intra_network_edges.get(net_name, 0)
                percentage = (total_count / (2 * len(edges_to_plot))) * 100
                network_stats_md += f"| {net_name} | {total_count} | {intra_count} | {percentage:.1f}% |\n"

            network_stats_md += "\n#### Top Inter-Network Connections\n\n| Connection Pair | Count |\n| :--- | :---: |\n"
            for pair_name, count in sorted(
                inter_network_edges.items(), key=lambda x: x[1], reverse=True
            )[:10]:
                network_stats_md += f"| {pair_name} | {count} |\n"
            network_stats_md += "\n---\n\n"

        abs_vals = np.abs(adj_matrix[adj_matrix != 0])
        vmax = np.percentile(abs_vals, 95) if len(abs_vals) > 0 else 0.1
        vmin = -vmax

        if show_legend:
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(
                2, 3, width_ratios=[1, 1, 0.5], wspace=0.3, hspace=0.3
            )
            ax1, ax2, ax3, ax4 = (
                fig.add_subplot(gs[0, 0]),
                fig.add_subplot(gs[0, 1]),
                fig.add_subplot(gs[1, 0]),
                fig.add_subplot(gs[1, 1]),
            )
            ax_legend = fig.add_subplot(gs[:, 2])
            axes = [ax1, ax2, ax3, ax4]
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            axes = axes.flatten()

        views = [
            ("l", "Left Lateral", False),
            ("z", "Axial", True),
            ("r", "Right Lateral", False),
            ("z", "Axial (different slice)", False),
        ]
        for ax, (mode, title_text, show_cbar) in zip(axes, views):
            plotting.plot_connectome(
                adj_matrix,
                node_coords,
                edge_cmap="bwr",
                edge_vmin=vmin,
                edge_vmax=vmax,
                node_size=node_sizes,
                node_color=node_colors,
                display_mode=mode,
                axes=ax,
                title=title_text,
                colorbar=show_cbar,
                alpha=0.8,
                edge_threshold=0.01,
            )

        if show_legend:
            legend_elements = []
            if network_assignments is not None and network_node_counts:
                for net_name in sorted(
                    network_node_counts.keys(),
                    key=lambda x: network_node_counts[x],
                    reverse=True,
                ):
                    color = glasser_network_colors.get(
                        str(net_name), "#000000"
                    )
                    legend_elements.append(
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            markerfacecolor=color,
                            markersize=12,
                            label=net_name,
                            markeredgecolor="black",
                            markeredgewidth=0.5,
                        )
                    )
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color="red",
                    linewidth=2,
                    label="Over-connectivity (MDD > HC)",
                )
            )
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color="blue",
                    linewidth=2,
                    label="Under-connectivity (MDD < HC)",
                )
            )
            ax_legend.axis("off")
            ax_legend.legend(
                handles=legend_elements,
                loc="center",
                fontsize=10,
                frameon=True,
                fancybox=True,
                shadow=True,
                title="Glasser Networks",
                title_fontsize=11,
            )

        plt.suptitle(
            f"PC {pc_idx + 1} (Index {pc_idx}) - {len(edges_to_plot)} edges",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        fig.patch.set_facecolor("white")
        fig.savefig(
            image_path, dpi=150, bbox_inches="tight", facecolor="white"
        )

        metadata = {
            "pc_index_0_based": float(pc_idx),
            "pc_number_1_based": float(pc_idx + 1),
            "num_edges": int(len(edges_to_plot)),
            "color_range": {"vmin": float(vmin), "vmax": float(vmax)},
            "edge_indices": [int(float(e)) for e in edges_to_plot],
            "network_statistics": {
                "nodes_per_network": {
                    k: int(v)
                    for k, v in sorted(
                        network_node_counts.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                },
                "per_network": {
                    k: int(v)
                    for k, v in sorted(
                        network_edge_counts.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                },
                "intra_network": {
                    k: int(v)
                    for k, v in sorted(
                        intra_network_edges.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                },
                "inter_network": {
                    k: int(v)
                    for k, v in sorted(
                        inter_network_edges.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                },
            },
        }
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=4)

        plt.close(fig)
        pc_image = mo.image(src=Path(image_path).read_bytes())

        description_md = f"""
        #### Plot Description for PC {pc_idx + 1}
        - **Total unique edges displayed:** {len(edges_to_plot)}
        - **Range:** `[{vmin:.3f}, {vmax:.3f}]`
        - **RED:** Over-connectivity in MDD (MDD > HC)
        - **BLUE:** Under-connectivity in MDD (MDD < HC)
        """

        pc_ui_elements = [mo.md(description_md)]
        if network_stats_md:
            pc_ui_elements.append(mo.md(network_stats_md))
        pc_ui_elements.append(pc_image)

        pc_ui = mo.vstack(
            [mo.vstack(pc_ui_elements, gap=2), mo.md("---")], gap=2
        )
        res["ui_elements"][pc_idx] = pc_ui

        res["results"][pc_idx] = {
            "adj_matrix": adj_matrix,
            "edges": edges_to_plot,
            "num_edges": len(edges_to_plot),
        }

    return res


@app.cell
def _():
    plot_pcs
    return


@app.cell
def _(
    coords_mni,
    harmonized_srpb_fc_matrices_hc_mdd_df,
    network_assignments,
    srpb_metadata_dir,
    srpb_plot_dir,
    srpb_results,
):
    srpb_target_pcs_to_plot = [1.0, 69.0]

    srpb_pc_plots_results = plot_pcs(
        results=srpb_results,
        fc_matrices_df=harmonized_srpb_fc_matrices_hc_mdd_df,
        node_coords=coords_mni,
        pcs_to_plot=srpb_target_pcs_to_plot,
        plot_dir=srpb_plot_dir,
        metadata_dir=srpb_metadata_dir,
        show_legend=True,
        network_assignments=network_assignments,
    )
    return (srpb_pc_plots_results,)


@app.cell
def _(srpb_pc_plots_results):
    mo.vstack(list(srpb_pc_plots_results["ui_elements"].values()), gap=3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### First draft of a conclusion
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The analysis of the BMB dataset confirms what was established in the first conclusion. While the selected PCs differ across datasets, their contributing FCs come from the same brain areas and show a majority of under-connectivity for MDD subjects.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Reproduction of the paper results using the `BMB` dataset
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Loading the harmonized FC matrices and their metadata
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We start by loading the harmonized FC matrices metadata.

    We use the same Glasser parcellation (446 ROIs) with GSR as for the SRPB dataset,
    to keep the analysis consistent across datasets.
    """)
    return


@app.cell
def _():
    harmonized_bmb_fc_matrices_metadata_path = Path(
        "/home/cbi-biomark03/ayumu/HARP/data/preproc_bmb_ts_harmonized/all_data_sub_Glasser_GSR1.csv"
    )
    return (harmonized_bmb_fc_matrices_metadata_path,)


@app.cell
def _(harmonized_bmb_fc_matrices_metadata_path):
    harmonized_bmb_fc_matrices_metadata_df = pl.read_csv(
        harmonized_bmb_fc_matrices_metadata_path
    )
    harmonized_bmb_fc_matrices_metadata_df.head()
    return (harmonized_bmb_fc_matrices_metadata_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And load the harmonized FC matrices themselves.
    """)
    return


@app.cell
def _():
    harmonized_bmb_fc_matrices = []


    with h5py.File(
        "/home/cbi-biomark03/ayumu/HARP/data/preproc_bmb_ts_harmonized/all_data_con_Glasser_GSR1_ortho1_harmonized.mat",
        "r",
    ) as _f:
        harmonized_bmb_fc_matrices = _f["X"][:]

    harmonized_bmb_fc_matrices
    return (harmonized_bmb_fc_matrices,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We now want to merge the metadata DF and the FC_matrices
    """)
    return


@app.cell
def _(harmonized_bmb_fc_matrices, harmonized_bmb_fc_matrices_metadata_df):
    harmonized_bmb_fc_matrices_df = (
        harmonized_bmb_fc_matrices_metadata_df.with_columns(
            harmonized_fc_matrix=harmonized_bmb_fc_matrices
        )
    )
    return (harmonized_bmb_fc_matrices_df,)


@app.cell
def _(harmonized_bmb_fc_matrices_df):
    harmonized_bmb_fc_matrices_df.head()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And only keep the HC and MDD subjects
    """)
    return


@app.cell
def _(harmonized_bmb_fc_matrices_df):
    harmonized_bmb_fc_matrices_hc_mdd_df = harmonized_bmb_fc_matrices_df.filter(
        pl.col("diag").is_in([0, 2])
    )
    return (harmonized_bmb_fc_matrices_hc_mdd_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Analyze the dataframe
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per disease
    """)
    return


@app.cell
def _(harmonized_bmb_fc_matrices_hc_mdd_df):
    harmonized_bmb_fc_matrices_hc_mdd_df.group_by("diag").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per sex
    """)
    return


@app.cell
def _(harmonized_bmb_fc_matrices_df):
    harmonized_bmb_fc_matrices_df.group_by("sex").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per age
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Please note that age is a float, likely the exact age at the time of the scan (e.g. 24.7 years), rather than an integer rounded to whole years.
    """)
    return


@app.cell
def _(harmonized_bmb_fc_matrices_hc_mdd_df):
    harmonized_bmb_fc_matrices_hc_mdd_df.group_by("age").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per BDI score
    """)
    return


@app.cell
def _(harmonized_bmb_fc_matrices_hc_mdd_df):
    harmonized_bmb_fc_matrices_hc_mdd_df.group_by("bdi").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per site
    """)
    return


@app.cell
def _(harmonized_bmb_fc_matrices_hc_mdd_df):
    harmonized_bmb_fc_matrices_hc_mdd_df.group_by("site").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per mean FD
    """)
    return


@app.cell
def _(harmonized_bmb_fc_matrices_hc_mdd_df):
    harmonized_bmb_fc_matrices_hc_mdd_df.group_by("meanFD").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Compute and plot the PCA feature selection for the whole dataset
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Calculate the differente PCA and T test features for the following metrics:
    - Mental Depressive Disorder (MDD) vs Healthy Control (HC)
    - Age
    - Bdi
    - Sex
    - Site
    - Mean FD
    """)
    return


@app.cell
def _():
    old_bmb_plot_dir = "./res/pca-dim-reduction/bmb/features-extraction/old/plots/"
    old_bmb_ttest_dir = "./res/pca-dim-reduction/bmb/features-extraction/old/t-tests/"
    old_bmb_cache_dir = "./res/pca-dim-reduction/bmb/features-extraction/old/cache/"
    old_bmb_metadata_dir = "./res/pca-dim-reduction/bmb/features-extraction/old/metadatas/"
    return old_bmb_cache_dir, old_bmb_plot_dir, old_bmb_ttest_dir


@app.cell
def _():
    bmb_plot_dir = "./res/pca-dim-reduction/bmb/features-extraction/plots/"
    bmb_ttest_dir = "./res/pca-dim-reduction/bmb/features-extraction/t-tests/"
    bmb_cache_dir = "./res/pca-dim-reduction/bmb/features-extraction/cache/"
    bmb_metadata_dir = "./res/pca-dim-reduction/bmb/features-extraction/metadatas/"
    return bmb_cache_dir, bmb_metadata_dir, bmb_plot_dir, bmb_ttest_dir


@app.cell
def _(harmonized_bmb_fc_matrices_hc_mdd_df):
    print(harmonized_bmb_fc_matrices_hc_mdd_df.columns)
    return


@app.function
def bmb_filter(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(
        (df["diag"] != -1000)
        & df["diag"].is_not_null()
        & (df["bdi"] != -1000)
        & df["bdi"].is_not_null()
        & (df["age"] != -1000)
        & df["age"].is_not_null()
        & (df["sex"] != -1000)
        & df["sex"].is_not_null()
        & df["site"].is_not_null()
        & df["meanFD"].is_not_null()
        & (df["visit"] == 1)
        & (df["session"] == 1)
    )


@app.cell
def _(
    harmonized_bmb_fc_matrices_hc_mdd_df,
    old_bmb_cache_dir,
    old_bmb_plot_dir,
    old_bmb_ttest_dir,
    old_metric_dict,
):
    old_bmb_metrics_dict = old_calculate_metrics(
        df=bmb_filter(harmonized_bmb_fc_matrices_hc_mdd_df),
        metric_dict=old_metric_dict,
        alpha_threshold=0.05,
        plot_dir=old_bmb_plot_dir,
        ttest_dir=old_bmb_ttest_dir,
        cache_dir=old_bmb_cache_dir,
    )

    old_bmb_results = old_bmb_metrics_dict["results"]
    old_bmb_ui = old_bmb_metrics_dict["ui"]
    return (old_bmb_results,)


@app.cell
def _(
    bmb_cache_dir,
    bmb_plot_dir,
    bmb_ttest_dir,
    harmonized_bmb_fc_matrices_hc_mdd_df,
    metric_dict,
):
    bmb_metrics_dict = calculate_metrics(
        df=bmb_filter(harmonized_bmb_fc_matrices_hc_mdd_df),
        metric_dict=metric_dict,
        alpha_threshold=0.05,
        n_pcs=5,
        plot_dir=bmb_plot_dir,
        ttest_dir=bmb_ttest_dir,
        cache_dir=bmb_cache_dir,
    )

    bmb_results = bmb_metrics_dict["results"]
    bmb_selected_pcs = bmb_metrics_dict["selected_pcs"]
    bmb_ui = bmb_metrics_dict["ui"]
    return bmb_results, bmb_selected_pcs, bmb_ui


@app.cell
def _(bmb_selected_pcs):
    bmb_selected_pcs
    return


@app.cell
def _(bmb_ui):
    bmb_ui
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Analyze the results of the PCA feature selection
    """)
    return


@app.cell
def _(old_bmb_results):
    old_bmb_selected_mdd_pcs = select_mdd_pc(old_bmb_results, ["bdi"])
    old_bmb_selected_mdd_pcs
    return


@app.cell
def _(bmb_results):
    bmb_selected_mdd_pcs = select_mdd_pc(bmb_results, ["bdi"])
    bmb_selected_mdd_pcs
    return


@app.cell
def _(
    bmb_metadata_dir,
    bmb_plot_dir,
    bmb_results,
    coords_mni,
    harmonized_bmb_fc_matrices_hc_mdd_df,
    network_assignments,
):
    bmb_target_pcs_to_plot = [1.0]

    bmb_pc_plots_results = plot_pcs(
        results=bmb_results,
        fc_matrices_df=harmonized_bmb_fc_matrices_hc_mdd_df,
        node_coords=coords_mni,
        pcs_to_plot=bmb_target_pcs_to_plot,
        plot_dir=bmb_plot_dir,
        metadata_dir=bmb_metadata_dir,
        show_legend=True,
        network_assignments=network_assignments,
    )
    return (bmb_pc_plots_results,)


@app.cell
def _(bmb_pc_plots_results):
    mo.vstack(list(bmb_pc_plots_results["ui_elements"].values()), gap=3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### First draft of a conclusion
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The PC extraction gives different results depending on the dataset, but the FC edges contributing to those PCs are largely consistent. The most represented networks (PrefrontalControlA, SomatoMotor, and Subcortical) are shared across datasets, and MDD subjects predominantly show under-connectivity in the selected PCs.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Extraction and harmonization of the regular the FC matrices of the SRPB dataset
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We will start by extracting regular FC matrices extraction, to see if the extraction method is correct and compare them to the harmonized ones
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Assign each time series file to its subject
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We start by creating a new Dataframe with the SRPB metadata and each subject time series file path
    """)
    return


@app.cell
def _():
    srpb_time_series_directory_paths = [
        Path(
            "/home/cbi-biomark03/ayumu/HARP/data/preproc_srpb/time_series/Glasser_filtering1_GSR1_scrubbing1/"
        ),
        Path(
            "/home/cbi-biomark03/ayumu/HARP/data/preproc_uhi_bipolar/time_series/Glasser_filtering1_GSR1_scrubbing1/"
        ),
    ]
    return (srpb_time_series_directory_paths,)


@app.cell
def _(srpb_time_series_directory_paths):
    srpb_time_series_directory_paths
    return


@app.cell
def _(srpb_time_series_directory_paths):
    srpb_time_series_directory_suffixes = {
        srpb_time_series_directory_paths[0]: "_restT1_parcel.csv",
        srpb_time_series_directory_paths[1]: "_BOLD_REST1_AP_parcel.csv",
    }
    return (srpb_time_series_directory_suffixes,)


@app.cell
def _(srpb_time_series_directory_suffixes):
    srpb_time_series_directory_suffixes
    return


@app.cell
def _(srpb_time_series_directory_paths):
    srpb_time_series_file_paths = [
        file
        for directory_path in srpb_time_series_directory_paths
        for file in directory_path.iterdir()
        if file.is_file()
    ]
    return


@app.cell
def _(srpb_time_series_directory_paths, srpb_time_series_directory_suffixes):
    srpb_time_series_separated_dicts_list = [
        {
            path.name.removesuffix(
                srpb_time_series_directory_suffixes[dir_path]
            ): path
            for path in dir_path.iterdir()
            if path.is_file()
            and path.name.endswith(srpb_time_series_directory_suffixes[dir_path])
        }
        for dir_path in srpb_time_series_directory_paths
    ]
    return (srpb_time_series_separated_dicts_list,)


@app.cell
def _(srpb_time_series_separated_dicts_list):
    srpb_combined_dict = {}
    for d in srpb_time_series_separated_dicts_list:
        srpb_combined_dict.update(d)

    srpb_time_series_file_paths_df = pl.DataFrame(
        {
            "sub_id": list(srpb_combined_dict.keys()),
            "time_series_path": [str(p) for p in srpb_combined_dict.values()],
        }
    )
    return (srpb_time_series_file_paths_df,)


@app.cell
def _(srpb_metadata_df, srpb_time_series_file_paths_df):
    srpb_time_series_file_df = srpb_time_series_file_paths_df.join(
        srpb_metadata_df, on="sub_id", how="inner"
    )
    return (srpb_time_series_file_df,)


@app.cell
def _(srpb_time_series_file_df):
    srpb_time_series_file_df.head()
    return


@app.cell
def _(srpb_time_series_file_df):
    srpb_time_series_file_df.height
    return


@app.cell
def _(srpb_metadata_df):
    srpb_metadata_df.height
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Assign each scrub file to its subject
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We also assign each subject its scrub file path
    """)
    return


@app.cell
def _():
    srpb_scrub_directory_paths = [
        Path(
            "/home/cbi-biomark03/ayumu/HARP/data/preproc_srpb/scrub/Glasser_filtering1_GSR1_scrubbing1/"
        ),
        Path(
            "/home/cbi-biomark03/ayumu/HARP/data/preproc_uhi_bipolar/scrub/Glasser_filtering1_GSR1_scrubbing1"
        ),
    ]
    return (srpb_scrub_directory_paths,)


@app.cell
def _(srpb_scrub_directory_paths):
    srpb_scrub_directory_paths
    return


@app.cell
def _(srpb_scrub_directory_paths):
    srpb_scrub_directory_suffixes = {
        srpb_scrub_directory_paths[0]: "_restT1_scrub.csv",
        srpb_scrub_directory_paths[1]: "_BOLD_REST1_AP_scrub.csv",
    }
    return (srpb_scrub_directory_suffixes,)


@app.cell
def _(srpb_scrub_directory_suffixes):
    srpb_scrub_directory_suffixes
    return


@app.cell
def _(srpb_scrub_directory_paths):
    srpb_scrub_file_paths = [
        file
        for directory_path in srpb_scrub_directory_paths
        for file in directory_path.iterdir()
        if file.is_file()
    ]
    return


@app.cell
def _(srpb_scrub_directory_suffixes):
    srpb_scrub_separated_dicts_list = [
        {
            path.name.removesuffix(suffix): path
            for path in dir_path.iterdir()
            if path.is_file() and path.name.endswith(suffix)
        }
        for dir_path, suffix in srpb_scrub_directory_suffixes.items()
    ]
    return (srpb_scrub_separated_dicts_list,)


@app.cell
def _(srpb_scrub_separated_dicts_list):
    srpb_scrub_combined_dict = {}
    for _d in srpb_scrub_separated_dicts_list:
        srpb_scrub_combined_dict.update(_d)

    srpb_scrub_file_paths_df = pl.DataFrame(
        {
            "sub_id": list(srpb_scrub_combined_dict.keys()),
            "scrub_path": [str(p) for p in srpb_scrub_combined_dict.values()],
        }
    )
    return (srpb_scrub_file_paths_df,)


@app.cell
def _(srpb_scrub_file_paths_df):
    srpb_scrub_file_paths_df.head()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Merge the scrub and time_series datasets together
    """)
    return


@app.cell
def _(srpb_scrub_file_paths_df, srpb_time_series_file_df):
    srpb_time_series_scrub_file_df = srpb_scrub_file_paths_df.join(
        srpb_time_series_file_df, on="sub_id", how="inner"
    )
    return (srpb_time_series_scrub_file_df,)


@app.cell
def _(srpb_time_series_scrub_file_df):
    srpb_time_series_scrub_file_df.height
    return


@app.cell
def _(srpb_time_series_scrub_file_df):
    srpb_time_series_scrub_file_df.height
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Calculate each subject regular FC matrix
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We will now define the functions used to extract the FC matrices from the subject's time series and scrub paths
    """)
    return


@app.function
def extract_subject_fc_matrix(
    subject_time_series_path: str,
    subject_scrub_path: str,
    corrcoef_function=np.corrcoef,
    apply_scrubbing=True,
    scrubbing_threshold=0.5,
) -> list:

    # Load the time_series
    time_series_data = pl.read_csv(subject_time_series_path)
    time_series_data = time_series_data.drop(time_series_data.columns[0])

    if apply_scrubbing:
        # Load the scrub file
        scrub_data = pl.read_csv(subject_scrub_path)
        scrub_array = scrub_data["0"].to_numpy()

        scrub_first_col = scrub_data.columns[0]
        scrub_data = scrub_data.drop(scrub_first_col)
        scrub_array = scrub_data.to_numpy().flatten()

        # Compute the mask
        mask = np.ones(len(scrub_array))

        for time_i, fd_i in enumerate(scrub_array):
            if fd_i > scrubbing_threshold:
                j1 = max(0, time_i)
                j2 = min(len(scrub_array), time_i + 2)
                mask[j1:j2] = 0

        # Apply the mask using
        mask_series = pl.Series(mask == 1)
        time_series_data_filtered = time_series_data.filter(mask_series)
    else:
        time_series_data_filtered = time_series_data

    # Convert to numpy for correlation computation
    ts_array = time_series_data_filtered.to_numpy()

    # Compute correlation matrix
    corr_matrix = corrcoef_function(ts_array.T)

    # Lower triangular + Fisher Z-transformation
    lower_tri = np.tril(corr_matrix, -1)
    lower_tri_z = np.arctanh(lower_tri)

    # Extract non-zero values
    connectivity = lower_tri_z[lower_tri_z != 0]

    return connectivity.tolist()


@app.cell
def _():
    extract_subject_fc_matrix
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And extract each FC matrix in parallel
    """)
    return


@app.cell
def _():
    srpb_fuzzy_fc_matrices_output_path = (
        "/home/cbi-biomark/olivier.amacker/fuzzy-fc-matrices/srpb"
    )
    return (srpb_fuzzy_fc_matrices_output_path,)


@app.cell
def _(srpb_fuzzy_fc_matrices_output_path):
    regular_run_name = "regular-matrices"

    srpb_run_fc_matrices_output_dir = (
        f"{srpb_fuzzy_fc_matrices_output_path}/{regular_run_name}/"
    )

    srpb_run_fc_matrices_cache_filename = (
        "Glasser_filtering1_GSR1_fuzzy_fc_matrices.pkl"
    )
    srpb_run_fc_matrices_cache_path = os.path.join(
        srpb_run_fc_matrices_output_dir, srpb_run_fc_matrices_cache_filename
    )
    return (
        regular_run_name,
        srpb_run_fc_matrices_cache_path,
        srpb_run_fc_matrices_output_dir,
    )


@app.cell
def _(srpb_time_series_scrub_file_df):
    srpb_ts_paths = srpb_time_series_scrub_file_df["time_series_path"].to_list()
    srpb_scrub_paths = srpb_time_series_scrub_file_df["scrub_path"].to_list()
    return srpb_scrub_paths, srpb_ts_paths


@app.function
def process_subject(
    subject_time_series_path: str,
    subject_scrub_path: str,
    corrcoef_function=np.corrcoef,
    apply_scrubbing=True,
    scrubbing_threshold=0.5,
):
    return extract_subject_fc_matrix(
        subject_time_series_path=subject_time_series_path,
        subject_scrub_path=subject_scrub_path,
        corrcoef_function=corrcoef_function,
        apply_scrubbing=apply_scrubbing,
        scrubbing_threshold=scrubbing_threshold,
    )


@app.function
def extract_regular_fc_matrices(
    run_fc_matrices_cache_path,
    run_fc_matrices_output_dir,
    time_series_scrub_file_df,
    ts_paths,
    scrub_paths,
):
    if os.path.exists(
        run_fc_matrices_cache_path,
    ):
        print(
            f"Loading cached results for extracted FC matrices from {run_fc_matrices_cache_path}"
        )

        with open(run_fc_matrices_cache_path, "rb") as f:
            results = pickle.load(f)

        # Reconstruct the DataFrame from the cached list
        extracted_fc_matrices_df = time_series_scrub_file_df.with_columns(
            pl.Series(
                name="fc_matrix", values=results, dtype=pl.List(pl.Float64)
            )
        ).select(
            "sub_id",
            "time_series_path",
            "fc_matrix",
            pl.exclude("sub_id", "time_series_path", "fc_matrix"),
        )

    else:
        print(f"Computing and caching extracted FC matrices...")

        results = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_subject, ts, sc)
                for ts, sc in zip(ts_paths, scrub_paths)
            ]

            for future in tqdm(futures, desc="Processing subjects"):
                results.append(future.result())

        extracted_fc_matrices_df = time_series_scrub_file_df.with_columns(
            pl.Series(
                name="fc_matrix", values=results, dtype=pl.List(pl.Float64)
            )
        ).select(
            "sub_id",
            "time_series_path",
            "fc_matrix",
            pl.exclude("sub_id", "time_series_path", "fc_matrix"),
        )

        # Ensure the output directory exists before saving
        os.makedirs(run_fc_matrices_output_dir, exist_ok=True)

        # Save to cache with the correct extension
        with open(run_fc_matrices_cache_path, "wb") as _f:
            pickle.dump(results, _f)

    return extracted_fc_matrices_df


@app.cell
def _(
    srpb_run_fc_matrices_cache_path,
    srpb_run_fc_matrices_output_dir,
    srpb_scrub_paths,
    srpb_time_series_scrub_file_df,
    srpb_ts_paths,
):
    srpb_extracted_fc_matrices_df = extract_regular_fc_matrices(
        srpb_run_fc_matrices_cache_path,
        srpb_run_fc_matrices_output_dir,
        srpb_time_series_scrub_file_df,
        srpb_ts_paths,
        srpb_scrub_paths,
    )
    return (srpb_extracted_fc_matrices_df,)


@app.cell
def _(srpb_extracted_fc_matrices_df):
    srpb_extracted_fc_matrices_df.head(1)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Comparison of the extracted FC matrices and the harmonized ones
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Please note that this plotting code was generated by Qwen3.7-Plus
    """)
    return


@app.function
def compare_fc_matrices(
    unharmonized_df: pl.DataFrame,
    unharmonized_col: str,
    harmonized_df: pl.DataFrame,
    harmonized_col: str,
    out_dir: str,
    sub_id_col: str = "sub_id",
) -> tuple[dict, pl.DataFrame]:

    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "fc_comparison_results.json")
    plot_path = os.path.join(out_dir, "fc_comparison_histogram.png")

    # ------------------------------------------------------------------
    # 1. SAFE JOIN (avoid suffix ambiguity entirely)
    # ------------------------------------------------------------------
    joined_df = unharmonized_df.join(
        harmonized_df.rename({harmonized_col: "harm"}),
        on=sub_id_col,
        how="left",
    )

    # ------------------------------------------------------------------
    # 2. EXTRACT DATA
    # ------------------------------------------------------------------
    unharmonized_list = joined_df[unharmonized_col].to_list()
    harmonized_list = joined_df["harm"].to_list()

    # ------------------------------------------------------------------
    # 3. COMPUTE METRICS
    # ------------------------------------------------------------------
    subject_correlations = []
    subject_maes = []

    for i in range(len(unharmonized_list)):
        my_vec_flat = unharmonized_list[i]
        ref_vec_flat = harmonized_list[i]

        # handle missing data
        if my_vec_flat is None or ref_vec_flat is None:
            subject_correlations.append(None)
            subject_maes.append(None)
            continue

        my_vec = np.asarray(my_vec_flat, dtype=np.float64)
        ref_vec = np.asarray(ref_vec_flat, dtype=np.float64)

        # length mismatch
        if len(my_vec) != len(ref_vec):
            subject_correlations.append(None)
            subject_maes.append(None)
            continue

        # remove invalid values
        mask = np.isfinite(my_vec) & np.isfinite(ref_vec)

        if not np.any(mask):
            subject_correlations.append(None)
            subject_maes.append(None)
            continue

        my_vec = my_vec[mask]
        ref_vec = ref_vec[mask]

        # correlation (safe)
        if len(my_vec) < 2 or np.std(my_vec) == 0 or np.std(ref_vec) == 0:
            r = np.nan
        else:
            try:
                r, _ = stats.pearsonr(my_vec, ref_vec)
            except Exception:
                r = np.nan

        subject_correlations.append(r)

        # MAE
        mae = np.mean(np.abs(my_vec - ref_vec))
        subject_maes.append(mae)

    # ------------------------------------------------------------------
    # 4. CLEAN AGGREGATION
    # ------------------------------------------------------------------
    valid_corrs = np.asarray(
        [c for c in subject_correlations if c is not None and np.isfinite(c)],
        dtype=np.float64,
    )

    valid_maes = np.asarray(
        [m for m in subject_maes if m is not None and np.isfinite(m)],
        dtype=np.float64,
    )

    results = {
        "subject_correlations": subject_correlations,
        "subject_maes": subject_maes,
        "avg_correlation": float(np.mean(valid_corrs))
        if valid_corrs.size
        else float("nan"),
        "avg_mae": float(np.mean(valid_maes))
        if valid_maes.size
        else float("nan"),
        "n_subjects": int(valid_corrs.size),
    }

    # ------------------------------------------------------------------
    # 5. ROBUST HISTOGRAM
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    if valid_corrs.size > 0:
        # handle degenerate case (your actual issue)
        if np.ptp(valid_corrs) < 1e-12:
            ax.hist(
                valid_corrs,
                bins=1,
                color="skyblue",
                edgecolor="black",
                alpha=0.8,
            )
        else:
            ax.hist(
                valid_corrs,
                bins="auto",
                color="skyblue",
                edgecolor="black",
                alpha=0.8,
            )

        ax.set_title(
            "Distribution of Per-Subject Pairwise Correlations\n(Unharmonized vs Harmonized)",
            fontsize=12,
        )
        ax.set_xlabel("Pearson Correlation (r)")
        ax.set_ylabel("Number of Subjects")

        mean_corr = np.mean(valid_corrs)
        median_corr = np.median(valid_corrs)

        ax.axvline(
            mean_corr,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_corr:.4f}",
        )
        ax.axvline(
            median_corr,
            color="green",
            linestyle="-.",
            linewidth=2,
            label=f"Median: {median_corr:.4f}",
        )

        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        ax.text(
            0.95,
            0.95,
            f"Mean: {mean_corr:.4f}\nMedian: {median_corr:.4f}\n"
            f"Std: {np.std(valid_corrs):.4f}\nMin: {np.min(valid_corrs):.4f}\nMax: {np.max(valid_corrs):.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    else:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # 6. SAVE RESULTS
    # ------------------------------------------------------------------
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)

    # ------------------------------------------------------------------
    # 7. UI
    # ------------------------------------------------------------------
    comparison_plot = mo.image(src=Path(plot_path).read_bytes())

    avg_corr = results["avg_correlation"]
    avg_mae = results["avg_mae"]

    avg_corr_str = f"{avg_corr:.4f}" if np.isfinite(avg_corr) else "N/A"
    avg_mae_str = f"{avg_mae:.4f}" if np.isfinite(avg_mae) else "N/A"

    median_corr_str = (
        f"{np.median(valid_corrs):.4f}" if valid_corrs.size else "N/A"
    )

    metrics_md = mo.md(f"""
#### FC Matrix Comparison Metrics

| Metric | Value |
|--------|-------|
| **Average Per-Subject Correlation** | {avg_corr_str} |
| **Median Per-Subject Correlation** | {median_corr_str} |
| **Average MAE** | {avg_mae_str} |
| **Number of Subjects** | {results["n_subjects"]} |
""")

    ui = mo.vstack([metrics_md, comparison_plot], gap=2)

    results["ui"] = ui

    return results


@app.cell
def _(regular_run_name):
    srpb_extracted_fc_matrices_correlation_output_dir = f"./res/pca-dim-reduction/srpb/extracted-fc-matrices-correlation/{regular_run_name}"
    return (srpb_extracted_fc_matrices_correlation_output_dir,)


@app.cell
def _(
    harmonized_srpb_fc_matrices_df,
    srpb_extracted_fc_matrices_correlation_output_dir,
    srpb_extracted_fc_matrices_df,
):
    srpb_extracted_fc_matrices_comparison_results = compare_fc_matrices(
        srpb_extracted_fc_matrices_df,
        "fc_matrix",
        harmonized_srpb_fc_matrices_df,
        "harmonized_fc_matrix",
        out_dir=srpb_extracted_fc_matrices_correlation_output_dir,
    )
    return (srpb_extracted_fc_matrices_comparison_results,)


@app.cell
def _(srpb_extracted_fc_matrices_comparison_results):
    srpb_extracted_fc_matrices_comparison_results["ui"]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Harmonization of the regular extracted matrices
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We start by defining the functions we need for the harmonization
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Please note that the following code is heavily inspired by the code provided by Ayumu Yamashita-san
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Function to load a Matlab connectivity file (.mat with HDF5 format)
    """)
    return


@app.function
def load_matlab_np_file(file_path: str) -> np.ndarray:
    with h5py.File(str(file_path), "r") as f:
        dset = f["/X"] if "/X" in f else f["X"]
        return np.array(dset).T


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Function to one hot encode a specific column
    """)
    return


@app.cell
def _(List):
    def get_dummy_values_and_labels(column: pd.Series) -> Tuple[np.ndarray, List]:
        dummies = pd.get_dummies(column, dtype=float)
        return dummies.values, dummies.columns.tolist()

    return (get_dummy_values_and_labels,)


@app.cell
def _(List):
    def stack_np_array_by_common_columns(
        np_array_1: np.ndarray,
        labels_1: List,
        np_array_2: np.ndarray,
        labels_2: List,
    ) -> Tuple[np.ndarray, List]:
        # Create lookup for array 2
        labels_2_dict = {label: i for i, label in enumerate(labels_2)}

        columns = []
        selected_columns = []

        # Iterate through array 1's labels
        for i, label in enumerate(labels_1):
            # Find this label in array 2
            label_idx = labels_2_dict.get(label, None)
            if label_idx is not None:  # Label exists in both
                col_1 = np_array_1[:, [i]]
                col_2 = np_array_2[:, [label_idx]]

                # Stack: array 1 on top, array 2 on bottom
                columns.append(np.vstack([col_1, col_2]))
                selected_columns.append(label)

        if not columns:
            return np.empty((np_array_1.shape[0] + np_array_2.shape[0], 0)), []

        return np.hstack(columns), selected_columns

    return (stack_np_array_by_common_columns,)


@app.function
def weighted_orthogonalize_sampling(
    measurement_dummies, sampling_dummies, weights
):
    """
    Orthogonalize sampling_dummies against measurement_dummies using weighted projection.

    Formula: P = X_M · (X_M^T · W · X_M)^{-1} · X_M^T · W
             X_S_perp = X_S - P · X_S
    """

    W = np.diag(weights)

    # X_M^T · W · X_M
    XtWX = measurement_dummies.T @ W @ measurement_dummies

    # (X_M^T · W · X_M)^{-1}
    XtWX_inv = np.linalg.pinv(XtWX)

    # P = X_M · (X_M^T · W · X_M)^{-1} · X_M^T · W
    P = measurement_dummies @ XtWX_inv @ (measurement_dummies.T @ W)

    # X_S_perp = X_S - P · X_S
    X_S_perp = sampling_dummies - P @ sampling_dummies

    return X_S_perp


@app.function
def mean_zero_row(number_of_rows: int, offset: int, length: int) -> np.ndarray:
    row = np.zeros(number_of_rows, dtype=float)

    if length > 0:
        row[offset : offset + length] = 1.0 / float(length)

    return row


@app.function
# TODO DOUBLE CHECK THE COMMENTS AND UNDERSTANDING OF THIS FUNCTION
def compute_gcv_score(
    dummy_values_with_bias,
    connectivities,
    a_equation,
    feature_dims,
    lambda_eff,
    tau_delta,
):
    """
    Calculates Generalized Cross-Validation (GCV), MSE, and Degrees of Freedom (dof)
    for a Regularized Linear Model with bias constraints.

    This function implements the analytical solution for GCV by reusing pre-computed
    global matrices (H, A, W, Z) rather than solving the full system for every target vector.
    This avoids calculating explicit Lagrange multipliers (y) inside the loop, which is
    computationally efficient for large datasets.
    """

    # --- 1. Extract Dimensions ---
    # n_samples: Total number of observations (rows)
    # n_features: Total number of coefficients (columns)
    n_samples, n_features = dummy_values_with_bias.shape
    n_targets = connectivities.shape[1]

    # Extract block counts from the 'feature_dims' dictionary safely
    count_sub = feature_dims.get("sub", 0) if feature_dims else 0
    count_measured = feature_dims.get("mea", 0) if feature_dims else 0

    # --- 2. Construct the Hessian Matrix (H) ---
    # Mathematical Definition: H = D^T @ D + lambda * I + bias_penalty
    # This matrix represents the curvature of the regularized loss function (r).
    # It combines the data fidelity (D^T @ D) and regularization (lambda * I).
    hessian_matrix = (
        dummy_values_with_bias.T @ dummy_values_with_bias
        + lambda_eff * np.eye(n_features)
    )

    # Apply Measurement Bias Penalty (Centering Constraint)
    # If 'tau_delta' is provided and we have measured blocks, we enforce that
    # the mean of the measured features is zero.
    # This adds a penalty term to H corresponding to the Identity matrix minus the
    # centering projection matrix.
    if count_measured > 0 and tau_delta != 0:
        # Create a Centering Matrix C = I - (1/S) * 1*1^T
        # Penalty term: (1/tau_delta^2) * C
        C = np.eye(count_measured) - np.ones(
            (count_measured, count_measured)
        ) / float(count_measured)
        hessian_matrix[
            count_measured : count_measured + count_measured,
            count_measured : count_measured + count_measured,
        ] += (1.0 / (tau_delta**2)) * C

    # --- 3. Matrix Factorization (Pre-computation) ---
    # We factorize H once to avoid recomputing the inverse for every target column.
    try:
        # H_cho_factor = cho_factor(H)
        # We store the factors directly to avoid tuple overhead in the loop
        h_factor, lower_h = cho_factor(
            hessian_matrix, overwrite_a=False, check_finite=False
        )
        # Helper function: given vector f, returns H^{-1} @ f
        solve_h = lambda f: cho_solve(
            (h_factor, lower_h), f, check_finite=False
        )
    except np.linalg.LinAlgError:
        # Fallback to LU decomposition if Cholesky fails (e.g., if matrix is singular)
        lu_factors_h = lu_factor(hessian_matrix)
        solve_h = lambda f: lu_solve(lu_factors_h, f)

    # --- 4. Pre-compute Global Terms (Z and W) ---
    # We compute these once outside the loop to save time.
    # In the KKT system:
    #   [ H    A^T ] [ x ]   [ -f ]
    #   [ A     0  ] [ y ] = [  0 ]
    #
    # We define Z = H^{-1} @ A^T.
    # Math: H * Z = A^T.
    # This splits into:
    # 1. L * U = A^T     (Compute intermediate U)
    # 2. L^T * Z = U     (Compute final Z)
    # Finally, Z = A^T @ H^{-1} (Note: In code, we solve H*Z=A^T, so Z = H^-1*A^T)
    # Note: a_equation.T corresponds to A^T in the KKT matrix structure above.

    # Calculate Z = H^{-1} @ A^T (invH_AT)
    # This is the cached term used in Step 6 to recover x.
    Z = solve_h(a_equation.T)

    # Calculate W = A @ Z = A @ H^{-1} @ A^T (Schur Complement)
    # W is the matrix we will solve for y (W * y = b)
    W = a_equation @ Z

    # --- 5. Factorize the Schur Complement (W) ---
    # W_cho_factor = cho_factor(W)
    # We factorize W for efficient solving in the loop.
    try:
        w_factor, lower_w = cho_factor(
            W, overwrite_a=False, check_finite=False
        )
        solve_w = lambda b: cho_solve(
            (w_factor, lower_w), b, check_finite=False
        )
    except np.linalg.LinAlgError:
        lu_factors_w = lu_factor(W)
        solve_w = lambda b: lu_solve(lu_factors_w, b)

    # --- 6. Calculate Degrees of Freedom (dof) ---
    # Formula: dof = trace(H^{-1}) - trace(W^{-1} @ T)
    # Where T = A @ M1 @ G (effectively A @ H^-1 @ A^T related terms)
    # Degrees of freedom measure the model complexity (bias-variance tradeoff).
    # T_Matrix corresponds to the term inside the trace of the Schur complement equation.
    T_Matrix = (
        a_equation
        @ solve_h(dummy_values_with_bias.T @ dummy_values_with_bias)
        @ a_equation.T
    )
    dof = np.trace(
        solve_h(dummy_values_with_bias.T @ dummy_values_with_bias)
    ) - np.trace(solve_w(T_Matrix))

    # --- 7. Calculate Residual Sum of Squares (RSS) ---
    # We iterate over each target column to calculate the error.
    # D_T corresponds to dummy_values_with_bias.T

    D_T = dummy_values_with_bias.T
    Total_RSS = 0.0

    for target_idx in range(n_targets):
        # Extract the specific target vector (Y column)
        # t corresponds to the target vector X[:, i] in the original logic
        # In the KKT system, this is the target vector 't' (or 'target' in some notations)
        target_vector = connectivities[:, target_idx]

        # The analytical solution for the estimated coefficients (x) for this target:
        # x = H^{-1} @ ( -D^T @ t ) - Z @ ( W^{-1} @ ( -A @ H^{-1} @ (D^T @ t) ) )
        #
        # Step A: Compute the raw gradient from the data
        # f = -(D^T @ target)
        # This represents the negative gradient term from the data fidelity loss.
        # In the KKT system, the RHS vector is -f.

        f = -(D_T @ target_vector)

        # Step B: Transform gradient through H^{-1}
        # Math: invH_residual = H^{-1} * f
        # This gives us the unconstrained least squares solution component.
        invH_residual = solve_h(f)

        # Step C: Transform through constraint projection (A)
        # Math: constraint_term = -A @ H^{-1} * f
        # This represents the right-hand side term for the equation W * y = b.
        # Note: In the KKT derivation: W * y = -A * H^{-1} * f
        # So b = -A * H^{-1} * f.
        b = -(a_equation @ invH_residual)

        # Step D: Solve for the constraint correction using Schur Complement
        # Math: constraint_correction = W^{-1} @ b
        # This effectively computes the term needed to solve for y implicitly.
        # In the KKT system: y = W^{-1} @ b
        y = solve_w(b)

        # Step E: Recover the primal solution vector x
        # Math: x = -H^{-1} * f - Z @ y
        # In the KKT system:
        #   Hx + A^T y = -f
        #   => Hx = -f - A^T y
        #   => x = H^{-1} @ (-f - A^T y)
        #   => x = -H^{-1} f - H^{-1} A^T y
        #   => x = -invH_residual - Z @ y
        x = -invH_residual - (Z @ y)

        # Step F: Compute prediction error and add to RSS
        # Prediction: D @ x
        # Error: t - D @ x
        RSS_component = target_vector - (dummy_values_with_bias @ x)
        Total_RSS += np.sum(RSS_component**2)

    # --- 8. Compute GCV Score and MSE ---
    # Mean Squared Error: Average of RSS across all targets and samples
    mse = Total_RSS / (n_samples * n_targets)

    # GCV: MSE adjusted by the ratio of dof to n_samples
    # GCV = MSE / ((1 - dof/n)^2)
    # This penalizes models with high degrees of freedom (overfitting).
    dof_ratio = dof / n_samples
    denominator = 1.0 - dof_ratio

    if np.isclose(denominator, 0):
        gcv_score = np.inf
    else:
        gcv_score = mse / (denominator**2)

    return gcv_score, mse, dof


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here is a breackdown of the method to estimate this different bias:
    - Subject
    - Site
    - Sampeling
    - (Protocol)

    We will do the following steps in order:

    1. Extract the TS (Trabeling Subjects) and RS (Regular Subjects) connectivity matrices and metadata
    1. Binary encode the desired bias metadata from both and the RS datasets
    2. Stack the different biases horizontally
    3. Define the bias constraints $A\mathbf{x} = \mathbf{b}$ where $A$ is a mean constraint matrix with entries of $1/N$ (where $N$ is the group size), and $\mathbf{b}$ is the zero vector. This enforces that the sum of means equals zero, forcing all group biases to be centered at zero.
        - $$\mathbf{A}\mathbf{x} = \mathbf{b} \quad \implies \quad \underbrace{\text{[Matrix of } 1/N\text{s]}}_{\mathbf{A}} \times \underbrace{\text{[Bias Coefficients]}}_{\mathbf{x}} = \underbrace{\text{[Target: 0]}}_{\mathbf{b}}$$

    4. Solve the KKT optimization problem

    To solve the constrained optimization problem, we first establish the Karush-Kuhn-Tucker (KKT) system. Our goal is to minimize the regularized objective function:

    $\min_{\mathbf{x}} \quad \frac{1}{2} \mathbf{x}^\top \mathbf{H} \mathbf{x} + \mathbf{f}^\top \mathbf{x} \quad \text{subject to} \quad \mathbf{A} \mathbf{x} = \mathbf{0}$

    Here, $\mathbf{x}$ represents the stacked vector of dummy and bias coefficients,

    $\mathbf{H} = \mathbf{D}\mathbf{M}^\top \mathbf{D}\mathbf{M} + \lambda \mathbf{I}$

    is the symmetric positive definite Hessian,

    $\mathbf{f} = -\mathbf{D}\mathbf{M}^\top \mathbf{y}_{\text{target}}$

    captures the residual terms, and $\mathbf{A}$ encodes the constraints (e.g., forcing the sum of biases to zero).

    By introducing Lagrange multipliers $\mathbf{y}$, we form the Lagrangian

    $\mathcal{L}(\mathbf{x}, \mathbf{y}) = \frac{1}{2} \mathbf{x}^\top \mathbf{H} \mathbf{x} + \mathbf{f}^\top \mathbf{x} + \mathbf{y}^\top \mathbf{A} \mathbf{x}$

    The necessary conditions for optimality yield the following KKT system:

    $$\begin{bmatrix}
    \mathbf{H} & \mathbf{A}^\top \\
    \mathbf{A} & \mathbf{0}
    \end{bmatrix}
    \begin{bmatrix}
    \mathbf{x} \\
    \mathbf{y}
    \end{bmatrix}
    =
    \begin{bmatrix}
    -\mathbf{f} \\
    \mathbf{0}
    \end{bmatrix}$$

    To avoid inverting the large $N \times N$ matrix $\mathbf{H}$ directly, we eliminate $\mathbf{x}$ from the system. From the first block equation, we have

    $\mathbf{H}\mathbf{x} + \mathbf{A}^\top \mathbf{y} = -\mathbf{f}$,

    which implies

    $\mathbf{x} = -\mathbf{H}^{-1}\mathbf{f} - \mathbf{H}^{-1}\mathbf{A}^\top \mathbf{y}$.

    Substituting this into the second equation ($\mathbf{A}\mathbf{x} = \mathbf{0}$) gives us a reduced system involving the Schur complement

    $\mathbf{W} = \mathbf{A}\mathbf{H}^{-1}\mathbf{A}^\top$

    $\mathbf{W} \mathbf{y} = -\mathbf{A} \mathbf{H}^{-1} \mathbf{f}$

    This allows us to solve for the smaller vector $\mathbf{y}$ first. Since $\mathbf{H}$ is positive definite, we compute its Cholesky decomposition

    $\mathbf{H} = \mathbf{L}\mathbf{L}^\top$

    We then efficiently compute the terms

    $\mathbf{z} = \mathbf{H}^{-1}\mathbf{A}^\top$
    and
    $\mathbf{r} = \mathbf{H}^{-1}\mathbf{f}$

    by solving two triangular systems ($\mathbf{L}\mathbf{u} = \mathbf{A}^\top$ and $\mathbf{L}^\top \mathbf{z} = \mathbf{u}$) rather than forming the inverse explicitly.

    Finally, once $\mathbf{y}$ is obtained by solving $\mathbf{W}\mathbf{y} = -\mathbf{A}\mathbf{r}$, we recover the full solution $\mathbf{x}$ using the relation $\mathbf{x} = -\mathbf{r} - \mathbf{z}\mathbf{y}$.
    """)
    return


@app.cell
def _(Dict, get_dummy_values_and_labels, p, stack_np_array_by_common_columns):
    # The default argument values are taken from the provided code
    def estimate_bias(
        output_dir_path: Path,
        rs_connectivity: list,
        dataset: str = "BMB",
        subjects_to_remove_df=pl.DataFrame(
            schema={"participants_id": pl.Int64, "session": pl.Int64}
        ),
        roi: str = "Glasser",
        gsr_flag: bool = True,
        protocol_flag: bool = True,
        permutation_flag: bool = False,
        lambda_value: float = 1.0,
        ortho_flag: bool = True,
        ts_weight: float = 1.0,
        tau_delta: float = 0.1,
    ):

        # Setup the different paths
        base_path = Path("/home/cbi-biomark03/ayumu/HARP/")

        if roi == "Glasser":
            number_of_regions = 446
        elif roi == "HCP_MMP":
            number_of_regions = 379

        # RS means regular subjects
        # TS means traveling subjects

        # Define the necessary paths
        rs_dataset_directory_path = base_path / f"data/preproc_{dataset.lower()}/"
        ts_dataset_directory_path = (
            base_path / f"data/preproc_{dataset.lower()}_ts/"
        )

        output_figure_path = output_dir_path / f"figs/"
        figure_name = "bias_correlation_heatmap.png"

        os.makedirs(output_figure_path, exist_ok=True)

        if permutation_flag == 1:
            output_directory_path = (
                output_dir_path / f"ts_harmonization_{dataset}_perm/"
            )
        else:
            output_directory_path = (
                output_dir_path / f"ts_harmonization_{dataset}_lambda/"
            )

        output_directory_path.mkdir(parents=True, exist_ok=True)

        effective_lambda = float(lambda_value) * 10.0

        outpute_file_path = (
            output_directory_path
            / f"EstimatedBias_{roi}_GSR{int(gsr_flag)}_protocol{int(protocol_flag)}_lambda{effective_lambda}_ortho{int(ortho_flag)}_wts{ts_weight}_tdelta{tau_delta}.mat"
        )

        # If the output file already exist, skip the calculation
        if outpute_file_path.exists():
            print(
                f"The bias estimation output file already exists at: {outpute_file_path}"
            )

            data = loadmat(str(outpute_file_path))

            solution_matrix = data["solution_matrix"]
            dummy_values_with_bias = data["dummy_values_with_bias"]

            raw_labels = data["dummy_labels"]
            dummy_labels_struct = np.array(
                [[str(item[0].flatten()[0])] for item in raw_labels], dtype=object
            )

            raw_fd = data["feature_dims"].flatten()[0]
            feature_dims_struct = {
                "sub": int(raw_fd["sub"].item()),
                "mea": int(raw_fd["mea"].item()),
                "sampling": int(raw_fd["sampling"].item()),
            }

            gcv = float(data["gcv"].item())
            mse = float(data["mse"].item())
            dof = float(data["dof"].item())

            image_path = output_figure_path / figure_name
            if image_path.exists():
                mo.image(src=str(image_path))
            else:
                print(f"Warning: Image file not found at {image_path}")

            return {
                "solution_matrix": solution_matrix,
                "dummy_values_with_bias": dummy_values_with_bias,
                "dummy_labels": dummy_labels_struct,
                "feature_dims": feature_dims_struct,
                "gcv": gcv,
                "mse": mse,
                "dof": dof,
            }

        # Extract connectivity and metadata from the TS dataset

        ts_connectivity_path = (
            ts_dataset_directory_path
            / f"all_data_con_{roi}_GSR{int(gsr_flag)}.mat"
        )
        ts_metadata_path = (
            ts_dataset_directory_path
            / f"all_data_sub_{roi}_GSR{int(gsr_flag)}.csv"
        )

        ts_connectivity = load_matlab_np_file(ts_connectivity_path)

        ts_metadata_dataframe = pd.read_csv(ts_metadata_path)

        # Correct the traveling subjects sites
        ts_metadata_dataframe["Site"] = ts_metadata_dataframe["Site"].replace(
            ["SWS"], "SWA"
        )
        ts_metadata_dataframe["Site"] = ts_metadata_dataframe["Site"].replace(
            ["UOP"], "UOS"
        )

        # Calculate the dummy values
        if dataset == "SRPB":
            ts_subject_dummy_values, ts_subject_dummy_labels = (
                get_dummy_values_and_labels(ts_metadata_dataframe["sub_id"])
            )
        elif dataset == "BMB":
            ts_subject_dummy_values, ts_subject_dummy_labels = (
                get_dummy_values_and_labels(
                    ts_metadata_dataframe["participants_id"]
                )
            )
        else:
            raise ValueError(f"Unexpected dataset value: {dataset}")

        ts_site_dummy_values, ts_site_dummy_labels = get_dummy_values_and_labels(
            ts_metadata_dataframe["Site"]
        )

        if dataset == "BMB":
            ts_protocol_dummy_values, ts_protocol_dummy_labels = (
                get_dummy_values_and_labels(ts_metadata_dataframe["protocol"])
            )

        ts_number_of_subjects = ts_metadata_dataframe.shape[0]

        # Do the same for the RS dataset

        """
        rs_connectivity_path = (
            rs_dataset_directory_path / f"all_data_con_{roi}_GSR{int(gsr_flag)}.mat"
        )
        """
        rs_metadata_path = (
            rs_dataset_directory_path
            / f"all_data_sub_{roi}_GSR{int(gsr_flag)}.csv"
        )

        print(f"{rs_metadata_path=}")

        """
        rs_connectivity = load_matlab_np_file(rs_connectivity_path)
        """
        rs_metadata_dataframe = pd.read_csv(rs_metadata_path)

        # TODO: Check if the 3000 range subjects should be included

        # Remove the unwanted subjects

        if len(subjects_to_remove_df) > 0:
            subjects_to_remove_df = subjects_to_remove_df.to_pandas()
            keys = [
                col
                for col in subjects_to_remove_df.columns
                if col in rs_metadata_dataframe.columns
            ]

            rs_metadata_dataframe = rs_metadata_dataframe.merge(
                subjects_to_remove_df,
                on=keys,
                how="left",
                indicator=True,
            )
            rs_metadata_dataframe = rs_metadata_dataframe[
                rs_metadata_dataframe["_merge"] == "left_only"
            ].drop(columns=["_merge"])

        if dataset == "BMB":
            columns_to_remove = ~np.isfinite(rs_connectivity).all(axis=0)

            if columns_to_remove.any():
                rs_connectivity = rs_connectivity[:, ~columns_to_remove]
                rs_metadata_dataframe = rs_metadata_dataframe.loc[
                    ~columns_to_remove, :
                ].reset_index(drop=True)

            rs_metadata_dataframe = rs_metadata_dataframe.copy()
            rs_metadata_dataframe["site"] = (
                rs_metadata_dataframe["participants_id"]
                .astype(str)
                .str.slice(0, 3)
            )

        rs_number_of_subjects = rs_metadata_dataframe.shape[0]

        # Group the ts and rs connectivities
        connectivities = np.vstack([ts_connectivity.T, rs_connectivity.T])

        # If we want to make a permutation of the connectivities
        if permutation_flag == 1:
            np.random.seed(42)

            permutation_indices = np.random.permutation(connectivities.shape[0])

            connectivities = connectivities[permutation_indices, :]
            print("Realised permutation with seed 42")

        total_number_of_subjects, number_of_connectivity = connectivities.shape

        # Combine the different dummy values and labels
        if dataset == "SRPB":
            sites_to_use = ["COI", "KUT", "UTO", "SWA"]

            rs_site_dummy_values, rs_site_dummy_labels = (
                get_dummy_values_and_labels(rs_metadata_dataframe["site"])
            )

            selected_site_indices = [
                indice
                for indice, label in enumerate(rs_site_dummy_labels)
                if label in sites_to_use
            ]

            rs_site_dummy_values = rs_site_dummy_values[:, selected_site_indices]

            rs_site_dummy_labels = [
                rs_site_dummy_labels[i] for i in selected_site_indices
            ]

        elif dataset == "BMB":
            rs_site_dummy_values, rs_site_dummy_labels = (
                get_dummy_values_and_labels(rs_metadata_dataframe["site"])
            )
            rs_protocol_dummy_values, rs_protocol_dummy_labels = (
                get_dummy_values_and_labels(rs_metadata_dataframe["protocol"])
            )

        else:
            raise ValueError(f"Unexpected dataset value: {dataset}")

        combined_site_dummy_values, combined_site_dummy_labels = (
            stack_np_array_by_common_columns(
                ts_site_dummy_values,
                ts_site_dummy_labels,
                rs_site_dummy_values,
                rs_site_dummy_labels,
            )
        )

        if dataset == "BMB":
            combined_protocol_dummy_values, combined_protocol_dummy_labels = (
                stack_np_array_by_common_columns(
                    ts_protocol_dummy_values,
                    ts_protocol_dummy_labels,
                    rs_protocol_dummy_values,
                    rs_protocol_dummy_labels,
                )
            )
        elif dataset == "SRPB":
            combined_protocol_dummy_values, combined_protocol_dummy_labels = (
                np.empty((ts_number_of_subjects + rs_number_of_subjects, 0)),
                [],
            )
        else:
            raise ValueError(f"Unexpected dataset value: {dataset}")

        combined_subject_dummy_values = np.vstack(
            [
                ts_subject_dummy_values,
                np.zeros(
                    (rs_number_of_subjects, ts_subject_dummy_values.shape[1]),
                    dtype=float,
                ),
            ],
        )

        # Initialize the sampling dummies
        combined_sampling_dummy_values = combined_site_dummy_values.copy()

        if combined_sampling_dummy_values.size > 0:
            combined_sampling_dummy_values[:ts_number_of_subjects, :] = 0.0

        label_site_sampling = [f"{s}_SAMPLING" for s in combined_site_dummy_labels]

        # Orthogonize if the flag is set
        if ortho_flag == 1:
            is_traveling_subject = np.zeros(total_number_of_subjects, dtype=bool)

            is_traveling_subject[:ts_number_of_subjects] = True

            ts_weights = np.ones(total_number_of_subjects, dtype=float)

            ts_weights[is_traveling_subject] = ts_weight

            combined_sampling_dummy_values = weighted_orthogonalize_sampling(
                combined_site_dummy_values,
                combined_sampling_dummy_values,
                ts_weights,
            )

        # Merge the dummies and values based on the protocol flag
        if dataset == "BMB" and protocol_flag == 1:
            dummy_values = (
                np.hstack(
                    [
                        combined_subject_dummy_values,
                        combined_site_dummy_values,
                        combined_sampling_dummy_values,
                        combined_protocol_dummy_values,
                    ]
                )
                if combined_site_dummy_values.size
                else np.hstack(
                    [
                        combined_subject_dummy_values,
                        combined_sampling_dummy_values,
                        combined_protocol_dummy_values,
                    ]
                )
            )

            dummy_labels = (
                list(ts_subject_dummy_labels)
                + list(combined_site_dummy_labels)
                + list(label_site_sampling)
                + list(combined_protocol_dummy_labels)
            )

            feature_dims: Dict[str, int] = {
                "sub": combined_subject_dummy_values.shape[1],
                "mea": combined_site_dummy_values.shape[1],
                "sampling": combined_sampling_dummy_values.shape[1],
                "protocol": combined_protocol_dummy_values.shape[1],
            }
        else:
            dummy_values = (
                np.hstack(
                    [
                        combined_subject_dummy_values,
                        combined_site_dummy_values,
                        combined_sampling_dummy_values,
                    ]
                )
                if combined_site_dummy_values.size
                else np.hstack(
                    [combined_subject_dummy_values, combined_sampling_dummy_values]
                )
            )

            dummy_labels = (
                list(ts_subject_dummy_labels)
                + list(combined_site_dummy_labels)
                + list(label_site_sampling)
            )

            feature_dims: Dict[str, int] = {
                "sub": combined_subject_dummy_values.shape[1],
                "mea": combined_site_dummy_values.shape[1],
                "sampling": combined_sampling_dummy_values.shape[1],
            }

        # Add a bias coefficient (intercept) of 1s to the dummy values
        dummy_values_with_bias = np.hstack(
            [dummy_values, np.ones((total_number_of_subjects, 1), dtype=float)]
        )

        number_of_coefficients = dummy_values_with_bias.shape[1]

        # Changes the 0-1 array into 0-1/m where m is the number of values
        rows = []
        offset = 0

        rows.append(
            mean_zero_row(number_of_coefficients, offset, feature_dims["sub"])
        )
        offset += feature_dims["sub"]
        rows.append(
            mean_zero_row(number_of_coefficients, offset, feature_dims["mea"])
        )
        offset += feature_dims["mea"]
        rows.append(
            mean_zero_row(number_of_coefficients, offset, feature_dims["sampling"])
        )
        offset += feature_dims["sampling"]

        if dataset == "BMB" and protocol_flag == 1:
            rows.append(
                mean_zero_row(
                    number_of_coefficients, offset, feature_dims["protocol"]
                )
            )
            offset += feature_dims["protocol"]

        # Define the different equations
        a_equations = np.vstack(rows) if rows else np.zeros((0, p), dtype=float)
        b_equations = np.zeros(a_equations.shape[0], dtype=float)

        # H = DM' DM + λ I
        hessian_matrix = (
            dummy_values_with_bias.T @ dummy_values_with_bias
            + effective_lambda * np.eye(number_of_coefficients)
        )

        # Add measurment bias constraint ridge regularization
        offset_mea = feature_dims["sub"]
        len_mea = feature_dims["mea"]

        if len_mea > 0:
            # C = I - (1/S)11^T (Centering matrix)
            # Penalty term: (1/tau_delta^2) * C
            C = np.eye(len_mea) - np.ones((len_mea, len_mea)) / float(len_mea)
            if tau_delta != 0:
                hessian_matrix[
                    offset_mea : offset_mea + len_mea,
                    offset_mea : offset_mea + len_mea,
                ] += (1.0 / tau_delta**2) * C

        # Our objective function is: Minimize 0.5 * x^T H x + f^T x  subject to  A x = 0
        # Where:
        #   x = stacked vector of [dummy coefficients; bias coefficients]
        #   y = Lagrange multipliers (enforce the constraint Ax = 0)
        #   H = DM^T @ DM + lambda * I  (plus bias centering penalty)
        #   f = -DM^T @ target  (linear term from data residuals)
        #   A = constraint matrix (e.g., forces sum of biases to 0)
        #
        # We can therefore see this problem as a KKT system:
        #   [ H    A^T ] [ x ]   [ -f ]
        #   [ A     0  ] [ y ] = [  0 ]
        # Where:
        #   - H is the Hessian matrix of our objective function
        #   - A is the constraint matrix (average of bias)
        #   - x is the vector of ALL coefficients (dummies + biases stacked together)
        #   - y is the vector of Lagrange multipliers (enforces the constraint)
        #   - f is the target vector
        #   - r is the Hessian matrix regularization term
        #   - 0 is the constrain target (bias sum = 0)
        #
        # We can calculate H value as follow
        # ~H = DM * DM^T + λ <=> H = ~H + Ridge
        #
        # And therefore obtain this 2 equations
        # Hx + A^Ty = -f
        # and
        # Ax = 0
        #
        # Which gives us:
        # Hx = -f - A^Ty <=> x = H^(-1)(-f - A^Ty) = -H^(-1)f - H^(-1)A^Ty
        #
        # Since we know Ax = 0 we can substitute x
        # -A H^(-1)f - (A H^(-1)A^T)y = 0
        # <=>
        # (A H^(-1)A^T)y = -A H^(-1)f
        #
        # We define W as the Schur complement: W = A * H^-1 * A^T
        # Then we solve: W * y = -A * H^-1 * f
        #
        # We want to compute W efficiently without inversing H
        # To do so,
        # We solve two triangular systems using Cholesky (H = L * L^T):
        # Let Z = H^-1 * A^T. Then H * Z = A^T.
        # This splits into:
        # 1. L * U = A^T     (Compute intermediate U)
        # 2. L^T * Z = U     (Compute final Z)
        # Finally, W = A * Z
        #
        # Once we have H, A, W, and Z, we can find each edge x value:
        # For each edge i, we compute its specific linear term
        # f = -DM^T * X[:, i]
        # By doing the following:
        # 1. Compute H^-1 * f
        # 2. Compute the projected target: -A * H^-1 * f
        # 3. Solve for y: W * y = -A * H^-1 * f
        # 4. Recover x: x = -H^-1 * f - H^-1 * A^T * y

        # Please note that the comments were generated by Qwen3.5-4b

        # 1. Calculate H_cho_factor = cho_factor(H)
        # H = DM^T @ DM + lambda * I + bias_penalty
        H_cho_factor, lower_H = cho_factor(
            hessian_matrix, overwrite_a=False, check_finite=False
        )
        # 2. Calculate Z = H^-1 @ A^T (invH_AT)
        # We solve H * Z = A^T using the Cholesky factors.
        # A here is the constraint matrix (rows = constraints, cols = total variables).
        invH_AT_cache = cho_solve(
            (H_cho_factor, lower_H), a_equations.T, check_finite=False
        )
        # 3. Calculate W = A @ Z = A @ H^-1 @ A^T (Schur Complement)
        # W is the matrix we will solve for y (W * y = b)
        W = a_equations @ invH_AT_cache
        # 4. Factorize W for efficient solving in the loop
        # W_cho_factor = cho_factor(W)
        W_cho_factor, lower_W = cho_factor(
            W, overwrite_a=False, check_finite=False
        )
        # 5. Prepare output storage
        # Rows = total variables (dummy coeffs + bias coeffs)
        # Cols = number of edges
        solution_matrix = np.empty(
            (dummy_values_with_bias.shape[1], number_of_connectivity), dtype=float
        )
        for i in range(number_of_connectivity):
            print(f"Edge {i + 1}/{number_of_connectivity} processed")

            # 1. Extract data for the current edge
            # t corresponds to the target vector X[:, i]
            edge_target = connectivities[:, i]

            # 2. Calculate residual vector f
            # Math: f = -(number_of_connectivity.T @ target)
            # This represents the negative gradient term from the data fidelity loss.
            residual_vector = -(dummy_values_with_bias.T @ edge_target)

            # 3. Solve for H^-1 * f (invH_residual)
            # Math: invH_residual = H^-1 * f
            invH_residual = cho_solve(
                (H_cho_factor, lower_H), residual_vector, check_finite=False
            )

            # 4. Calculate the Right-Hand Side (RHS) for the Schur system
            # Math: RHS = -(A * (H^-1 * f) + constraint_rhs_constant)
            # Note: If the constraint is purely homogeneous (Ax=0), the 'b_equations' term
            # might be zero or represent a bias offset. Assuming b_equations handles the offset.
            rhs_y = -(b_equations + a_equations @ invH_residual)

            # 5. Solve for Lagrange multipliers y
            # Math: W * y = rhs_y  =>  y = W^-1 * rhs_y
            lagrange_multipliers = cho_solve(
                (W_cho_factor, lower_W), rhs_y, check_finite=False
            )

            # 6. Recover the primal solution vector x
            # Math: x = -H^-1 * f - H^-1 * A^T * y
            # We use the cached invH_AT_cache for the second term.
            solution_vector = -invH_residual - (
                invH_AT_cache @ lagrange_multipliers
            )

            # 7. Store the result
            solution_matrix[:, i] = solution_vector

        # Extract the different bias coefficients
        # shape: (num_mea, num_edges)
        mea_coeffs = solution_matrix[
            feature_dims["sub"] : feature_dims["sub"] + feature_dims["mea"], :
        ]

        # shape: (num_sampling, num_edges)
        sampling_coeffs = solution_matrix[
            feature_dims["sub"] + feature_dims["mea"] : feature_dims["sub"]
            + feature_dims["mea"]
            + feature_dims["sampling"],
            :,
        ]

        # Calculate the correlation of the site and sampling coeffs
        corr_matrix = np.corrcoef(mea_coeffs, sampling_coeffs)

        # Create a correlation heatmap
        sns.heatmap(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Bias Analysis Heatmap and GCV and MSE Summary")

        plt.savefig(output_figure_path / figure_name)

        plt.show()

        gcv, mse, dof = compute_gcv_score(
            dummy_values_with_bias,
            connectivities,
            a_equations,
            lambda_eff=effective_lambda,
            tau_delta=tau_delta,
            feature_dims=feature_dims,
        )
        print(f"GCV={gcv:.6g}, MSE={mse:.6g}, df={dof:.3f}")

        mat_obj = {
            "solution_matrix": solution_matrix,
            "dummy_values_with_bias": dummy_values_with_bias,
            "dummy_labels": np.array(dummy_labels, dtype=object).reshape(-1, 1),
            "feature_dims": {
                k: np.array(v, dtype=np.int32) for k, v in feature_dims.items()
            },
            "gcv": gcv,
            "mse": mse,
            "dof": dof,
        }

        savemat(str(outpute_file_path), mat_obj)

        return mat_obj

    return (estimate_bias,)


@app.function
def harmonize_connectivity(
    bias_dictionnary: dict,
    connectivity: np.ndarray,
    subjects_metadata_dataframe: pl.DataFrame,
    output_path: Path,
    subjects_to_remove_df=None,
    dataset: str = "BMB",
    roi: str = "Glasser",
    gsr_flag: bool = True,
    ortho_flag: bool = True,
    prot_flag: bool = True,
    harm_type: str = "ts",
):
    output_path.mkdir(parents=True, exist_ok=True)

    out_mat_file_name = f"all_data_con_{roi}_GSR{int(gsr_flag)}_ortho{int(ortho_flag)}_harmonized.mat"
    subjects_metadata_file_name = f"all_data_sub_{roi}_GSR{int(gsr_flag)}.csv"
    site_column = "site"

    if (output_path / out_mat_file_name).exists():
        print(
            f"The harmonization output file already exists at: {output_path / out_mat_file_name}"
        )
        with h5py.File(output_path / out_mat_file_name, "r") as f:
            return f["X"][:]

    if harm_type != "ts":
        raise ValueError(
            "The harmonization can only be done for the ts harm type"
        )

    if subjects_to_remove_df is not None and subjects_to_remove_df.height > 0:
        subjects_metadata_dataframe_filtered = (
            subjects_metadata_dataframe.join(
                subjects_to_remove_df,
                on=["participants_id", "session"],
                how="anti",
            )
        )
    else:
        subjects_metadata_dataframe_filtered = subjects_metadata_dataframe

    assert (
        subjects_metadata_dataframe_filtered.height == connectivity.shape[0]
    ), (
        f"Metadata rows ({subjects_metadata_dataframe_filtered.height}) don't match "
        f"connectivity subjects ({connectivity.shape[0]})"
    )

    connectivity_corrected = connectivity.copy()
    subjects_metadata_dataframe_filtered = (
        subjects_metadata_dataframe_filtered.with_columns(
            pl.lit(0).alias("ts_harmonize")
        )
    )

    bias = bias_dictionnary["solution_matrix"]
    dummy_labels = bias_dictionnary["dummy_labels"]

    dummy_labels_flatten = np.array(dummy_labels).flatten().astype(str)
    dummy_labels_flatten = np.append(dummy_labels_flatten, "const")

    assert len(dummy_labels_flatten) == bias.shape[0], (
        f"Label count ({len(dummy_labels_flatten)}) doesn't match bias rows ({bias.shape[0]})"
    )
    print(f"Available labels: {dummy_labels_flatten}")

    if dataset == "BMB":
        subjects_metadata_dataframe_filtered = (
            subjects_metadata_dataframe_filtered.with_columns(
                pl.col("participants_id").str.slice(0, 3).alias("site")
            )
        )
        sites_to_harmonize = np.unique(
            subjects_metadata_dataframe_filtered[site_column]
        )
    elif dataset == "SRPB":
        sites_to_harmonize = ["SWA", "COI", "UTO", "KUT"]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    for site_i in sites_to_harmonize:
        print(f"{dataset}: Harmonizing in {site_i}")
        use_sub_index = np.where(
            subjects_metadata_dataframe_filtered[site_column] == site_i
        )[0]

        if np.any(dummy_labels_flatten == site_i):
            connectivity_corrected[use_sub_index, :] = (
                connectivity_corrected[use_sub_index, :]
                - bias[dummy_labels_flatten == site_i, :]
            )
            ts_harmonize = (
                subjects_metadata_dataframe_filtered["ts_harmonize"]
                .to_numpy()
                .copy()
            )
            ts_harmonize[use_sub_index] = 1
            subjects_metadata_dataframe_filtered = (
                subjects_metadata_dataframe_filtered.with_columns(
                    pl.Series("ts_harmonize", ts_harmonize)
                )
            )
        else:
            print(f"WARNING: '{site_i}' not found in labels - skipping")

    if dataset == "BMB" and prot_flag:
        for protocol_i in np.unique(
            subjects_metadata_dataframe_filtered["protocol"]
        ):
            print(f"Harmonize {protocol_i}")
            use_sub_index = np.where(
                (
                    subjects_metadata_dataframe_filtered["protocol"]
                    == protocol_i
                )
                & (subjects_metadata_dataframe_filtered["ts_harmonize"] == 1)
            )[0]

            if np.any(dummy_labels_flatten == protocol_i):
                connectivity_corrected[use_sub_index, :] = (
                    connectivity_corrected[use_sub_index, :]
                    - bias[dummy_labels_flatten == protocol_i, :]
                )
            else:
                print(
                    f"WARNING: '{protocol_i}' not found in labels - skipping"
                )

    print("Finished traveling subject harmonization!!")

    with h5py.File(output_path / out_mat_file_name, "w") as f:
        f.create_dataset("X", data=connectivity_corrected)

    metadata_for_csv = subjects_metadata_dataframe_filtered.select(
        pl.col(pl.Utf8),
        pl.col(pl.Int8),
        pl.col(pl.Int16),
        pl.col(pl.Int32),
        pl.col(pl.Int64),
        pl.col(pl.UInt8),
        pl.col(pl.UInt16),
        pl.col(pl.UInt32),
        pl.col(pl.UInt64),
        pl.col(pl.Float32),
        pl.col(pl.Float64),
        pl.col(pl.Boolean),
    )

    metadata_for_csv.write_csv(output_path / subjects_metadata_file_name)

    return connectivity_corrected


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We now want to extract the SRPB bias
    """)
    return


@app.cell
def _():
    srpb_fc_matrices_bias_output_dir = Path(
        f"/home/cbi-biomark/olivier.amacker/bias/srpb"
    )
    return (srpb_fc_matrices_bias_output_dir,)


@app.cell
def _(regular_run_name, srpb_fc_matrices_bias_output_dir):
    srpb_regular_fc_matrices_bias_output_dir = (
        srpb_fc_matrices_bias_output_dir / regular_run_name
    )
    return (srpb_regular_fc_matrices_bias_output_dir,)


@app.cell
def _(
    estimate_bias,
    srpb_extracted_fc_matrices_df,
    srpb_regular_fc_matrices_bias_output_dir,
):
    # TODO: Why is axis 1 needed?
    srpb_bias_dict = estimate_bias(
        output_dir_path=srpb_regular_fc_matrices_bias_output_dir,
        rs_connectivity=np.stack(
            srpb_extracted_fc_matrices_df["fc_matrix"].to_numpy(), axis=1
        ),
        dataset="SRPB",
    )
    return (srpb_bias_dict,)


@app.cell
def _(srpb_bias_dict):
    srpb_bias_dict
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And harmonize the extracted FC matrices
    """)
    return


@app.cell
def _():
    srpb_harmonized_fc_matrices_output_dir = Path(
        f"/home/cbi-biomark/olivier.amacker/harmonized-fc-matrices/srpb"
    )
    return (srpb_harmonized_fc_matrices_output_dir,)


@app.cell
def _(regular_run_name, srpb_harmonized_fc_matrices_output_dir):
    srpb_harmonized_regular_fc_matrices_output_dir = (
        srpb_harmonized_fc_matrices_output_dir / regular_run_name
    )
    return (srpb_harmonized_regular_fc_matrices_output_dir,)


@app.cell
def _(
    srpb_bias_dict,
    srpb_extracted_fc_matrices_df,
    srpb_harmonized_regular_fc_matrices_output_dir,
):
    srpb_extracted_harmonized_fc_matrices = harmonize_connectivity(
        bias_dictionnary=srpb_bias_dict,
        connectivity=np.stack(srpb_extracted_fc_matrices_df["fc_matrix"], axis=0),
        subjects_metadata_dataframe=srpb_extracted_fc_matrices_df,
        dataset="SRPB",
        output_path=srpb_harmonized_regular_fc_matrices_output_dir,
    )
    return (srpb_extracted_harmonized_fc_matrices,)


@app.cell
def _(srpb_extracted_fc_matrices_df, srpb_extracted_harmonized_fc_matrices):
    srpb_extracted_harmonized_fc_matrices_df = (
        srpb_extracted_fc_matrices_df.with_columns(
            pl.Series(
                name="harmonized_fc_matrix",
                values=srpb_extracted_harmonized_fc_matrices,
            )
        ).select(
            "sub_id",
            "time_series_path",
            "fc_matrix",
            "harmonized_fc_matrix",
            pl.exclude(
                "sub_id", "time_series_path", "fc_matrix", "harmonized_fc_matrix"
            ),
        )
    )
    return (srpb_extracted_harmonized_fc_matrices_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We now want to compare the previously obtained harmonized FC matrices and the provided ones
    """)
    return


@app.cell
def _(
    harmonized_srpb_fc_matrices_df,
    srpb_extracted_harmonized_fc_matrices_df,
):
    (
        srpb_extracted_harmonized_fc_matrices_df["harmonized_fc_matrix"].to_numpy()
        - harmonized_srpb_fc_matrices_df["harmonized_fc_matrix"].to_numpy()
    ).sum() / len(
        harmonized_srpb_fc_matrices_df["harmonized_fc_matrix"].to_numpy()
    )
    return


@app.cell
def _():
    srpb_harmonized_fc_matrices_correlation_output_dir = f"./res/pca-dim-reduction/srpb/extracted-harmonized-fc-matrices-correlation"
    return (srpb_harmonized_fc_matrices_correlation_output_dir,)


@app.cell
def _(regular_run_name, srpb_harmonized_fc_matrices_correlation_output_dir):
    srpb_harmonized_regular_fc_matrices_correlation_output_dir = (
        srpb_harmonized_fc_matrices_correlation_output_dir + f"/{regular_run_name}"
    )
    return (srpb_harmonized_regular_fc_matrices_correlation_output_dir,)


@app.cell
def _(
    harmonized_srpb_fc_matrices_df,
    srpb_extracted_harmonized_fc_matrices_df,
    srpb_harmonized_regular_fc_matrices_correlation_output_dir,
):
    srpb_extracted_harmonized_fc_matrices_comparison_results = compare_fc_matrices(
        srpb_extracted_harmonized_fc_matrices_df,
        "harmonized_fc_matrix",
        harmonized_srpb_fc_matrices_df,
        "harmonized_fc_matrix",
        out_dir=srpb_harmonized_regular_fc_matrices_correlation_output_dir,
    )
    return (srpb_extracted_harmonized_fc_matrices_comparison_results,)


@app.cell
def _(srpb_extracted_harmonized_fc_matrices_comparison_results):
    srpb_extracted_harmonized_fc_matrices_comparison_results["ui"]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Extraction and harmonization of the SRPB perturbated FC matrices
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Extraction of the fuzzy FC matrices
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We start by defining  the functions needed for the fuzzy extraction
    """)
    return


@app.cell
def _():
    fuzzy_container_image = (
        "verificarlo/fuzzy:v2.0.0-lapack-python3.8.5-numpy-scipy-sklearn"
    )
    return (fuzzy_container_image,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Please note that this function was written by Qwen3.7-Plus
    """)
    return


@app.function
def fuzzy_np_corrcoef(x, container_name: str, precision_binary64=53):
    """
    Run np.corrcoef with fuzzy perturbations inside an already running Docker container.
    """
    import io
    import numpy as np
    import subprocess

    # 1. Serialize the input array to raw bytes
    input_buffer = io.BytesIO()
    np.save(input_buffer, x)
    input_bytes = input_buffer.getvalue()

    # 2. Script to run inside the container
    py_script = """
import sys
import io
import numpy as np

try:
    input_bytes = sys.stdin.buffer.read()
    x = np.load(io.BytesIO(input_bytes))
    res = np.corrcoef(x)

    output_buffer = io.BytesIO()
    np.save(output_buffer, res)
    sys.stdout.buffer.write(output_buffer.getvalue())

except Exception as e:
    sys.stderr.write(f"Error inside container: {str(e)}\\n")
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""

    # 3. Execute in Docker with error catching
    try:
        result = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                container_name,
                "python3",
                "-c",
                py_script,
            ],
            input=input_bytes,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        # THIS IS THE MAGIC: Print what actually failed inside the container
        print(f"\n{'=' * 20} ERROR IN CONTAINER {container_name} {'=' * 20}")
        print("STDERR OUTPUT:")
        print(e.stderr.decode("utf-8", errors="ignore"))
        print("=" * 70 + "\n")
        raise

    # 4. Deserialize the output
    output_buffer = io.BytesIO(result.stdout)
    return np.load(output_buffer)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Please note that this function was written by Qwen3.7-Plus but based on the function to run the extraction I wrote myself.
    """)
    return


@app.function
def run_fuzzy_extraction_runs(
    num_fuzzy_run: int,
    container_name: str,
    container_image: str,
    time_series_scrub_file_df: pl.Dataframe,
    ts_paths: list,
    scrub_paths: list,
    fuzzy_fc_matrices_output_path: str,
):
    all_dataframes = []

    for run_number in range(num_fuzzy_run):
        run_fc_matrices_output_dir = (
            f"{fuzzy_fc_matrices_output_path}/run-{run_number}/"
        )

        run_fc_matrices_cache_filename = (
            "Glasser_filtering1_GSR1_fuzzy_fc_matrices.pkl"
        )

        run_fc_matrices_cache_path = os.path.join(
            run_fc_matrices_output_dir, run_fc_matrices_cache_filename
        )

        if os.path.exists(
            run_fc_matrices_cache_path,
        ):
            print(
                f"Loading cached results for extracted FC matrices run {run_number} from {run_fc_matrices_cache_path}"
            )

            with open(run_fc_matrices_cache_path, "rb") as f:
                results = pickle.load(f)

                # Reconstruct the DataFrame from the cached list
                extracted_fc_matrices_df = (
                    time_series_scrub_file_df.with_columns(
                        pl.Series(
                            name="fc_matrix",
                            values=results,
                            dtype=pl.List(pl.Float64),
                        )
                    ).select(
                        "sub_id",
                        "time_series_path",
                        "fc_matrix",
                        pl.exclude("sub_id", "time_series_path", "fc_matrix"),
                    )
                )

                all_dataframes.append(extracted_fc_matrices_df)

        else:
            run_container_name = f"{container_name}-run-{run_number}"

            try:
                print(
                    f"Computing and caching extracted FC matrices for run {run_number}"
                )

                print(
                    f"Starting Docker container {run_container_name} with image {container_image}"
                )
                # Force remove just in case a zombie container exists from a previous crash
                subprocess.run(
                    ["docker", "rm", "-f", run_container_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                subprocess.run(
                    [
                        "docker",
                        "run",
                        "-d",
                        "--name",
                        run_container_name,
                        "--entrypoint",
                        "sleep",
                        container_image,
                        "infinity",
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                fuzzy_func = partial(
                    fuzzy_np_corrcoef, container_name=run_container_name
                )

                results = []
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(
                            process_subject,
                            ts,
                            sc,
                            fuzzy_func,
                        )
                        for ts, sc in zip(ts_paths, scrub_paths)
                    ]

                    for future in tqdm(
                        futures,
                        desc="Processing subjects",
                    ):
                        results.append(future.result())

                fuzzy_fc_matrices_df = time_series_scrub_file_df.with_columns(
                    pl.Series(
                        name="fc_matrix",
                        values=results,
                        dtype=pl.List(pl.Float64),
                    )
                ).select(
                    "sub_id",
                    "time_series_path",
                    "fc_matrix",
                    pl.exclude("sub_id", "time_series_path", "fc_matrix"),
                )

                all_dataframes.append(fuzzy_fc_matrices_df)

                # Ensure the output directory exists before saving
                os.makedirs(run_fc_matrices_output_dir, exist_ok=True)

                # Save to cache with the correct extension
                with open(run_fc_matrices_cache_path, "wb") as _f:
                    pickle.dump(results, _f)

            finally:
                print(
                    f"Stopping and removing Docker container {run_container_name}"
                )
                subprocess.run(
                    ["docker", "stop", run_container_name],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                subprocess.run(
                    ["docker", "rm", run_container_name],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

    return all_dataframes


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Please note that the following comment was written by Opencode Kimi K2.7-Code
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We use 100 fuzzy runs for the SRPB dataset as a bootstrap-like perturbation of the FC extraction.
    Each run perturbs the floating-point precision of `np.corrcoef` via Verificarlo, producing slightly
    different correlation matrices. We chose 100 runs as a reasonable trade-off between statistical
    power (for consensus/robustness analysis) and computation time.
    """)
    return


@app.cell
def _():
    num_fuzzy_srpb_run = 100
    return (num_fuzzy_srpb_run,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And extract the fuzzy FC matrices
    """)
    return


@app.cell
def _(
    fuzzy_container_image,
    num_fuzzy_srpb_run,
    srpb_fuzzy_fc_matrices_output_path,
    srpb_scrub_paths,
    srpb_time_series_scrub_file_df,
    srpb_ts_paths,
):
    srpb_fuzzy_extracted_fc_matrices_df_list = run_fuzzy_extraction_runs(
        num_fuzzy_srpb_run,
        container_name="fuzzy-container",
        container_image=fuzzy_container_image,
        time_series_scrub_file_df=srpb_time_series_scrub_file_df,
        ts_paths=srpb_ts_paths,
        scrub_paths=srpb_scrub_paths,
        fuzzy_fc_matrices_output_path=srpb_fuzzy_fc_matrices_output_path,
    )
    return (srpb_fuzzy_extracted_fc_matrices_df_list,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Harmonization of the SRPB fuzzy FC matrices
    """)
    return


@app.cell
def _(
    estimate_bias,
    srpb_extracted_fc_matrices_df,
    srpb_fc_matrices_bias_output_dir,
    srpb_fuzzy_extracted_fc_matrices_df_list,
    srpb_harmonized_fc_matrices_output_dir,
):
    srpb_fuzzy_extracted_harmonized_fc_matrices_df_list = []

    for _run_idx, fuzzy_srpb_extracted_fc_matrices_df in enumerate(
        srpb_fuzzy_extracted_fc_matrices_df_list
    ):
        print(f"Harmonizing run {_run_idx}")
        srpb_fuzzy_extracted_bias_dict = estimate_bias(
            output_dir_path=srpb_fc_matrices_bias_output_dir / f"run-{_run_idx}",
            rs_connectivity=np.stack(
                srpb_extracted_fc_matrices_df["fc_matrix"], axis=1
            ),
            dataset="SRPB",
        )

        srpb_fuzzy_extracted_harmonized_fc_matrices = harmonize_connectivity(
            bias_dictionnary=srpb_fuzzy_extracted_bias_dict,
            connectivity=np.stack(
                fuzzy_srpb_extracted_fc_matrices_df["fc_matrix"].to_numpy(), axis=0
            ),
            subjects_metadata_dataframe=fuzzy_srpb_extracted_fc_matrices_df,
            dataset="SRPB",
            output_path=srpb_harmonized_fc_matrices_output_dir / f"run-{_run_idx}",
        )

        srpb_fuzzy_extracted_harmonized_fc_matrices_df = (
            fuzzy_srpb_extracted_fc_matrices_df.with_columns(
                pl.Series(
                    name="harmonized_fc_matrix",
                    values=srpb_fuzzy_extracted_harmonized_fc_matrices,
                )
            ).select(
                "sub_id",
                "time_series_path",
                "fc_matrix",
                "harmonized_fc_matrix",
                pl.exclude(
                    "sub_id",
                    "time_series_path",
                    "fc_matrix",
                    "harmonized_fc_matrix",
                ),
            )
        )

        srpb_fuzzy_extracted_harmonized_fc_matrices_df_list.append(
            srpb_fuzzy_extracted_harmonized_fc_matrices_df
        )
    return (srpb_fuzzy_extracted_harmonized_fc_matrices_df_list,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Comparison of the SRPB PCA features extraction with regular and perturbated FC matrices
    """)
    return


@app.cell
def _():
    old_srpb_fuzzy_plot_dir = (
        "./res/pca-dim-reduction/srpb/fuzzy-features-extraction/old/plots"
    )
    old_srpb_fuzzy_ttest_dir = (
        "./res/pca-dim-reduction/srpb/fuzzy-features-extraction/old/t-tests"
    )
    old_srpb_fuzzy_cache_dir = (
        "./res/pca-dim-reduction/srpb/fuzzy-features-extraction/old/cache"
    )
    old_srpb_fuzzy_metadata_dir = (
        "./res/pca-dim-reduction/srpb/fuzzy-features-extraction/old/metadatas"
    )
    return (
        old_srpb_fuzzy_cache_dir,
        old_srpb_fuzzy_plot_dir,
        old_srpb_fuzzy_ttest_dir,
    )


@app.cell
def _():
    srpb_fuzzy_plot_dir = (
        "./res/pca-dim-reduction/srpb/fuzzy-features-extraction/plots"
    )
    srpb_fuzzy_ttest_dir = (
        "./res/pca-dim-reduction/srpb/fuzzy-features-extraction/t-tests"
    )
    srpb_fuzzy_cache_dir = (
        "./res/pca-dim-reduction/srpb/fuzzy-features-extraction/cache"
    )
    srpb_fuzzy_metadata_dir = (
        "./res/pca-dim-reduction/srpb/fuzzy-features-extraction/metadatas"
    )
    return (
        srpb_fuzzy_cache_dir,
        srpb_fuzzy_metadata_dir,
        srpb_fuzzy_plot_dir,
        srpb_fuzzy_ttest_dir,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Filter the srpb_fuzzy_extracted_harmonized_fc_matrices_df_list to only keep the MDD and HC subjects
    """)
    return


@app.cell
def _(srpb_fuzzy_extracted_harmonized_fc_matrices_df_list):
    srpb_fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df_list = []
    for (
        _run_idx,
        _srpb_fuzzy_extracted_harmonized_fc_matrices_df,
    ) in enumerate(srpb_fuzzy_extracted_harmonized_fc_matrices_df_list):
        srpb_fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df_list.append(
            _srpb_fuzzy_extracted_harmonized_fc_matrices_df.filter(
                pl.col("diag").is_in([0, 2])
            )
        )
    return (srpb_fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df_list,)


@app.cell
def _(
    old_metric_dict,
    old_srpb_fuzzy_cache_dir,
    old_srpb_fuzzy_plot_dir,
    old_srpb_fuzzy_ttest_dir,
    srpb_fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df_list,
):
    old_srpb_fuzzy_metrics_results_list = []
    old_srpb_fuzzy_selected_pcs = []
    old_srpb_fuzzy_metrics_ui_list = []


    for (
        _run_idx,
        _srpb_fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df,
    ) in enumerate(srpb_fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df_list):
        print(f"Calculating metris for run {_run_idx}")

        old_srpb_fuzy_metrics_dict = old_calculate_metrics(
            df=srpb_filter(_srpb_fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df),
            metric_dict=old_metric_dict,
            alpha_threshold=0.05,
            plot_dir=old_srpb_fuzzy_plot_dir + f"/run-{_run_idx}/",
            ttest_dir=old_srpb_fuzzy_ttest_dir + f"/run-{_run_idx}/",
            cache_dir=old_srpb_fuzzy_cache_dir + f"/run-{_run_idx}/",
        )

        old_srpb_fuzzy_metrics_results_list.append(old_srpb_fuzy_metrics_dict["results"])
        old_srpb_fuzzy_metrics_ui_list.append(old_srpb_fuzy_metrics_dict["ui"])
    return (old_srpb_fuzzy_metrics_results_list,)


@app.cell
def _(
    old_metric_dict,
    srpb_fuzzy_cache_dir,
    srpb_fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df_list,
    srpb_fuzzy_plot_dir,
    srpb_fuzzy_ttest_dir,
):
    srpb_fuzzy_metrics_results_list = []
    srpb_fuzzy_selected_pcs = []
    srpb_fuzzy_metrics_ui_list = []


    for (
        _run_idx,
        srpb_fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df,
    ) in enumerate(srpb_fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df_list):
        print(f"Calculating metris for run {_run_idx}")

        srpb_fuzy_metrics_dict = calculate_metrics(
            df=srpb_filter(srpb_fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df),
            metric_dict=old_metric_dict,
            alpha_threshold=0.05,
            n_pcs=5,
            plot_dir=srpb_fuzzy_plot_dir + f"/run-{_run_idx}/",
            ttest_dir=srpb_fuzzy_ttest_dir + f"/run-{_run_idx}/",
            cache_dir=srpb_fuzzy_cache_dir + f"/run-{_run_idx}/",
        )

        srpb_fuzzy_metrics_results_list.append(srpb_fuzy_metrics_dict["results"])
        srpb_fuzzy_selected_pcs.append(srpb_fuzy_metrics_dict["selected_pcs"])
        srpb_fuzzy_metrics_ui_list.append(srpb_fuzy_metrics_dict["ui"])
    return srpb_fuzzy_metrics_results_list, srpb_fuzzy_metrics_ui_list


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Comparison of the Fuzzy and Regular SRPB PC metrics graphs
    """)
    return


@app.cell
def _(srpb_fuzzy_metrics_ui_list, srpb_ui):
    srpb_pc_metrics_comparison = mo.vstack(
        [
            mo.vstack(
                [
                    mo.md(
                        f"### Comparison of the Fuzzy run {_run_idx} vs Regular PC metrics graphs"
                    ),
                    mo.hstack(
                        [
                            _srpb_fuzzy_metrics_ui_list,
                            srpb_ui,
                        ],
                        gap=2,
                    ),
                ]
            )
            for _run_idx, _srpb_fuzzy_metrics_ui_list in enumerate(
                srpb_fuzzy_metrics_ui_list
            )
        ],
        gap=3,
    )

    srpb_pc_metrics_comparison
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Comparison of the SRPB Fuzzy and Regular PC metrics results
    """)
    return


@app.cell
def _(old_srpb_fuzzy_metrics_results_list):
    old_srpb_fuzzy_selected_mdd_pcs_list = list(
        map(
            lambda x: select_mdd_pc(x, ["bdi"]),
            old_srpb_fuzzy_metrics_results_list,
        )
    )

    old_srpb_fuzzy_selected_mdd_pcs_list
    return


@app.cell
def _(srpb_fuzzy_metrics_results_list):
    srpb_fuzzy_top_pc = list(
        map(lambda x: x["diag"]["selected_pcs_with_scores"][0][0], srpb_fuzzy_metrics_results_list)
    )
    return (srpb_fuzzy_top_pc,)


@app.cell
def _(srpb_fuzzy_top_pc):
    set(srpb_fuzzy_top_pc)
    return


@app.cell
def _(
    coords_mni,
    network_assignments,
    srpb_fuzzy_extracted_harmonized_fc_matrices_df_list,
    srpb_fuzzy_metadata_dir,
    srpb_fuzzy_metrics_results_list,
    srpb_fuzzy_plot_dir,
):
    srpb_fuzzy_target_pcs_to_plot = [1.0]

    srpb_fuzzy_pc_plots_result_list = [
        plot_pcs(
            results=x,
            fc_matrices_df=srpb_fuzzy_extracted_harmonized_fc_matrices_df_list[
                _run_idx
            ],
            node_coords=coords_mni,
            pcs_to_plot=srpb_fuzzy_target_pcs_to_plot,
            plot_dir=srpb_fuzzy_plot_dir + f"/run-{_run_idx}/",
            metadata_dir=srpb_fuzzy_metadata_dir + f"/run-{_run_idx}/",
            show_legend=True,
            network_assignments=network_assignments,
        )
        for _run_idx, x in enumerate(srpb_fuzzy_metrics_results_list)
    ]
    return srpb_fuzzy_pc_plots_result_list, srpb_fuzzy_target_pcs_to_plot


@app.cell
def _(srpb_fuzzy_pc_plots_result_list, srpb_pc_plots_results):
    srpb_pc_plots_result_comparison = mo.vstack(
        [
            mo.vstack(
                [
                    mo.md(
                        f"### Comparison of the Fuzzy run {_run_idx} vs Regular PC plots"
                    ),
                    mo.hstack(
                        [
                            srpb_fuzzy_pc_plots_result["ui_elements"][1.0],
                            srpb_pc_plots_results["ui_elements"][1.0],
                        ]
                    ),
                ]
            )
            for _run_idx, srpb_fuzzy_pc_plots_result in enumerate(
                srpb_fuzzy_pc_plots_result_list
            )
        ],
        gap=3,
    )

    srpb_pc_plots_result_comparison
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Define a function to see the different PCs across the runs
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Please note that this code was written by Qwen3.7-Plus and needs to be double-checked
    """)
    return


@app.function
def plot_pcs_consensus_publication_ready(
    results_list: list[dict],
    fc_matrices_df_list: list,  # list of pl.DataFrame
    node_coords: np.ndarray,
    pcs_to_plot: list,
    plot_dir: str,
    metadata_dir: str,
    consensus_thresholds: list[float] = [1.0, 0.75, 0.50, 0.25],
    node_colors=None,
    show_legend=True,
    network_assignments=None,
    skip_existing: bool = True,
) -> dict:
    """
    Generates consensus connectome plots across multiple bootstrap/fuzzy runs.

    If skip_existing=True, checks if plot files already exist FIRST and skips
    data extraction if all plots are cached.

    Parameters
    ----------
    skip_existing : bool
        If True, skip regenerating plots that already exist on disk.
        Checks cache BEFORE any data extraction.

    Returns
    -------
    dict with keys 'results' (nested by PC → threshold) and 'ui'.
    """

    res: dict = {"results": {}}
    ui_elements: list = []
    n_runs = len(results_list)

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    print(f"Processing {n_runs} runs for consensus analysis...")
    print(f"Plot directory: {plot_dir}")
    print(f"Metadata directory: {metadata_dir}")
    print(f"Skip existing: {skip_existing}")

    # ==================================================================
    # 0.  CHECK CACHE FIRST - before any data extraction
    # ==================================================================
    print(
        f"\n[0/3] Checking cache for {len(pcs_to_plot)} PCs × {len(consensus_thresholds)} thresholds..."
    )

    all_cached = True
    cache_status = {}

    if skip_existing:
        for pc_idx in pcs_to_plot:
            cache_status[pc_idx] = {}
            for thresh in consensus_thresholds:
                pct_label = int(thresh * 100)
                safe_thresh = str(pct_label)
                # Use pc_idx + 1 directly (will be float like 2.0) to match existing files
                pc_label = pc_idx + 1
                image_path = os.path.join(
                    plot_dir,
                    f"brain_pc_{pc_label}_consensus_{safe_thresh}pct.png",
                )
                json_path = os.path.join(
                    metadata_dir,
                    f"brain_pc_{pc_label}_consensus_{safe_thresh}pct_metadata.json",
                )

                img_exists = os.path.exists(image_path)
                json_exists = os.path.exists(json_path)
                is_cached = img_exists and json_exists

                cache_status[pc_idx][thresh] = {
                    "cached": is_cached,
                    "image_path": image_path,
                    "json_path": json_path,
                }

                if not is_cached:
                    all_cached = False

        cached_count = sum(
            1
            for pc in cache_status.values()
            for t in pc.values()
            if t["cached"]
        )
        total_count = len(pcs_to_plot) * len(consensus_thresholds)
        print(
            f"  Cache status: {cached_count}/{total_count} plots already exist"
        )

        if all_cached:
            print(
                "  ✓ All plots cached! Loading from cache without data extraction..."
            )
    else:
        print("  skip_existing=False, will regenerate all plots")

    # ── colour map ──────────────────────────────────────────────────
    glasser_network_colors = {
        "Visual": "#00BFC4",
        "SomatoMotor": "#F8766D",
        "DorsalAttention": "#7CAE00",
        "DefaultMode": "#FFC300",
        "Limbic": "#FF61C3",
        "Salience": "#C77CFF",
        "PrefrontalControlA": "#00BAFF",
        "PrefrontalControlB": "#0097A7",
        "Subcortical": "#333333",
        "Cerebellum": "#D2691E",
        "Unknown": "#000000",
    }

    # ==================================================================
    # 1.  PER-RUN EXTRACTION (only if not all cached)
    # ==================================================================
    if not all_cached:
        print(f"\n[1/3] Extracting data from {n_runs} runs...")
        per_run_mean_diff: list[np.ndarray] = []
        per_run_pc_edges: list[dict[float, np.ndarray]] = []
        n_nodes: int | None = None

        for run_idx, (run_result, fc_df) in enumerate(
            zip(results_list, fc_matrices_df_list)
        ):
            if (run_idx + 1) % 10 == 0 or run_idx == 0:
                print(f"  Processing run {run_idx + 1}/{n_runs}...")

            # --- mean difference vector ---
            target_col = fc_df["diag"]
            mask = (
                (target_col != -1000)
                & (target_col.is_not_null())
                & ((target_col == 0) | (target_col == 2))
            )
            filtered = fc_df.filter(mask)
            g0 = filtered.filter(pl.col("diag") == 0)[
                "harmonized_fc_matrix"
            ].to_list()
            g2 = filtered.filter(pl.col("diag") == 2)[
                "harmonized_fc_matrix"
            ].to_list()

            md = np.mean(np.array(g2), axis=0) - np.mean(np.array(g0), axis=0)
            per_run_mean_diff.append(md)

            if n_nodes is None:
                n_nodes = int((1 + np.sqrt(1 + 8 * len(md))) / 2)

            # --- edge sets per PC for this run ---
            diag_data = run_result["diag"]
            pc_edge_map: dict[float, np.ndarray] = {}
            for pc in pcs_to_plot:
                mask_pc = np.isclose(diag_data["cons_pc"], pc)
                pc_edge_map[pc] = np.array(
                    diag_data["cons"][mask_pc], dtype=int
                )
            per_run_pc_edges.append(pc_edge_map)

        # Average mean_diff across runs
        avg_mean_diff = np.mean(np.array(per_run_mean_diff), axis=0)
    else:
        # If all cached, we still need n_nodes from the first cached file
        print("\n[1/3] Skipping data extraction")
        # Get the first cached JSON path correctly
        first_pc_idx = list(cache_status.keys())[0]
        first_thresh = list(cache_status[first_pc_idx].keys())[0]
        first_json = cache_status[first_pc_idx][first_thresh]["json_path"]

        with open(first_json, "r") as f:
            first_metadata = json.load(f)
        # Infer n_nodes from edge indices
        max_edge_idx = (
            max(first_metadata["edge_indices"])
            if first_metadata["edge_indices"]
            else 0
        )
        n_nodes = int((1 + np.sqrt(1 + 8 * (max_edge_idx + 1))) / 2)
        avg_mean_diff = None  # Not needed if all cached
        per_run_pc_edges = None

    # ==================================================================
    # 2.  PAD COORDINATES / COLOURS / ASSIGNMENTS ONCE
    # ==================================================================
    node_coords = np.array(node_coords, dtype=float)
    if n_nodes > len(node_coords):
        pad = np.zeros((n_nodes - len(node_coords), 3))
        node_coords = np.vstack([node_coords, pad])
        if network_assignments is not None:
            network_assignments = list(network_assignments) + ["Unknown"] * (
                n_nodes - len(network_assignments)
            )

    if node_colors is None:
        if network_assignments is not None:
            node_colors = [
                glasser_network_colors.get(str(net), "#000000")
                for net in network_assignments
            ]
        else:
            node_colors = [
                "#a6cee3" if c[0] < 0 else "#fb9a99" for c in node_coords
            ]
    elif isinstance(node_colors, list) and len(node_colors) < n_nodes:
        node_colors = node_colors + ["#000000"] * (n_nodes - len(node_colors))

    i_idx, j_idx = np.tril_indices(n_nodes, k=1)

    # ==================================================================
    # 3.  LOOP:  PC  ×  THRESHOLD
    # ==================================================================
    print(
        f"\n[2/3] Processing {len(pcs_to_plot)} PCs × {len(consensus_thresholds)} thresholds..."
    )

    for pc_idx in pcs_to_plot:
        pc_ui_elements: list = []
        pc_ui_elements.append(
            mo.md(
                f"### PC {pc_idx + 1} (Index {pc_idx}) — Consensus across {n_runs} runs"
            )
        )

        res["results"][pc_idx] = {}

        for thresh in consensus_thresholds:
            min_runs = max(1, int(np.ceil(thresh * n_runs)))
            pct_label = int(thresh * 100)
            safe_thresh = str(pct_label)

            cache_info = cache_status[pc_idx][thresh]

            # --- check if plot already exists ---
            if skip_existing and cache_info["cached"]:
                print(f"PC {pc_idx + 1} @ {pct_label}% - Loading from cache")

                # Load existing metadata
                with open(cache_info["json_path"], "r") as f:
                    metadata = json.load(f)

                # Reconstruct adjacency matrix from saved edge indices
                adj = np.zeros((n_nodes, n_nodes))
                for e in metadata["edge_indices"]:
                    i, j = i_idx[e], j_idx[e]
                    # Use stored weight if available, otherwise recompute
                    if avg_mean_diff is not None:
                        adj[i, j] = avg_mean_diff[e]
                        adj[j, i] = avg_mean_diff[e]
                    else:
                        # Find weight from edge_details if available
                        for edge_detail in metadata.get("edge_details", []):
                            if edge_detail["edge_idx"] == e:
                                adj[i, j] = edge_detail["weight"]
                                adj[j, i] = edge_detail["weight"]
                                break

                # Store results
                res["results"][pc_idx][thresh] = {
                    "adj_matrix": adj,
                    "edges": metadata["edge_indices"],
                    "num_edges": metadata["num_edges"],
                }

                # UI fragment
                desc_md = f"""
#### {pct_label}% Consensus (≥{min_runs}/{n_runs} runs) — PC {pc_idx + 1}
- **Edges displayed:** {metadata["num_edges"]}
- **Range:** `[{metadata["color_range"]["vmin"]:.3f}, {metadata["color_range"]["vmax"]:.3f}]`
- **RED:** Over-connectivity in MDD (MDD > HC)
- **BLUE:** Under-connectivity in MDD (MDD < HC)
"""
                pc_ui_elements.append(mo.md(desc_md))

                # Reconstruct network stats markdown if available
                if "network_statistics" in metadata:
                    net_stats = metadata["network_statistics"]

                    # Nodes table
                    network_stats_md = (
                        "#### Participating Nodes per Network\n\n"
                    )
                    network_stats_md += (
                        "| Network | Node Count |\n| :--- | :---: |\n"
                    )
                    for net_name, node_count in net_stats.get(
                        "nodes_per_network", {}
                    ).items():
                        network_stats_md += f"| {net_name} | {node_count} |\n"

                    # Edges table
                    network_stats_md += "\n#### FC Edges per Network\n\n"
                    network_stats_md += "| Network | Total Edges | Intra-Network | % of Total |\n| :--- | :---: | :---: | :---: |\n"
                    per_network = net_stats.get("per_network", {})
                    intra_network = net_stats.get("intra_network", {})
                    for net_name, total_count in per_network.items():
                        intra_count = intra_network.get(net_name, 0)
                        percentage = (
                            total_count / (2 * metadata["num_edges"])
                        ) * 100
                        network_stats_md += f"| {net_name} | {total_count} | {intra_count} | {percentage:.1f}% |\n"

                    # Inter-network table
                    network_stats_md += "\n#### Top Inter-Network Connections\n\n| Connection Pair | Count |\n| :--- | :---: |\n"
                    for pair_name, count in list(
                        net_stats.get("inter_network", {}).items()
                    )[:10]:
                        network_stats_md += f"| {pair_name} | {count} |\n"
                    network_stats_md += "\n---\n\n"

                    pc_ui_elements.append(mo.md(network_stats_md))

                # Reconstruct edge details table if available
                if "edge_details" in metadata:
                    edge_details_md = (
                        "\n#### Edge Details (Top 20 by |Weight|)\n\n"
                    )
                    edge_details_md += "| Edge Index | Node i | Node j | Network i | Network j | Weight | Runs | Type |\n"
                    edge_details_md += "| :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: |\n"

                    # Sort by absolute weight
                    sorted_edges = sorted(
                        metadata["edge_details"],
                        key=lambda x: abs(x["weight"]),
                        reverse=True,
                    )[:20]

                    for edge_info in sorted_edges:
                        edge_type = (
                            "Intra"
                            if edge_info["network_i"] == edge_info["network_j"]
                            else "Inter"
                        )
                        edge_details_md += f"| {edge_info['edge_idx']} | {edge_info['node_i']} | {edge_info['node_j']} | {edge_info['network_i']} | {edge_info['network_j']} | {edge_info['weight']:.3f} | {edge_info['run_count']}/{n_runs} | {edge_type} |\n"
                    edge_details_md += "\n---\n\n"
                    pc_ui_elements.append(mo.md(edge_details_md))

                pc_ui_elements.append(mo.image(src=cache_info["image_path"]))
                pc_ui_elements.append(mo.md("---"))
                continue

            # --- need to generate this plot ---
            print(f"PC {pc_idx + 1} @ {pct_label}% - Generating...")

            # Collect edge counts across runs
            edge_run_count: dict[int, int] = {}
            for run_idx in range(n_runs):
                for e in per_run_pc_edges[run_idx].get(pc_idx, []):
                    edge_run_count[int(e)] = edge_run_count.get(int(e), 0) + 1

            consensus_edges = [
                e for e, cnt in edge_run_count.items() if cnt >= min_runs
            ]

            # --- regenerate plot ---
            if not consensus_edges:
                pc_ui_elements.append(
                    mo.md(
                        f"**{pct_label}% consensus** (≥{min_runs}/{n_runs} runs): "
                        f"*no edges*"
                    )
                )
                res["results"][pc_idx][thresh] = {
                    "adj_matrix": np.zeros((n_nodes, n_nodes)),
                    "edges": [],
                    "num_edges": 0,
                }
                continue

            # --- adjacency matrix ---
            adj = np.zeros((n_nodes, n_nodes))
            for e in consensus_edges:
                i, j = i_idx[e], j_idx[e]
                adj[i, j] = avg_mean_diff[e]
                adj[j, i] = avg_mean_diff[e]

            # --- participating nodes ---
            participating = np.zeros(n_nodes, dtype=bool)
            for e in consensus_edges:
                participating[i_idx[e]] = True
                participating[j_idx[e]] = True
            node_sizes = np.where(participating, 20, 0)

            # --- network stats ---
            network_node_counts: dict[str, int] = {}
            network_edge_counts: dict[str, int] = {}
            intra_network_edges: dict[str, int] = {}
            inter_network_edges: dict[str, int] = {}
            network_stats_md = ""

            # --- edge details ---
            edge_details_list = []

            if network_assignments is not None:
                for ni in range(n_nodes):
                    if participating[ni]:
                        net = (
                            network_assignments[ni]
                            if ni < len(network_assignments)
                            else "Unknown"
                        )
                        network_node_counts[net] = (
                            network_node_counts.get(net, 0) + 1
                        )

                for e in consensus_edges:
                    i, j = i_idx[e], j_idx[e]
                    ni = (
                        network_assignments[i]
                        if i < len(network_assignments)
                        else "Unknown"
                    )
                    nj = (
                        network_assignments[j]
                        if j < len(network_assignments)
                        else "Unknown"
                    )
                    network_edge_counts[ni] = (
                        network_edge_counts.get(ni, 0) + 1
                    )
                    network_edge_counts[nj] = (
                        network_edge_counts.get(nj, 0) + 1
                    )
                    if ni == nj:
                        intra_network_edges[ni] = (
                            intra_network_edges.get(ni, 0) + 1
                        )
                    else:
                        si, sj = str(ni), str(nj)
                        pair = f"{si} ↔ {sj}" if si < sj else f"{sj} ↔ {si}"
                        inter_network_edges[pair] = (
                            inter_network_edges.get(pair, 0) + 1
                        )

                    # Store edge details
                    edge_details_list.append(
                        {
                            "edge_idx": int(e),
                            "node_i": int(i),
                            "node_j": int(j),
                            "network_i": str(ni),
                            "network_j": str(nj),
                            "weight": float(avg_mean_diff[e]),
                            "run_count": int(edge_run_count[e]),
                            "type": "Intra" if ni == nj else "Inter",
                        }
                    )

                network_stats_md = (
                    f"#### {pct_label}% Consensus — Participating Nodes\n\n"
                )
                network_stats_md += (
                    "| Network | Node Count |\n| :--- | :---: |\n"
                )
                for net, cnt in sorted(
                    network_node_counts.items(),
                    key=lambda x: x[1],
                    reverse=True,
                ):
                    network_stats_md += f"| {net} | {cnt} |\n"

                network_stats_md += (
                    f"\n#### {pct_label}% Consensus — FC Edges per Network\n\n"
                )
                network_stats_md += "| Network | Total Edges | Intra-Network | % of Total |\n| :--- | :---: | :---: | :---: |\n"
                for net, total in sorted(
                    network_edge_counts.items(),
                    key=lambda x: x[1],
                    reverse=True,
                ):
                    intra = intra_network_edges.get(net, 0)
                    pct = (total / (2 * len(consensus_edges))) * 100
                    network_stats_md += (
                        f"| {net} | {total} | {intra} | {pct:.1f}% |\n"
                    )

                network_stats_md += (
                    f"\n#### {pct_label}% Consensus — Top Inter-Network\n\n"
                )
                network_stats_md += (
                    "| Connection Pair | Count |\n| :--- | :---: |\n"
                )
                for pair, cnt in sorted(
                    inter_network_edges.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]:
                    network_stats_md += f"| {pair} | {cnt} |\n"
                network_stats_md += "\n---\n\n"

            # --- colour scale ---
            abs_vals = np.abs(adj[adj != 0])
            vmax = (
                float(np.percentile(abs_vals, 95))
                if len(abs_vals) > 0
                else 0.1
            )
            vmin = -vmax

            # --- figure ---
            if show_legend:
                fig = plt.figure(figsize=(16, 12))
                gs = fig.add_gridspec(
                    2, 3, width_ratios=[1, 1, 0.5], wspace=0.3, hspace=0.3
                )
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0, 1])
                ax3 = fig.add_subplot(gs[1, 0])
                ax4 = fig.add_subplot(gs[1, 1])
                ax_legend = fig.add_subplot(gs[:, 2])
                axes = [ax1, ax2, ax3, ax4]
            else:
                fig, axes = plt.subplots(2, 2, figsize=(12, 12))
                axes = axes.flatten()

            views = [
                ("l", "Left Lateral", False),
                ("z", "Axial", True),
                ("r", "Right Lateral", False),
                ("z", "Axial (different slice)", False),
            ]
            for ax, (mode, title_text, show_cbar) in zip(axes, views):
                plotting.plot_connectome(
                    adj,
                    node_coords,
                    edge_cmap="bwr",
                    edge_vmin=vmin,
                    edge_vmax=vmax,
                    node_size=node_sizes,
                    node_color=node_colors,
                    display_mode=mode,
                    axes=ax,
                    title=title_text,
                    colorbar=show_cbar,
                    alpha=0.8,
                    edge_threshold=0.01,
                )

            # --- legend ---
            if show_legend:
                legend_elements = []
                if network_assignments is not None and network_node_counts:
                    for net_name in sorted(
                        network_node_counts,
                        key=lambda x: network_node_counts[x],
                        reverse=True,
                    ):
                        color = glasser_network_colors.get(
                            str(net_name), "#000000"
                        )
                        legend_elements.append(
                            Line2D(
                                [0],
                                [0],
                                marker="o",
                                color="w",
                                markerfacecolor=color,
                                markersize=12,
                                label=net_name,
                                markeredgecolor="black",
                                markeredgewidth=0.5,
                            )
                        )
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        color="red",
                        linewidth=2,
                        label="Over-connectivity (MDD > HC)",
                    )
                )
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        color="blue",
                        linewidth=2,
                        label="Under-connectivity (MDD < HC)",
                    )
                )
                ax_legend.axis("off")
                ax_legend.legend(
                    handles=legend_elements,
                    loc="center",
                    fontsize=10,
                    frameon=True,
                    fancybox=True,
                    shadow=True,
                    title="Glasser Networks",
                    title_fontsize=11,
                )

            plt.suptitle(
                f"PC {pc_idx + 1} (Index {pc_idx}) — {pct_label}% Consensus "
                f"(≥{min_runs}/{n_runs} runs) — {len(consensus_edges)} edges",
                fontsize=14,
                fontweight="bold",
                y=0.98,
            )
            plt.tight_layout()
            fig.patch.set_facecolor("white")

            # --- save image ---
            fig.savefig(
                cache_info["image_path"],
                dpi=150,
                bbox_inches="tight",
                facecolor="white",
            )

            # --- save metadata ---
            metadata = {
                "pc_index_0_based": int(pc_idx),
                "pc_number_1_based": int(pc_idx + 1),
                "consensus_threshold": float(thresh),
                "min_runs_required": int(min_runs),
                "total_runs": int(n_runs),
                "num_edges": int(len(consensus_edges)),
                "color_range": {"vmin": float(vmin), "vmax": float(vmax)},
                "edge_indices": [int(e) for e in consensus_edges],
                "edge_run_counts": {
                    str(e): edge_run_count[e] for e in consensus_edges
                },
                "edge_details": edge_details_list,
                "network_statistics": {
                    "nodes_per_network": {
                        k: int(v)
                        for k, v in sorted(
                            network_node_counts.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    },
                    "per_network": {
                        k: int(v)
                        for k, v in sorted(
                            network_edge_counts.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    },
                    "intra_network": {
                        k: int(v)
                        for k, v in sorted(
                            intra_network_edges.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    },
                    "inter_network": {
                        k: int(v)
                        for k, v in sorted(
                            inter_network_edges.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    },
                },
            }
            with open(cache_info["json_path"], "w") as f:
                json.dump(metadata, f, indent=4)

            plt.close(fig)

            # --- store results ---
            res["results"][pc_idx][thresh] = {
                "adj_matrix": adj,
                "edges": consensus_edges,
                "num_edges": len(consensus_edges),
            }

            # --- UI fragment ---
            desc_md = f"""
#### {pct_label}% Consensus (≥{min_runs}/{n_runs} runs) — PC {pc_idx + 1}
- **Edges displayed:** {len(consensus_edges)}
- **Range:** `[{vmin:.3f}, {vmax:.3f}]`
- **RED:** Over-connectivity in MDD (MDD > HC)
- **BLUE:** Under-connectivity in MDD (MDD < HC)
"""
            frag = [mo.md(desc_md)]
            if network_stats_md:
                frag.append(mo.md(network_stats_md))

            # Add edge details table
            if edge_details_list:
                edge_details_md = (
                    "\n#### Edge Details (Top 20 by |Weight|)\n\n"
                )
                edge_details_md += "| Edge Index | Node i | Node j | Network i | Network j | Weight | Runs | Type |\n"
                edge_details_md += "| :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: |\n"

                # Sort by absolute weight
                sorted_edges = sorted(
                    edge_details_list,
                    key=lambda x: abs(x["weight"]),
                    reverse=True,
                )[:20]

                for edge_info in sorted_edges:
                    edge_details_md += f"| {edge_info['edge_idx']} | {edge_info['node_i']} | {edge_info['node_j']} | {edge_info['network_i']} | {edge_info['network_j']} | {edge_info['weight']:.3f} | {edge_info['run_count']}/{n_runs} | {edge_info['type']} |\n"
                edge_details_md += "\n---\n\n"
                frag.append(mo.md(edge_details_md))

            frag.append(mo.image(src=cache_info["image_path"]))
            pc_ui_elements.append(mo.vstack(frag, gap=2))
            pc_ui_elements.append(mo.md("---"))

        # end threshold loop
        ui_elements.append(mo.vstack(pc_ui_elements, gap=2))

    # end PC loop
    print(f"\n[3/3] Complete!")
    res["ui"] = mo.vstack(ui_elements, gap=3)
    return res


@app.cell
def _(
    coords_mni,
    network_assignments,
    srpb_fuzzy_extracted_harmonized_fc_matrices_df_list,
    srpb_fuzzy_metadata_dir,
    srpb_fuzzy_metrics_results_list,
    srpb_fuzzy_plot_dir,
    srpb_fuzzy_target_pcs_to_plot,
):
    srpb_fuzzy_consensus_target_pcs_to_plot = [1.0]

    srpb_fuzzy_consensus_pc_plots_result = plot_pcs_consensus_publication_ready(
        results_list=srpb_fuzzy_metrics_results_list,
        fc_matrices_df_list=srpb_fuzzy_extracted_harmonized_fc_matrices_df_list,
        node_coords=coords_mni,
        pcs_to_plot=srpb_fuzzy_target_pcs_to_plot,
        plot_dir=srpb_fuzzy_plot_dir + "/consensus-thresholds/",
        metadata_dir=srpb_fuzzy_metadata_dir + "/consensus-thresholds/",
        consensus_thresholds=[1.0, 0.75, 0.50, 0.25],
        show_legend=True,
        network_assignments=network_assignments,
    )
    return (srpb_fuzzy_consensus_pc_plots_result,)


@app.cell
def _(srpb_fuzzy_consensus_pc_plots_result):
    srpb_fuzzy_consensus_pc_plots_result["ui"]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We also want to run some other analysis
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Please note that this code was written by Qwen3.7-Plus and needs to be double-checked
    """)
    return


@app.function
def plot_robustness_comparison_analysis(
    results_list: list[dict],
    fc_matrices_df_list: list,  # list of pl.DataFrame
    node_coords: np.ndarray,
    pcs_to_plot: list,
    plot_dir: str,
    metadata_dir: str,
    consensus_thresholds: list[float] = [1.0, 0.75, 0.50, 0.25],
    network_assignments=None,
    skip_existing: bool = True,
) -> dict:
    """
    robustness comparison analysis across bootstrap/fuzzy runs.

    Performs 8 different analyses:
    1. Edge weight stability (CV)
    2. Jaccard similarity between runs
    3. Effect sizes with confidence intervals
    4. Convergence analysis
    5. Network topology comparison
    6. Statistical testing (permutation with sign-flipping)
    7. Rich club analysis
    8. Cross-threshold comparison

    Returns
    -------
    dict with keys 'results' and 'ui'.
    """
    import os, json
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    import marimo as mo
    import seaborn as sns
    import networkx as nx

    res: dict = {"results": {}, "plots": {}}
    ui_elements: list = []
    n_runs = len(results_list)

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    print(f"Robustness comparison analysis for {n_runs} runs...")
    print(f"Plot directory: {plot_dir}")
    print(f"Metadata directory: {metadata_dir}")

    # ==================================================================
    # 0.  CHECK CACHE FIRST
    # ==================================================================
    print(f"\n[0/2] Checking cache...")

    pc_plot_names = [
        "weight_stability",
        "jaccard_similarity",
        "effect_sizes",
        "convergence",
        "topology_metrics",
        "pvalue_distribution",
        "rich_club",
        "cross_threshold_comparison",
    ]

    all_plots = []
    all_metas = []
    for pc_idx in pcs_to_plot:
        for p in pc_plot_names:
            all_plots.append(os.path.join(plot_dir, f"pc_{pc_idx + 1}_{p}.png"))
        all_metas.append(
            os.path.join(metadata_dir, f"pc_{pc_idx + 1}_analysis_metadata.json")
        )

    all_cached = all(os.path.exists(p) for p in all_plots) and all(
        os.path.exists(m) for m in all_metas
    )

    if skip_existing and all_cached:
        print("All plots and metadata cached! Loading from cache...")
    else:
        print(f"  Generating {len(all_plots)} plots and metadata...")

    # ==================================================================
    # 1.  DATA EXTRACTION (only if not fully cached)
    # ==================================================================
    per_run_mean_diff = None
    per_run_pc_edges = None
    avg_mean_diff = None
    i_idx, j_idx = None, None
    n_nodes = None

    if not (skip_existing and all_cached):
        print(f"\n[1/2] Extracting data from {n_runs} runs...")
        per_run_mean_diff = []
        per_run_pc_edges = []

        for run_idx, (srpb_res, fc_df) in enumerate(
            zip(results_list, fc_matrices_df_list)
        ):
            if (run_idx + 1) % 20 == 0 or run_idx == 0:
                print(f"  Processing run {run_idx + 1}/{n_runs}...")

            target_col = fc_df["diag"]
            mask = (
                (target_col != -1000)
                & (target_col.is_not_null())
                & ((target_col == 0) | (target_col == 2))
            )
            filtered = fc_df.filter(mask)
            g0 = filtered.filter(pl.col("diag") == 0)[
                "harmonized_fc_matrix"
            ].to_list()
            g2 = filtered.filter(pl.col("diag") == 2)[
                "harmonized_fc_matrix"
            ].to_list()

            md = np.mean(np.array(g2), axis=0) - np.mean(np.array(g0), axis=0)
            per_run_mean_diff.append(md)

            if n_nodes is None:
                n_nodes = int((1 + np.sqrt(1 + 8 * len(md))) / 2)

            diag_data = srpb_res["diag"]
            pc_edge_map = {}
            for pc in pcs_to_plot:
                mask_pc = np.isclose(diag_data["cons_pc"], pc)
                pc_edge_map[pc] = np.array(
                    diag_data["cons"][mask_pc], dtype=int
                )
            per_run_pc_edges.append(pc_edge_map)

        avg_mean_diff = np.mean(np.array(per_run_mean_diff), axis=0)
        i_idx, j_idx = np.tril_indices(n_nodes, k=1)
    else:
        print("\n[1/2] Skipping data extraction")
        # Load n_nodes from the first available metadata file
        with open(all_metas[0], "r") as f:
            meta = json.load(f)
        n_nodes = meta["n_nodes"]
        i_idx, j_idx = np.tril_indices(n_nodes, k=1)

    # ==================================================================
    # 2.  ANALYSIS LOOP
    # ==================================================================
    print(f"\n[2/2] Running analyses for {len(pcs_to_plot)} PCs...")

    for pc_idx in pcs_to_plot:
        print(f"\n  Processing PC {pc_idx + 1} (Index {pc_idx})...")

        pc_results = {}
        pc_ui_elements = []
        pc_ui_elements.append(
            mo.md(
                f"### Robustness Analysis — PC {pc_idx + 1} (Index {pc_idx})"
            )
        )

        # Check if this specific PC is fully cached
        meta_path = os.path.join(
            metadata_dir, f"pc_{pc_idx + 1}_analysis_metadata.json"
        )
        is_pc_cached = (
            skip_existing
            and os.path.exists(meta_path)
            and all(
                os.path.exists(os.path.join(plot_dir, f"pc_{pc_idx + 1}_{p}.png"))
                for p in pc_plot_names
            )
        )

        # ==================================================================
        # IF CACHED: Just load the JSON
        # ==================================================================
        if is_pc_cached:
            print(f"    Loading PC {pc_idx + 1} results from cache...")
            with open(meta_path, "r") as f:
                cached_meta = json.load(f)

            # Restore all analysis results directly from the saved JSON
            pc_results = {
                "weight_stability": cached_meta.get("weight_stability", {}),
                "jaccard_similarity": cached_meta.get(
                    "jaccard_similarity", {}
                ),
                "effect_sizes": cached_meta.get("effect_sizes", {}),
                "convergence": cached_meta.get("convergence", {}),
                "topology": cached_meta.get("topology", {}),
                "permutation_test": cached_meta.get("permutation_test", {}),
                "rich_club": cached_meta.get("rich_club", {}),
                "cross_threshold": cached_meta.get("cross_threshold", {}),
            }
            n_nodes = cached_meta.get("n_nodes", n_nodes)

        # ==================================================================
        # IF NOT CACHED: Run the 8 heavy analyses
        # ==================================================================
        else:
            print(f"    Generating PC {pc_idx + 1} analyses from scratch...")

            # Collect all edges across runs
            all_edges = set()
            edge_run_count = {}
            for run_idx in range(n_runs):
                for e in per_run_pc_edges[run_idx].get(pc_idx, []):
                    all_edges.add(int(e))
                    edge_run_count[int(e)] = edge_run_count.get(int(e), 0) + 1
            all_edges = sorted(all_edges)

            # --- ANALYSIS 1: Edge Weight Stability ---
            if not (
                skip_existing
                and os.path.exists(
                    os.path.join(plot_dir, f"pc_{pc_idx + 1}_weight_stability.png")
                )
            ):
                print("    [1/8] Edge weight stability...")
                edge_stability = {}
                for e in all_edges:
                    weights = [
                        per_run_mean_diff[r][e]
                        for r in range(n_runs)
                        if e in per_run_pc_edges[r].get(pc_idx, [])
                    ]
                    if len(weights) > 1:
                        mean_w, std_w = np.mean(weights), np.std(weights)
                        cv = std_w / np.abs(mean_w) if mean_w != 0 else np.inf
                        edge_stability[e] = {
                            "mean": mean_w,
                            "std": std_w,
                            "cv": cv,
                            "n_runs": len(weights),
                        }

                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                cv_vals = [
                    edge_stability[e]["cv"]
                    for e in edge_stability
                    if edge_stability[e]["cv"] < 10
                ]
                axes[0].hist(cv_vals, bins=30, edgecolor="black", alpha=0.7)
                axes[0].set_title("Edge Weight Stability (CV)")
                axes[0].axvline(1.0, color="red", linestyle="--", label="CV=1")
                axes[0].legend()

                means = [edge_stability[e]["mean"] for e in edge_stability]
                stds = [edge_stability[e]["std"] for e in edge_stability]
                axes[1].scatter(means, stds, alpha=0.5, s=10)
                axes[1].set_title("Mean vs Variability")
                axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)

                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        plot_dir, f"pc_{pc_idx + 1}_weight_stability.png"
                    ),
                    dpi=150,
                )
                plt.close(fig)
                pc_results["weight_stability"] = edge_stability

            # --- ANALYSIS 2: Jaccard Similarity ---
            if not (
                skip_existing
                and os.path.exists(
                    os.path.join(
                        plot_dir, f"pc_{pc_idx + 1}_jaccard_similarity.png"
                    )
                )
            ):
                print("    [2/8] Jaccard similarity...")
                jaccard_matrix = np.zeros((n_runs, n_runs))
                for i in range(n_runs):
                    edges_i = set(per_run_pc_edges[i].get(pc_idx, []))
                    for j in range(i, n_runs):
                        edges_j = set(per_run_pc_edges[j].get(pc_idx, []))
                        intersection = len(edges_i & edges_j)
                        union = len(edges_i | edges_j)
                        jaccard_matrix[i, j] = (
                            intersection / union if union > 0 else 0
                        )
                        jaccard_matrix[j, i] = jaccard_matrix[i, j]

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    jaccard_matrix,
                    cmap="YlOrRd",
                    ax=ax,
                    xticklabels=False,
                    yticklabels=False,
                )
                ax.set_title(
                    f"PC {pc_idx + 1} — Jaccard Similarity Between Runs"
                )
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        plot_dir, f"pc_{pc_idx + 1}_jaccard_similarity.png"
                    ),
                    dpi=150,
                )
                plt.close(fig)

                pc_results["jaccard_similarity"] = {
                    "matrix": jaccard_matrix.tolist(),
                    "mean": float(
                        np.mean(jaccard_matrix[np.triu_indices(n_runs, k=1)])
                    ),
                    "std": float(
                        np.std(jaccard_matrix[np.triu_indices(n_runs, k=1)])
                    ),
                }

            # --- ANALYSIS 3: Effect Sizes with CIs ---
            if not (
                skip_existing
                and os.path.exists(
                    os.path.join(plot_dir, f"pc_{pc_idx + 1}_effect_sizes.png")
                )
            ):
                print("    [3/8] Effect sizes...")
                effect_sizes = {}
                for e in all_edges:
                    diffs = [
                        per_run_mean_diff[r][e]
                        for r in range(n_runs)
                        if e in per_run_pc_edges[r].get(pc_idx, [])
                    ]
                    if len(diffs) > 1:
                        mean_diff, std_diff = np.mean(diffs), np.std(diffs)
                        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
                        ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])
                        effect_sizes[e] = {
                            "cohens_d": float(cohens_d),
                            "mean": float(mean_diff),
                            "ci_95": (float(ci_lower), float(ci_upper)),
                            "significant": not (ci_lower <= 0 <= ci_upper),
                        }

                sorted_edges = sorted(
                    effect_sizes.keys(),
                    key=lambda e: abs(effect_sizes[e]["cohens_d"]),
                    reverse=True,
                )[:20]
                fig, ax = plt.subplots(figsize=(12, 8))
                y_pos = np.arange(len(sorted_edges))
                means = [effect_sizes[e]["mean"] for e in sorted_edges]
                ci_lowers = [effect_sizes[e]["ci_95"][0] for e in sorted_edges]
                ci_uppers = [effect_sizes[e]["ci_95"][1] for e in sorted_edges]

                ax.barh(
                    y_pos,
                    means,
                    xerr=[
                        np.array(means) - np.array(ci_lowers),
                        np.array(ci_uppers) - np.array(means),
                    ],
                    color="steelblue",
                    alpha=0.7,
                    capsize=3,
                )
                ax.set_yticks(y_pos)
                ax.set_yticklabels([f"Edge {e}" for e in sorted_edges])
                ax.set_xlabel("Mean Difference (MDD - HC)")
                ax.set_title(
                    f"PC {pc_idx + 1} — Top 20 Edges by Effect Size (95% CI)"
                )
                ax.axvline(0, color="red", linestyle="--", alpha=0.5)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(plot_dir, f"pc_{pc_idx + 1}_effect_sizes.png"),
                    dpi=150,
                )
                plt.close(fig)
                pc_results["effect_sizes"] = effect_sizes

            # --- ANALYSIS 4: Convergence Analysis ---
            if not (
                skip_existing
                and os.path.exists(
                    os.path.join(plot_dir, f"pc_{pc_idx + 1}_convergence.png")
                )
            ):
                print("    [4/8] Convergence analysis...")
                convergence_curve = []
                for n in range(10, n_runs + 1, max(1, n_runs // 20)):
                    edge_counts = {}
                    for r in range(n):
                        for e in per_run_pc_edges[r].get(pc_idx, []):
                            edge_counts[int(e)] = (
                                edge_counts.get(int(e), 0) + 1
                            )
                    convergence_curve.append(
                        {
                            "n_runs": n,
                            "stable_100": sum(
                                1
                                for cnt in edge_counts.values()
                                if cnt >= 1.0 * n
                            ),
                            "stable_75": sum(
                                1
                                for cnt in edge_counts.values()
                                if cnt >= 0.75 * n
                            ),
                            "stable_50": sum(
                                1
                                for cnt in edge_counts.values()
                                if cnt >= 0.50 * n
                            ),
                        }
                    )

                fig, ax = plt.subplots(figsize=(10, 6))
                n_vals = [c["n_runs"] for c in convergence_curve]
                ax.plot(
                    n_vals,
                    [c["stable_100"] for c in convergence_curve],
                    "o-",
                    label="100% consensus",
                    linewidth=2,
                )
                ax.plot(
                    n_vals,
                    [c["stable_75"] for c in convergence_curve],
                    "s-",
                    label="75% consensus",
                    linewidth=2,
                )
                ax.plot(
                    n_vals,
                    [c["stable_50"] for c in convergence_curve],
                    "^-",
                    label="50% consensus",
                    linewidth=2,
                )
                ax.set_title(f"PC {pc_idx + 1} — Convergence Analysis")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(plot_dir, f"pc_{pc_idx + 1}_convergence.png"),
                    dpi=150,
                )
                plt.close(fig)
                pc_results["convergence"] = convergence_curve

            # --- ANALYSIS 5: Network Topology ---
            if not (
                skip_existing
                and os.path.exists(
                    os.path.join(plot_dir, f"pc_{pc_idx + 1}_topology_metrics.png")
                )
            ):
                print("    [5/8] Network topology...")
                topology_metrics = {}
                for thresh in consensus_thresholds:
                    min_runs = max(1, int(np.ceil(thresh * n_runs)))
                    consensus_edges = [
                        e
                        for e, cnt in edge_run_count.items()
                        if cnt >= min_runs
                    ]
                    if consensus_edges:
                        adj = np.zeros((n_nodes, n_nodes))
                        for e in consensus_edges:
                            i, j = i_idx[e], j_idx[e]
                            adj[i, j] = 1
                            adj[j, i] = 1
                        G = nx.from_numpy_array(adj)
                        G.remove_nodes_from(list(nx.isolates(G)))
                        if len(G.nodes()) > 0:
                            topology_metrics[thresh] = {
                                "n_nodes": len(G.nodes()),
                                "n_edges": len(G.edges()),
                                "density": nx.density(G),
                                "clustering": nx.average_clustering(G),
                                "avg_degree": np.mean(
                                    [d for _, d in G.degree()]
                                ),
                                "connected": nx.is_connected(G),
                            }
                        else:
                            topology_metrics[thresh] = None
                    else:
                        topology_metrics[thresh] = None

                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                thresh_vals = [
                    t
                    for t in consensus_thresholds
                    if topology_metrics.get(t) is not None
                ]
                if thresh_vals:
                    axes[0, 0].plot(
                        thresh_vals,
                        [topology_metrics[t]["n_nodes"] for t in thresh_vals],
                        "o-",
                        linewidth=2,
                    )
                    axes[0, 0].set_title("Nodes vs Threshold")
                    axes[0, 0].grid(True, alpha=0.3)
                    axes[0, 1].plot(
                        thresh_vals,
                        [topology_metrics[t]["n_edges"] for t in thresh_vals],
                        "s-",
                        linewidth=2,
                    )
                    axes[0, 1].set_title("Edges vs Threshold")
                    axes[0, 1].grid(True, alpha=0.3)
                    axes[1, 0].plot(
                        thresh_vals,
                        [topology_metrics[t]["density"] for t in thresh_vals],
                        "^-",
                        linewidth=2,
                    )
                    axes[1, 0].set_title("Density vs Threshold")
                    axes[1, 0].grid(True, alpha=0.3)
                    axes[1, 1].plot(
                        thresh_vals,
                        [
                            topology_metrics[t]["clustering"]
                            for t in thresh_vals
                        ],
                        "D-",
                        linewidth=2,
                    )
                    axes[1, 1].set_title("Clustering vs Threshold")
                    axes[1, 1].grid(True, alpha=0.3)
                plt.suptitle(
                    f"PC {pc_idx + 1} — Network Topology Metrics",
                    fontsize=14,
                    fontweight="bold",
                )
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        plot_dir, f"pc_{pc_idx + 1}_topology_metrics.png"
                    ),
                    dpi=150,
                )
                plt.close(fig)
                pc_results["topology"] = topology_metrics

            # --- ANALYSIS 6: Statistical Testing (Permutation) ---
            if not (
                skip_existing
                and os.path.exists(
                    os.path.join(
                        plot_dir, f"pc_{pc_idx + 1}_pvalue_distribution.png"
                    )
                )
            ):
                print("    [6/8] Statistical testing...")
                permutation_pvalues = {}
                n_permutations = 1000
                for idx, e in enumerate(all_edges[:100]):
                    if idx % 10 == 0:
                        print(f"      Permutation test: {idx}/100 edges...")
                    observed_weight = avg_mean_diff[e]
                    edge_weights = [
                        per_run_mean_diff[r][e]
                        for r in range(n_runs)
                        if e in per_run_pc_edges[r].get(pc_idx, [])
                    ]
                    if len(edge_weights) < 2:
                        continue

                    null_weights = []
                    for _ in range(n_permutations):
                        permuted = np.array(edge_weights) * np.random.choice(
                            [-1, 1], size=len(edge_weights)
                        )
                        null_weights.append(np.mean(permuted))
                    p_value = np.mean(
                        np.abs(null_weights) >= np.abs(observed_weight)
                    )
                    permutation_pvalues[e] = float(p_value)

                fig, ax = plt.subplots(figsize=(10, 6))
                pvals = list(permutation_pvalues.values())
                ax.hist(pvals, bins=20, edgecolor="black", alpha=0.7)
                ax.set_title(
                    f"PC {pc_idx + 1} — Permutation Test P-value Distribution"
                )
                ax.axvline(
                    0.05,
                    color="red",
                    linestyle="--",
                    label="α=0.05 (uncorrected)",
                )
                if pvals:
                    sorted_pvals = sorted(pvals)
                    fdr_threshold = (
                        0.05
                        * np.arange(1, len(sorted_pvals) + 1)
                        / len(sorted_pvals)
                    )
                    significant = sorted_pvals <= fdr_threshold
                    if np.any(significant):
                        fdr_line = sorted_pvals[
                            np.max(np.where(significant)[0])
                        ]
                        ax.axvline(
                            fdr_line,
                            color="green",
                            linestyle="--",
                            label=f"FDR q=0.05 (p≤{fdr_line:.3f})",
                        )
                ax.legend()
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        plot_dir, f"pc_{pc_idx + 1}_pvalue_distribution.png"
                    ),
                    dpi=150,
                )
                plt.close(fig)
                pc_results["permutation_test"] = permutation_pvalues

            # --- ANALYSIS 7: Rich Club Analysis (Multi-Threshold) ---
            if not (
                skip_existing
                and os.path.exists(
                    os.path.join(plot_dir, f"pc_{pc_idx + 1}_rich_club.png")
                )
            ):
                print("    [7/8] Rich club analysis (all thresholds)...")

                rich_club_results = {}

                # Loop through every consensus threshold
                for thresh in consensus_thresholds:
                    min_runs = max(1, int(np.ceil(thresh * n_runs)))
                    consensus_edges = [
                        e
                        for e, cnt in edge_run_count.items()
                        if cnt >= min_runs
                    ]

                    if not consensus_edges:
                        continue

                    # Build adjacency matrix for this threshold
                    adj = np.zeros((n_nodes, n_nodes))
                    for e in consensus_edges:
                        i, j = i_idx[e], j_idx[e]
                        adj[i, j] = 1
                        adj[j, i] = 1

                    G = nx.from_numpy_array(adj)
                    degrees = np.array([d for _, d in G.degree()])

                    # Calculate Rich Club Coefficient for this threshold
                    rich_club_k = {}
                    # Use max degree of this specific network
                    max_degree = (
                        int(np.percentile(degrees, 95))
                        if len(degrees) > 0
                        else 0
                    )

                    for k in range(1, max_degree + 1):
                        rich_nodes = np.where(degrees >= k)[0]
                        if len(rich_nodes) > 1:
                            rich_edges = adj[np.ix_(rich_nodes, rich_nodes)]
                            n_possible = len(rich_nodes) * (
                                len(rich_nodes) - 1
                            )
                            rich_club_k[k] = (
                                np.sum(rich_edges) / n_possible
                                if n_possible > 0
                                else 0
                            )

                    rich_club_results[thresh] = rich_club_k

                # --- Plotting Multiple Lines ---
                fig, ax = plt.subplots(figsize=(10, 6))
                has_data = False

                # Define colors for the lines
                colors = {
                    1.0: "#e74c3c",
                    0.75: "#f39c12",
                    0.50: "#2ecc71",
                    0.25: "#3498db",
                }

                for thresh, data in rich_club_results.items():
                    if data:
                        has_data = True
                        # Sort by degree k to ensure line connects correctly
                        sorted_k = sorted(data.keys())
                        sorted_phi = [data[k] for k in sorted_k]

                        label = f"{int(thresh * 100)}% Consensus"
                        color = colors.get(thresh, "gray")

                        ax.plot(
                            sorted_k,
                            sorted_phi,
                            "o-",
                            linewidth=2,
                            color=color,
                            label=label,
                        )

                if has_data:
                    ax.set_xlabel("Degree Threshold (k)")
                    ax.set_ylabel("Rich Club Coefficient φ(k)")
                    ax.set_title(
                        f"PC {pc_idx + 1} — Rich Club Analysis by Consensus Threshold"
                    )
                    ax.legend(loc="upper left")
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No edges found at any threshold",
                        ha="center",
                        va="center",
                    )

                plt.tight_layout()
                plt.savefig(
                    os.path.join(plot_dir, f"pc_{pc_idx + 1}_rich_club.png"),
                    dpi=150,
                )
                plt.close(fig)

                # Save results (convert keys to strings for JSON)
                pc_results["rich_club"] = {
                    str(thresh): {str(k): v for k, v in data.items()}
                    for thresh, data in rich_club_results.items()
                }

            # --- ANALYSIS 8: Cross-Threshold Comparison ---
            if not (
                skip_existing
                and os.path.exists(
                    os.path.join(
                        plot_dir, f"pc_{pc_idx + 1}_cross_threshold_comparison.png"
                    )
                )
            ):
                print("    [8/8] Cross-threshold comparison...")
                edge_presence = {}
                for e in all_edges:
                    edge_presence[e] = {
                        thresh: (
                            edge_run_count.get(e, 0)
                            >= max(1, int(np.ceil(thresh * n_runs)))
                        )
                        for thresh in consensus_thresholds
                    }

                always_present = [
                    e for e in all_edges if all(edge_presence[e].values())
                ]
                never_present = [
                    e for e in all_edges if not any(edge_presence[e].values())
                ]
                threshold_dependent = [
                    e
                    for e in all_edges
                    if any(edge_presence[e].values())
                    and not all(edge_presence[e].values())
                ]

                fig, ax = plt.subplots(figsize=(10, 6))
                categories = [
                    "Always Present",
                    "Threshold Dependent",
                    "Never Present",
                ]
                counts = [
                    len(always_present),
                    len(threshold_dependent),
                    len(never_present),
                ]
                colors = ["#2ecc71", "#f39c12", "#e74c3c"]
                bars = ax.bar(
                    categories,
                    counts,
                    color=colors,
                    edgecolor="black",
                    alpha=0.7,
                )
                ax.set_title(
                    f"PC {pc_idx + 1} — Edge Stability Across Thresholds"
                )
                for bar, count in zip(bars, counts):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        str(count),
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        plot_dir, f"pc_{pc_idx + 1}_cross_threshold_comparison.png"
                    ),
                    dpi=150,
                )
                plt.close(fig)

                pc_results["cross_threshold"] = {
                    "always_present": len(always_present),
                    "threshold_dependent": len(threshold_dependent),
                    "never_present": len(never_present),
                    "edge_presence": {
                        str(e): {str(t): v for t, v in presence.items()}
                        for e, presence in edge_presence.items()
                    },
                }

            # --- SAVE METADATA ---
            print("    Saving metadata to JSON...")
            metadata = {
                "pc_index": int(pc_idx),
                "n_runs": n_runs,
                "n_nodes": n_nodes,
                "total_edges_seen": len(all_edges),
                "consensus_thresholds": consensus_thresholds,
                "weight_stability": pc_results.get("weight_stability", {}),
                "jaccard_similarity": pc_results.get("jaccard_similarity", {}),
                "effect_sizes": pc_results.get("effect_sizes", {}),
                "convergence": pc_results.get("convergence", {}),
                "topology": pc_results.get("topology", {}),
                "permutation_test": pc_results.get("permutation_test", {}),
                "rich_club": pc_results.get("rich_club", {}),
                "cross_threshold": pc_results.get("cross_threshold", {}),
            }
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=4)

        # ==================================================================
        # BUILD UI (Runs for BOTH cached and newly generated plots)
        # ==================================================================
        print(f"    Building UI for PC {pc_idx + 1}...")

        summary_md = f"""
#### Analysis Summary
- **Total runs:** {n_runs}
- **Consensus thresholds:** {consensus_thresholds}
"""
        if "cross_threshold" in pc_results:
            ct = pc_results["cross_threshold"]
            summary_md += (
                f"- **Always Present:** {ct.get('always_present', 0)}\n"
            )
            summary_md += f"- **Threshold Dependent:** {ct.get('threshold_dependent', 0)}\n"
            summary_md += (
                f"- **Never Present:** {ct.get('never_present', 0)}\n"
            )

        pc_ui_elements.append(mo.md(summary_md))

        # Add all plots to UI
        plot_titles = [
            ("Weight Stability", "weight_stability"),
            ("Jaccard Similarity", "jaccard_similarity"),
            ("Effect Sizes", "effect_sizes"),
            ("Convergence", "convergence"),
            ("Network Topology", "topology_metrics"),
            ("P-value Distribution", "pvalue_distribution"),
            ("Rich Club", "rich_club"),
            ("Cross-Threshold Comparison", "cross_threshold_comparison"),
        ]

        for title, filename_suffix in plot_titles:
            plot_path = os.path.join(
                plot_dir, f"pc_{pc_idx + 1}_{filename_suffix}.png"
            )
            if os.path.exists(plot_path):
                pc_ui_elements.append(mo.md(f"#### {title}"))
                pc_ui_elements.append(mo.image(src=plot_path))

        ui_elements.append(mo.vstack(pc_ui_elements, gap=2))
        res["results"][pc_idx] = pc_results

    print("\nRobustness analysis complete!")
    res["ui"] = mo.vstack(ui_elements, gap=3)
    return res


@app.cell
def _(
    coords_mni,
    network_assignments,
    srpb_fuzzy_extracted_harmonized_fc_matrices_df_list,
    srpb_fuzzy_metadata_dir,
    srpb_fuzzy_metrics_results_list,
    srpb_fuzzy_plot_dir,
):
    srpb_robustness_analysis_result = plot_robustness_comparison_analysis(
        results_list=srpb_fuzzy_metrics_results_list,
        fc_matrices_df_list=srpb_fuzzy_extracted_harmonized_fc_matrices_df_list,
        node_coords=coords_mni,
        pcs_to_plot=[1.0],
        plot_dir=srpb_fuzzy_plot_dir + "/robustness-analysis/",
        metadata_dir=srpb_fuzzy_metadata_dir + "/robustness-analysis/",
        consensus_thresholds=[1.0, 0.75, 0.50, 0.25],
        network_assignments=network_assignments,
        skip_existing=True,
    )
    return (srpb_robustness_analysis_result,)


@app.cell
def _(srpb_robustness_analysis_result):
    srpb_robustness_analysis_result["ui"]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Extraction and harmonization of the regular the FC matrices of the BMB dataset
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We first need to gather all the different subjects scrub and time series files
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Assign each time series file to its subject
    """)
    return


@app.cell
def _():
    bnb_time_series_directory_paths = [
        Path(
            "/home/cbi-data34/ayumu/BMB_UHI/derivatives/HCP/time_series/Glasser_filtering1_GSR1_scrubbing1/"
        ),
        Path(
            "/home/cbi-biomark03/ayumu/BMB_NEW/derivatives/HCP/time_series/Glasser_filtering1_GSR1_scrubbing1/"
        ),
        Path(
            "/home/cbi-biomark03/ayumu/BMB_ALL/derivatives/HCP/time_series/Glasser_filtering1_GSR1_scrubbing1/"
        ),
        Path(
            "/home/cbi-biomark03/ayumu/BMB_2023/derivatives/HCP/time_series/Glasser_filtering1_GSR1_scrubbing1/"
        ),
    ]
    return (bnb_time_series_directory_paths,)


@app.cell
def _(bnb_time_series_directory_paths):
    bmb_time_series_directory_suffixes = {
        bnb_time_series_directory_paths[0]: "_parcel.csv",
        bnb_time_series_directory_paths[1]: "_parcel.csv",
        bnb_time_series_directory_paths[2]: "_parcel.csv",
        bnb_time_series_directory_paths[3]: "_parcel.csv",
    }
    return (bmb_time_series_directory_suffixes,)


@app.cell
def _(bnb_time_series_directory_paths):
    bmb_time_series_file_paths = [
        file
        for directory_path in bnb_time_series_directory_paths
        for file in directory_path.iterdir()
        if file.is_file()
    ]
    return


@app.cell
def _(bmb_time_series_directory_suffixes, bnb_time_series_directory_paths):
    bmb_time_series_separated_dicts_list = [
        {
            (
                re.split(
                    r"[_-]?rFMRI",
                    path.name.removesuffix(
                        bmb_time_series_directory_suffixes[_dir_path]
                    ),
                    flags=re.IGNORECASE,
                )[0],
                int(re.search(r"rFMRI(\d+)", path.name, re.IGNORECASE).group(1)),
            ): path
            for path in _dir_path.iterdir()
            if path.is_file()
            and path.name.endswith(bmb_time_series_directory_suffixes[_dir_path])
            and re.search(r"rFMRI(\d+)", path.name, re.IGNORECASE)
        }
        for _dir_path in bnb_time_series_directory_paths
    ]
    return (bmb_time_series_separated_dicts_list,)


@app.cell
def _(bmb_time_series_separated_dicts_list):
    bmb_time_series_combined_dict = {}
    for _d in bmb_time_series_separated_dicts_list:
        bmb_time_series_combined_dict.update(_d)

    time_series_sub_ids = [k[0] for k in bmb_time_series_combined_dict.keys()]
    time_series_sessions = [k[1] for k in bmb_time_series_combined_dict.keys()]

    bmb_time_series_file_paths_df = pl.DataFrame(
        {
            "participants_id": time_series_sub_ids,
            "session": time_series_sessions,
            "time_series_path": [
                str(p) for p in bmb_time_series_combined_dict.values()
            ],
        }
    )
    return (bmb_time_series_file_paths_df,)


@app.cell
def _(bmb_time_series_file_paths_df):
    bmb_time_series_file_paths_df.head()
    return


@app.cell
def _(harmonized_bmb_fc_matrices_metadata_df):
    harmonized_bmb_fc_matrices_metadata_df.head()
    return


@app.cell
def _(bmb_time_series_file_paths_df, harmonized_bmb_fc_matrices_metadata_df):
    bmb_time_series_file_df = bmb_time_series_file_paths_df.join(
        harmonized_bmb_fc_matrices_metadata_df,
        on=["participants_id", "session"],
        how="inner",
    )
    return (bmb_time_series_file_df,)


@app.cell
def _(harmonized_bmb_fc_matrices_metadata_df):
    harmonized_bmb_fc_matrices_metadata_df.height
    return


@app.cell
def _(bmb_time_series_file_df):
    bmb_time_series_file_df.height
    return


@app.cell
def _(bmb_time_series_file_paths_df, harmonized_bmb_fc_matrices_metadata_df):
    bmb_time_series_missing_df = harmonized_bmb_fc_matrices_metadata_df.join(
        bmb_time_series_file_paths_df,
        on=["participants_id", "session"],
        how="anti",
    )
    return (bmb_time_series_missing_df,)


@app.cell
def _(bmb_time_series_missing_df):
    bmb_time_series_missing_df.height
    return


@app.cell
def _(bmb_time_series_missing_df):
    bmb_time_series_missing_df.select("participants_id", "session")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Please note that this code was generated by Qwen3.6-27b
    """)
    return


@app.cell
def _(bmb_time_series_missing_df):
    bmb_time_series_missing_df.group_by("participants_id").agg(
        pl.len().alias("num_missing"),
        pl.col("session").sort().implode().alias("missing_sessions"),
    ).group_by("missing_sessions").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Assign each scrub file to its subject
    """)
    return


@app.cell
def _():
    bmb_scrub_directory_paths = [
        Path(
            "/home/cbi-data34/ayumu/BMB_UHI/derivatives/HCP/scrub//Glasser_filtering1_GSR1_scrubbing1/"
        ),
        Path(
            "/home/cbi-biomark03/ayumu/BMB_NEW/derivatives/HCP/scrub/Glasser_filtering1_GSR1_scrubbing1/"
        ),
        Path(
            "/home/cbi-biomark03/ayumu/BMB_ALL/derivatives/HCP/scrub/Glasser_filtering1_GSR1_scrubbing1/"
        ),
        Path(
            "/home/cbi-biomark03/ayumu/BMB_2023/derivatives/HCP/scrub/Glasser_filtering1_GSR1_scrubbing1/"
        ),
    ]
    return (bmb_scrub_directory_paths,)


@app.cell
def _(bmb_scrub_directory_paths):
    bmb_scrub_directory_suffixes = {
        bmb_scrub_directory_paths[0]: "_scrub.csv",
        bmb_scrub_directory_paths[1]: "_scrub.csv",
        bmb_scrub_directory_paths[2]: "_scrub.csv",
        bmb_scrub_directory_paths[3]: "_scrub.csv",
    }
    return (bmb_scrub_directory_suffixes,)


@app.cell
def _(bnb_time_series_directory_paths):
    bmb_scub_file_paths = [
        file
        for directory_path in bnb_time_series_directory_paths
        for file in directory_path.iterdir()
        if file.is_file()
    ]
    return


@app.cell
def _(bmb_scrub_directory_paths, bmb_scrub_directory_suffixes):
    bmb_scrub_separated_dicts_list = [
        {
            (
                re.split(
                    r"[_-]?rFMRI",
                    path.name.removesuffix(
                        bmb_scrub_directory_suffixes[_dir_path]
                    ),
                    flags=re.IGNORECASE,
                )[0],
                int(re.search(r"rFMRI(\d+)", path.name, re.IGNORECASE).group(1)),
            ): path
            for path in _dir_path.iterdir()
            if path.is_file()
            and path.name.endswith(bmb_scrub_directory_suffixes[_dir_path])
            and re.search(r"rFMRI(\d+)", path.name, re.IGNORECASE)
        }
        for _dir_path in bmb_scrub_directory_paths
    ]
    return (bmb_scrub_separated_dicts_list,)


@app.cell
def _(bmb_scrub_separated_dicts_list):
    bmb_scrub_combined_dict = {}
    for _d in bmb_scrub_separated_dicts_list:
        bmb_scrub_combined_dict.update(_d)

    scrub_sub_ids = [k[0] for k in bmb_scrub_combined_dict.keys()]
    scrub_sessions = [k[1] for k in bmb_scrub_combined_dict.keys()]

    bmb_scrub_file_paths_df = pl.DataFrame(
        {
            "participants_id": scrub_sub_ids,
            "session": scrub_sessions,
            "scrub_path": [str(p) for p in bmb_scrub_combined_dict.values()],
        }
    )
    return (bmb_scrub_file_paths_df,)


@app.cell
def _(bmb_scrub_file_paths_df):
    bmb_scrub_file_paths_df.head()
    return


@app.cell
def _(bmb_scrub_file_paths_df, bmb_time_series_file_df):
    bmb_time_series_scrub_file_df = bmb_scrub_file_paths_df.join(
        bmb_time_series_file_df,
        on=["participants_id", "session"],
        how="inner",
    )
    return (bmb_time_series_scrub_file_df,)


@app.cell
def _(bmb_time_series_file_df):
    bmb_time_series_file_df.height
    return


@app.cell
def _(bmb_time_series_scrub_file_df):
    bmb_time_series_scrub_file_df.height
    return


@app.cell
def _(bmb_scrub_file_paths_df, bmb_time_series_file_df):
    bmb_scrub_missing_df = bmb_time_series_file_df.join(
        bmb_scrub_file_paths_df, on=["participants_id", "session"], how="anti"
    )
    return (bmb_scrub_missing_df,)


@app.cell
def _(bmb_scrub_missing_df):
    bmb_scrub_missing_df.height
    return


@app.cell
def _(bmb_scrub_missing_df):
    bmb_scrub_missing_df.head()
    return


@app.cell
def _(bmb_scrub_missing_df):
    bmb_scrub_missing_df.group_by("participants_id").agg(
        pl.len().alias("num_missing"),
        pl.col("session").sort().implode().alias("missing_sessions"),
    ).group_by("missing_sessions").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Extraction and harmonization of the BMB perturbated FC matrices
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Remove file where we don't have permission
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Extraction of the fuzzy FC matrices
    """)
    return


@app.cell
def _():
    bmb_fuzzy_fc_matrices_output_path = (
        "/home/cbi-biomark/olivier.amacker/fuzzy-fc-matrices/bmb"
    )
    return (bmb_fuzzy_fc_matrices_output_path,)


@app.cell
def _(bmb_time_series_scrub_file_df):
    bmb_ts_paths = bmb_time_series_scrub_file_df["time_series_path"].to_list()
    bmb_scrub_paths = bmb_time_series_scrub_file_df["scrub_path"].to_list()
    return bmb_scrub_paths, bmb_ts_paths


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Please note that the following comment was written by Opencode Kimi K2.7-Code
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Extract the BMB fuzzy FC matrices. We use only 15 fuzzy runs here (compared to 100 for SRPB)
    as a quick exploratory analysis to see if the perturbation effects are similar across datasets.
    The smaller number also reduces computation time given the larger BMB sample size
    """)
    return


@app.cell
def _():
    num_fuzzy_bmb_run = 15
    return (num_fuzzy_bmb_run,)


@app.cell
def _(
    bmb_fuzzy_fc_matrices_output_path,
    bmb_scrub_paths,
    bmb_time_series_scrub_file_df,
    bmb_ts_paths,
    fuzzy_container_image,
    num_fuzzy_bmb_run,
):
    bmb_fuzzy_extracted_fc_matrices_df_list = run_fuzzy_extraction_runs(
        num_fuzzy_run=num_fuzzy_bmb_run,
        container_name="fuzzy-container",
        container_image=fuzzy_container_image,
        time_series_scrub_file_df=bmb_time_series_scrub_file_df,
        ts_paths=bmb_ts_paths,
        scrub_paths=bmb_scrub_paths,
        fuzzy_fc_matrices_output_path=bmb_fuzzy_fc_matrices_output_path,
    )
    return (bmb_fuzzy_extracted_fc_matrices_df_list,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Harmonization of the BMB fuzzy FC matrices
    """)
    return


@app.cell
def _(bmb_fuzzy_fc_matrices_output_path, regular_run_name):
    bmb_run_fc_matrices_output_dir = (
        f"{bmb_fuzzy_fc_matrices_output_path}/{regular_run_name}/"
    )

    bmb_run_fc_matrices_cache_filename = (
        "Glasser_filtering1_GSR1_fuzzy_fc_matrices.pkl"
    )
    bmb_run_fc_matrices_cache_path = os.path.join(
        bmb_run_fc_matrices_output_dir, bmb_run_fc_matrices_cache_filename
    )
    return bmb_run_fc_matrices_cache_path, bmb_run_fc_matrices_output_dir


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We start by extracting the different FC matrices
    """)
    return


@app.cell
def _(
    bmb_run_fc_matrices_cache_path,
    bmb_run_fc_matrices_output_dir,
    bmb_scrub_paths,
    bmb_time_series_scrub_file_df,
    bmb_ts_paths,
):
    bmb_extracted_fc_matrices_df = extract_regular_fc_matrices(
        bmb_run_fc_matrices_cache_path,
        bmb_run_fc_matrices_output_dir,
        bmb_time_series_scrub_file_df,
        bmb_ts_paths,
        bmb_scrub_paths,
    )
    return (bmb_extracted_fc_matrices_df,)


@app.cell
def _(bmb_extracted_fc_matrices_df):
    bmb_extracted_fc_matrices_df.height
    return


@app.cell
def _():
    bmb_fc_matrices_bias_output_dir = Path(
        f"/home/cbi-biomark/olivier.amacker/bias/bmb"
    )

    bmb_harmonized_fc_matrices_output_dir = Path(
        f"/home/cbi-biomark/olivier.amacker/harmonized-fc-matrices/bmb"
    )
    return (
        bmb_fc_matrices_bias_output_dir,
        bmb_harmonized_fc_matrices_output_dir,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Get the subjects that are in both files df but not in the metadata one
    """)
    return


@app.cell
def _():
    bmb_metadata_dataframe = pl.read_csv(
        "/home/cbi-biomark03/ayumu/HARP/data/preproc_bmb/all_data_sub_Glasser_GSR1.csv",
        columns=["participants_id", "session"],
        null_values=["-", "NA", "N/A", ""],
    )
    return


app._unparsable_cell(
    r"""
        bmb_missing_df = bmb_metadata_dataframe.join(
        bmb_time_series_file_df.join(
            bmb_scrub_file_paths_df, on=["participants_id", "session"], how="inner"
        ),
        on=["participants_id", "session"],
        how="anti",
    )
    bmb_missing_df.height
    """,
    name="_"
)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Please note that the following comment was written by Opencode Kimi K2.7-Code
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Harmonize each BMB fuzzy run using the same traveling-subject harmonization approach.
    For each run, we estimate the site and protocol biases from the fuzzy-extracted matrices
    and subtract them to obtain harmonized FC matrices.
    """)
    return


@app.cell
def _(
    bmb_extracted_fc_matrices_df,
    bmb_fc_matrices_bias_output_dir,
    bmb_fuzzy_extracted_fc_matrices_df_list,
    bmb_harmonized_fc_matrices_output_dir,
    bmb_missing_df,
    estimate_bias,
):
    bmb_fuzzy_extracted_harmonized_fc_matrices_df_list = []

    for _run_idx, fuzzy_bmb_extracted_fc_matrices_df in enumerate(
        bmb_fuzzy_extracted_fc_matrices_df_list
    ):
        print(f"Harmonizing run {_run_idx}")

        bmb_fuzzy_extracted_bias_dict = estimate_bias(
            output_dir_path=bmb_fc_matrices_bias_output_dir / f"run-{_run_idx}",
            rs_connectivity=np.stack(
                bmb_extracted_fc_matrices_df["fc_matrix"].to_numpy(), axis=1
            ),
            dataset="BMB",
            subjects_to_remove_df=bmb_missing_df.select(
                "participants_id", "session"
            ),
        )

        bmb_fuzzy_extracted_harmonized_fc_matrices = harmonize_connectivity(
            bias_dictionnary=bmb_fuzzy_extracted_bias_dict,
            connectivity=np.stack(
                fuzzy_bmb_extracted_fc_matrices_df["fc_matrix"].to_numpy(), axis=0
            ),
            subjects_metadata_dataframe=fuzzy_bmb_extracted_fc_matrices_df,
            subjects_to_remove_df=bmb_missing_df.select(
                "participants_id", "session"
            ),
            dataset="BMB",
            output_path=bmb_harmonized_fc_matrices_output_dir / f"run-{_run_idx}",
        )

        bmb_fuzzy_extracted_harmonized_fc_matrices_df = (
            fuzzy_bmb_extracted_fc_matrices_df.with_columns(
                pl.Series(
                    name="harmonized_fc_matrix",
                    values=bmb_fuzzy_extracted_harmonized_fc_matrices,
                )
            ).select(
                "sub_id",
                "time_series_path",
                "fc_matrix",
                "harmonized_fc_matrix",
                pl.exclude(
                    "sub_id",
                    "time_series_path",
                    "fc_matrix",
                    "harmonized_fc_matrix",
                ),
            )
        )

        bmb_fuzzy_extracted_harmonized_fc_matrices_df_list.append(
            bmb_fuzzy_extracted_harmonized_fc_matrices_df
        )
    return (bmb_fuzzy_extracted_harmonized_fc_matrices_df_list,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Comparison of the BMB PCA features extraction with regular and perturbated FC matrices
    """)
    return


@app.cell
def _():
    old_bmb_fuzzy_plot_dir = (
        "./res/pca-dim-reduction/bmb/fuzzy-features-extraction/old/plots"
    )
    old_bmb_fuzzy_ttest_dir = (
        "./res/pca-dim-reduction/bmb/fuzzy-features-extraction/old/t-tests"
    )
    old_bmb_fuzzy_cache_dir = (
        "./res/pca-dim-reduction/bmb/fuzzy-features-extraction/old/cache"
    )
    old_bmb_fuzzy_metadata_dir = (
        "./res/pca-dim-reduction/bmb/fuzzy-features-extraction/old/metadatas"
    )
    return (
        old_bmb_fuzzy_cache_dir,
        old_bmb_fuzzy_plot_dir,
        old_bmb_fuzzy_ttest_dir,
    )


@app.cell
def _():
    bmb_fuzzy_plot_dir = (
        "./res/pca-dim-reduction/bmb/fuzzy-features-extraction/plots"
    )
    bmb_fuzzy_ttest_dir = (
        "./res/pca-dim-reduction/bmb/fuzzy-features-extraction/t-tests"
    )
    bmb_fuzzy_cache_dir = (
        "./res/pca-dim-reduction/bmb/fuzzy-features-extraction/cache"
    )
    bmb_fuzzy_metadata_dir = (
        "./res/pca-dim-reduction/bmb/fuzzy-features-extraction/metadatas"
    )
    return (
        bmb_fuzzy_cache_dir,
        bmb_fuzzy_metadata_dir,
        bmb_fuzzy_plot_dir,
        bmb_fuzzy_ttest_dir,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Filter the BMB fuzzy-extracted harmonized FC matrices to only keep the MDD and HC subjects
    """)
    return


@app.cell
def _(bmb_fuzzy_extracted_harmonized_fc_matrices_df_list):
    bmb_fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df_list = []
    for (
        _run_idx,
        _bmb_fuzzy_extracted_harmonized_fc_matrices_df,
    ) in enumerate(bmb_fuzzy_extracted_harmonized_fc_matrices_df_list):
        bmb_fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df_list.append(
            _bmb_fuzzy_extracted_harmonized_fc_matrices_df.filter(
                pl.col("diag").is_in([0, 2])
            )
        )
    return (bmb_fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df_list,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Comparison of the Fuzzy and Regular PC BMB metrics graphs
    """)
    return


@app.cell
def _(
    bmb_fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df_list,
    old_bmb_fuzzy_cache_dir,
    old_bmb_fuzzy_plot_dir,
    old_bmb_fuzzy_ttest_dir,
    old_metric_dict,
):
    old_bmb_fuzzy_metrics_results_list = []
    old_bmb_fuzzy_selected_pcs = []
    old_bmb_fuzzy_metrics_ui_list = []


    for (
        _run_idx,
        _bmb_fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df,
    ) in enumerate(bmb_fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df_list):
        print(f"Calculating metrics for run {_run_idx}")

        old_bmb_fuzy_metrics_dict = old_calculate_metrics(
            df=bmb_filter(_bmb_fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df),
            metric_dict=old_metric_dict,
            alpha_threshold=0.05,
            plot_dir=old_bmb_fuzzy_plot_dir + f"/run-{_run_idx}/",
            ttest_dir=old_bmb_fuzzy_ttest_dir + f"/run-{_run_idx}/",
            cache_dir=old_bmb_fuzzy_cache_dir + f"/run-{_run_idx}/",
        )

        old_bmb_fuzzy_metrics_results_list.append(old_bmb_fuzy_metrics_dict["results"])
        old_bmb_fuzzy_metrics_ui_list.append(old_bmb_fuzy_metrics_dict["ui"])
    return (old_bmb_fuzzy_metrics_results_list,)


@app.cell
def _(
    bmb_fuzzy_cache_dir,
    bmb_fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df_list,
    bmb_fuzzy_plot_dir,
    bmb_fuzzy_ttest_dir,
    metric_dict,
):
    bmb_fuzzy_metrics_results_list = []
    bmb_fuzzy_selected_pcs = []
    bmb_fuzzy_metrics_ui_list = []


    for (
        _run_idx,
        bmb_fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df,
    ) in enumerate(bmb_fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df_list):
        print(f"Calculating metrics for run {_run_idx}")

        bmb_fuzy_metrics_dict = calculate_metrics(
            df=bmb_filter(bmb_fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df),
            metric_dict=metric_dict,
            alpha_threshold=0.05,
            n_pcs=5,
            plot_dir=bmb_fuzzy_plot_dir + f"/run-{_run_idx}/",
            ttest_dir=bmb_fuzzy_ttest_dir + f"/run-{_run_idx}/",
            cache_dir=bmb_fuzzy_cache_dir + f"/run-{_run_idx}/",
        )

        bmb_fuzzy_metrics_results_list.append(bmb_fuzy_metrics_dict["results"])
        bmb_fuzzy_selected_pcs.append(bmb_fuzy_metrics_dict["selected_pcs"])
        bmb_fuzzy_metrics_ui_list.append(bmb_fuzy_metrics_dict["ui"])
    return bmb_fuzzy_metrics_results_list, bmb_fuzzy_metrics_ui_list


@app.cell
def _(bmb_fuzzy_metrics_ui_list, bmb_ui):
    bmb_pc_metrics_comparison = mo.vstack(
        [
            mo.vstack(
                [
                    mo.md(
                        f"### Comparison of the Fuzzy run {_run_idx} vs Regular PC metrics graphs"
                    ),
                    mo.hstack(
                        [
                            _bmb_fuzzy_metrics_ui_list,
                            bmb_ui,
                        ],
                        gap=2,
                    ),
                ]
            )
            for _run_idx, _bmb_fuzzy_metrics_ui_list in enumerate(
                bmb_fuzzy_metrics_ui_list
            )
        ],
        gap=3,
    )

    bmb_pc_metrics_comparison
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Comparison of the BMB Fuzzy and Regular PC metrics results
    """)
    return


@app.cell
def _(old_bmb_fuzzy_metrics_results_list):
    old_bmb_fuzzy_selected_mdd_pcs_list = list(
            map(lambda x: select_mdd_pc(x, ["bdi"]), old_bmb_fuzzy_metrics_results_list)
        )

    old_bmb_fuzzy_selected_mdd_pcs_list
    return


@app.cell
def _(bmb_fuzzy_metrics_results_list):
    bmb_fuzzy_top_pc = list(
        map(lambda x: x["diag"]["selected_pcs_with_scores"][0][0], bmb_fuzzy_metrics_results_list)
    )
    return (bmb_fuzzy_top_pc,)


@app.cell
def _(bmb_fuzzy_top_pc):
    set(bmb_fuzzy_top_pc)
    return


@app.cell
def _(
    bmb_fuzzy_extracted_harmonized_fc_matrices_df_list,
    bmb_fuzzy_metadata_dir,
    bmb_fuzzy_metrics_results_list,
    bmb_fuzzy_plot_dir,
    coords_mni,
    network_assignments,
):
    bmb_fuzzy_target_pcs_to_plot = [1.0]

    bmb_fuzzy_pc_plots_result_list = [
        plot_pcs(
            results=x,
            fc_matrices_df=bmb_fuzzy_extracted_harmonized_fc_matrices_df_list[
                _run_idx
            ],
            node_coords=coords_mni,
            pcs_to_plot=bmb_fuzzy_target_pcs_to_plot,
            plot_dir=bmb_fuzzy_plot_dir + f"/run-{_run_idx}/",
            metadata_dir=bmb_fuzzy_metadata_dir + f"/run-{_run_idx}/",
            show_legend=True,
            network_assignments=network_assignments,
        )
        for _run_idx, x in enumerate(bmb_fuzzy_metrics_results_list)
    ]
    return bmb_fuzzy_pc_plots_result_list, bmb_fuzzy_target_pcs_to_plot


@app.cell
def _(bmb_fuzzy_pc_plots_result_list, bmb_pc_plots_results):
    bmb_pc_plots_result_comparison = mo.vstack(
        [
            mo.vstack(
                [
                    mo.md(
                        f"### Comparison of the Fuzzy run {_run_idx} vs Regular PC plots"
                    ),
                    mo.hstack(
                        [
                            bmb_fuzzy_pc_plots_result["ui_elements"][1.0],
                            bmb_pc_plots_results["ui_elements"][1.0],
                        ],
                        gap=2,
                    ),
                ]
            )
            for _run_idx, bmb_fuzzy_pc_plots_result in enumerate(
                bmb_fuzzy_pc_plots_result_list
            )
        ],
        gap=3,
    )

    bmb_pc_plots_result_comparison
    return


@app.cell
def _(
    bmb_fuzzy_extracted_harmonized_fc_matrices_df_list,
    bmb_fuzzy_metadata_dir,
    bmb_fuzzy_metrics_results_list,
    bmb_fuzzy_plot_dir,
    bmb_fuzzy_target_pcs_to_plot,
    coords_mni,
    network_assignments,
):
    bmb_fuzzy_consensus_target_pcs_to_plot = [1.0]

    bmb_fuzzy_consensus_pc_plots_result = plot_pcs_consensus_publication_ready(
        results_list=bmb_fuzzy_metrics_results_list,
        fc_matrices_df_list=bmb_fuzzy_extracted_harmonized_fc_matrices_df_list,
        node_coords=coords_mni,
        pcs_to_plot=bmb_fuzzy_target_pcs_to_plot,
        plot_dir=bmb_fuzzy_plot_dir + "/consensus-thresholds/",
        metadata_dir=bmb_fuzzy_metadata_dir + "/consensus-thresholds/",
        consensus_thresholds=[1.0, 0.75, 0.50, 0.25],
        show_legend=True,
        network_assignments=network_assignments,
    )
    return (bmb_fuzzy_consensus_pc_plots_result,)


@app.cell
def _(bmb_fuzzy_consensus_pc_plots_result):
    bmb_fuzzy_consensus_pc_plots_result["ui"]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We also want to run a robustness analysis
    """)
    return


@app.cell
def _(
    bmb_fuzzy_extracted_harmonized_fc_matrices_df_list,
    bmb_fuzzy_metadata_dir,
    bmb_fuzzy_metrics_results_list,
    bmb_fuzzy_plot_dir,
    coords_mni,
    network_assignments,
):
    bmb_robustness_analysis_result = plot_robustness_comparison_analysis(
        results_list=bmb_fuzzy_metrics_results_list,
        fc_matrices_df_list=bmb_fuzzy_extracted_harmonized_fc_matrices_df_list,
        node_coords=coords_mni,
        pcs_to_plot=[1.0],
        plot_dir=bmb_fuzzy_plot_dir + "/robustness-analysis/",
        metadata_dir=bmb_fuzzy_metadata_dir + "/robustness-analysis/",
        consensus_thresholds=[1.0, 0.75, 0.50, 0.25],
        network_assignments=network_assignments,
        skip_existing=True,
    )
    return (bmb_robustness_analysis_result,)


@app.cell
def _(bmb_robustness_analysis_result):
    bmb_robustness_analysis_result["ui"]
    return


if __name__ == "__main__":
    app.run()
