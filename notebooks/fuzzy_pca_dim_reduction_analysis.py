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
    from pcafeat import select_pca_features, select_ttest_features
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
    We start by loading the harmonized FC matrices metadata

    Please note that we are using the Glasser parcelation with the global signal regression.
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
    harmonized_srpb_fc_matrices_df["harmonized_fc_matrix"].head(1)
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
def calculate_features(
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
        pca_extract_multipletests_alpha=alpha_threshold,
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Define a function to calculate the PCA features extraction for the different metris
    """)
    return


@app.function
def calculate_metrics(
    df: pd.DataFrame,
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
            results = calculate_features(
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
    srpb_plot_dir = "./res/pca-dim-reduction/srpb/features-extraction/plots/"
    srpb_ttest_dir = "./res/pca-dim-reduction/srpb/features-extraction/t-tests/"
    srpb_cache_dir = "./res/pca-dim-reduction/srpb/features-extraction/cache/"
    srpb_metadata_dir = (
        "./res/pca-dim-reduction/srpb/features-extraction/metadatas/"
    )
    return srpb_cache_dir, srpb_metadata_dir, srpb_plot_dir, srpb_ttest_dir


@app.cell
def _():
    metric_dict = {
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
    return (metric_dict,)


@app.cell
def _(
    harmonized_srpb_fc_matrices_hc_mdd_df,
    metric_dict,
    srpb_cache_dir,
    srpb_plot_dir,
    srpb_ttest_dir,
):
    srpb_metrics_dict = calculate_metrics(
        df=harmonized_srpb_fc_matrices_hc_mdd_df,
        metric_dict=metric_dict,
        alpha_threshold=0.05,
        plot_dir=srpb_plot_dir,
        ttest_dir=srpb_ttest_dir,
        cache_dir=srpb_cache_dir,
    )

    srpb_results = srpb_metrics_dict["results"]
    srpb_ui = srpb_metrics_dict["ui"]
    return srpb_results, srpb_ui


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
    We start by obtaining the brain parcelation
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
    return coords_mni, labels, network_assignments


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
def plot_pcs_publication_ready(
    srpb_results: dict,
    fc_matrices_df,  # pl.DataFrame
    node_coords: np.ndarray,
    pcs_to_plot: list,
    plot_dir: str,
    metadata_dir: str,
    node_colors=None,
    show_legend=True,
    network_assignments=None,
) -> dict:
    """
    Generates a complete Marimo UI for a list of Principal Components.
    Returns a dictionary with 'results' (keyed by PC index) and 'ui'.
    Saves a JSON metadata file for each PC with sorted network statistics.
    """
    res = {"results": {}}
    ui_elements = []

    # Ensure the plot directory exists
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    # ==========================================
    # DEFINE COLOR MAP ONCE AT THE TOP
    # ==========================================
    glasser_network_colors = {
        "Visual": "#00BFC4",  # Cyan
        "SomatoMotor": "#F8766D",  # Red
        "DorsalAttention": "#7CAE00",  # Green
        "DefaultMode": "#FFC300",  # Yellow/Orange
        "Limbic": "#FF61C3",  # Pink
        "Salience": "#C77CFF",  # Purple
        "PrefrontalControlA": "#00BAFF",  # Light Blue
        "PrefrontalControlB": "#0097A7",  # Darker Cyan/Teal
        "Subcortical": "#333333",  # Dark Gray
        "Cerebellum": "#D2691E",  # Brown/Orange
        "Unknown": "#000000",  # Black
    }

    # 1. Extract Data (Done once for all PCs)
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

    # Pad coordinates and colors once
    if n_nodes > len(node_coords):
        padding = np.zeros((n_nodes - len(node_coords), 3))
        node_coords = np.vstack([node_coords, padding])
        if network_assignments is not None:
            network_assignments = network_assignments + ["Unknown"] * (
                n_nodes - len(network_assignments)
            )

    # ==========================================
    # NEW LOGIC: Color nodes by Network
    # ==========================================
    if node_colors is None:
        if network_assignments is not None:
            # Map each node's network name to its hex color!
            node_colors = [
                glasser_network_colors.get(str(net), "#000000")
                for net in network_assignments
            ]
        else:
            # Fallback to hemisphere coloring if no network info is provided
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
        # Extract edges for this specific PC
        diag_data = srpb_results["diag"]
        mask_pc = np.isclose(diag_data["cons_pc"], pc_idx)
        edges_to_plot = list(diag_data["cons"][mask_pc])

        if not edges_to_plot:
            continue  # Skip if no edges found for this PC

        # Create Adjacency Matrix
        adj_matrix = np.zeros((n_nodes, n_nodes))
        i_idx, j_idx = np.tril_indices(n_nodes, k=1)
        for edge_idx in edges_to_plot:
            i, j = i_idx[edge_idx], j_idx[edge_idx]
            adj_matrix[i, j] = mean_diff[edge_idx]
            adj_matrix[j, i] = mean_diff[edge_idx]

        # ── hide nodes that aren't part of any drawn edge ──
        participating = np.zeros(n_nodes, dtype=bool)
        for edge_idx in edges_to_plot:
            participating[i_idx[edge_idx]] = True
            participating[j_idx[edge_idx]] = True
        node_sizes = np.where(participating, 20, 0)  # 0 = invisible
        # ──────────────────────────────────────────────────────────

        # Initialize stats dictionaries to prevent NameError if network_assignments is None
        network_node_counts = {}
        network_edge_counts = {}
        intra_network_edges = {}
        inter_network_edges = {}

        # Network Stats Markdown
        network_stats_md = ""
        if network_assignments is not None:
            # Count participating nodes per network
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
                    # Force conversion to string to prevent TypeError when comparing int/float to str
                    str_i, str_j = str(net_i), str(net_j)
                    pair_name = (
                        f"{str_i} ↔ {str_j}"
                        if str_i < str_j
                        else f"{str_j} ↔ {str_i}"
                    )
                    inter_network_edges[pair_name] = (
                        inter_network_edges.get(pair_name, 0) + 1
                    )

            # 1. Nodes Table
            network_stats_md = "#### Participating Nodes per Network\n\n"
            network_stats_md += "| Network | Node Count |\n| :--- | :---: |\n"
            sorted_node_nets = sorted(
                network_node_counts.items(), key=lambda x: x[1], reverse=True
            )
            for net_name, node_count in sorted_node_nets:
                network_stats_md += f"| {net_name} | {node_count} |\n"

            # 2. Edges Table
            network_stats_md += "\n#### FC Edges per Network\n\n"
            network_stats_md += "| Network | Total Edges | Intra-Network | % of Total |\n| :--- | :---: | :---: | :---: |\n"
            sorted_networks = sorted(
                network_edge_counts.items(), key=lambda x: x[1], reverse=True
            )
            for net_name, total_count in sorted_networks:
                intra_count = intra_network_edges.get(net_name, 0)
                percentage = (total_count / (2 * len(edges_to_plot))) * 100
                network_stats_md += f"| {net_name} | {total_count} | {intra_count} | {percentage:.1f}% |\n"

            # 3. Inter-Network Table
            network_stats_md += "\n#### Top Inter-Network Connections\n\n| Connection Pair | Count |\n| :--- | :---: |\n"
            sorted_inter = sorted(
                inter_network_edges.items(), key=lambda x: x[1], reverse=True
            )
            for pair_name, count in sorted_inter[:10]:
                network_stats_md += f"| {pair_name} | {count} |\n"
            network_stats_md += "\n---\n\n"

        # Color Scale
        abs_vals = np.abs(adj_matrix[adj_matrix != 0])
        vmax = np.percentile(abs_vals, 95) if len(abs_vals) > 0 else 0.1
        vmin = -vmax

        # Plotting
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

        # ==========================================
        # LEGEND LOGIC
        # ==========================================
        if show_legend:
            legend_elements = []

            # Only include networks that are actually participating in the current PC's plot
            if network_assignments is not None and network_node_counts:
                # Sort by node count descending for a cleaner, consistent legend
                unique_nets_in_plot = sorted(
                    network_node_counts.keys(),
                    key=lambda x: network_node_counts[x],
                    reverse=True,
                )

                for net_name in unique_nets_in_plot:
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

            # Always include the connectivity direction legend elements
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

        # ==========================================
        # SAVE IMAGE & JSON METADATA
        # ==========================================

        # 1. Save the figure to disk
        image_path = os.path.join(plot_dir, f"brain_pc_{pc_idx + 1}fc.png")
        fig.savefig(
            image_path, dpi=150, bbox_inches="tight", facecolor="white"
        )

        # 2. Save the metadata to JSON
        metadata = {
            "pc_index_0_based": int(pc_idx),
            "pc_number_1_based": int(pc_idx + 1),
            "num_edges": int(len(edges_to_plot)),
            "color_range": {"vmin": float(vmin), "vmax": float(vmax)},
            "edge_indices": [int(e) for e in edges_to_plot],
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

        json_path = os.path.join(
            metadata_dir, f"brain_pc_{pc_idx + 1}_metadata.json"
        )
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=4)

        # 3. Close the figure to free memory (CRITICAL in loops)
        plt.close(fig)

        # 4. Load the image into Marimo
        pc_image = mo.image(src=image_path)

        # 5. Create description markdown
        description_md = f"""
        #### Plot Description for PC {pc_idx + 1}
        - **Total unique edges displayed:** {len(edges_to_plot)}
        - **Range:** `[{vmin:.3f}, {vmax:.3f}]`
        - **RED:** Over-connectivity in MDD (MDD > HC)
        - **BLUE:** Under-connectivity in MDD (MDD < HC)
        """

        # 6. Assemble the UI section for this PC
        pc_ui_elements = [mo.md(description_md)]
        if network_stats_md:
            pc_ui_elements.append(mo.md(network_stats_md))

        pc_ui_elements.append(pc_image)

        pc_ui = mo.vstack(
            [
                mo.vstack(pc_ui_elements, gap=2),
                mo.md("---"),
            ],
            gap=2,
        )

        ui_elements.append(pc_ui)

        # 7. Store results keyed by PC index
        res["results"][pc_idx] = {
            "adj_matrix": adj_matrix,
            "edges": edges_to_plot,
            "num_edges": len(edges_to_plot),
        }

    # 3. Combine all PCs into one complete UI
    res["ui"] = mo.vstack(ui_elements, gap=3)
    return res


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

    srpb_pc_plots_results = plot_pcs_publication_ready(
        srpb_results=srpb_results,
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
    srpb_pc_plots_results["ui"]
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
    We see that if we compare the results obtained in the paper, which were obtained by splitting the srpb dataset in 2, the different PCA plots have the same first PC for almost every metric (except for the sex and FD mean metrics).

    However, when looking at which PC were significant for the MDD diagnosis, we don't obtain the same results.
    In the paper, the PC 2 was only selected for the diag metric. In our case, it was selected for the diag, BDI, age, site and mean FD metrics. The only PC which was caracterized as significant for the MDD diagnosis is PC 70.

    It should also be noted that when plotting the FC of the PC 2, we have more connections that what is said in the paper (117 vs 58 for discovery and 45 for validation) and that they are more diverse across brain regions.

    However, we can see that for the PCs 2 and 70, the MDD subjects have an under connevtivity for most of the PC's FCs
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
    We start by loading the harmonized FC matrices metadata
    Please note that we are using the Glasser parcelation with the global signal regression.
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


@app.cell
def _(harmonized_bmb_fc_matrices_metadata_df):
    print(harmonized_bmb_fc_matrices_metadata_df.columns)
    return


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
    Why is the age a floating point here ?
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
    bmb_plot_dir = "./res/pca-dim-reduction/bmb/features-extraction/plots/"
    bmb_ttest_dir = "./res/pca-dim-reduction/bmb/features-extraction/t-tests/"
    bmb_cache_dir = "./res/pca-dim-reduction/bmb/features-extraction/cache/"
    bmb_metadata_dir = "./res/pca-dim-reduction/bmb/features-extraction/metadatas/"
    return bmb_cache_dir, bmb_metadata_dir, bmb_plot_dir, bmb_ttest_dir


@app.cell
def _(
    bmb_cache_dir,
    bmb_plot_dir,
    bmb_ttest_dir,
    harmonized_bmb_fc_matrices_hc_mdd_df,
    metric_dict,
):
    bmb_metrics_dict = calculate_metrics(
        df=harmonized_bmb_fc_matrices_hc_mdd_df,
        metric_dict=metric_dict,
        alpha_threshold=0.05,
        plot_dir=bmb_plot_dir,
        ttest_dir=bmb_ttest_dir,
        cache_dir=bmb_cache_dir,
    )

    bmb_results = bmb_metrics_dict["results"]
    bmb_ui = bmb_metrics_dict["ui"]
    return bmb_results, bmb_ui


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
    bmb_target_pcs_to_plot = [1.0, 7.0, 60.0]

    bmb_pc_plots_results = plot_pcs_publication_ready(
        srpb_results=bmb_results,
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
    bmb_pc_plots_results["ui"]
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
    We can see that for the BMB dataset, the different PC extracted, are quite different than the one in the SRPB dataset.

    However, we can see that for the PCs 2 and 60, the MDD subjects have an under connevtivity for most of the PC's FCs, just like in the SRPB dataset.
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
    srpb_time_series_directory_suffixes = {
        srpb_time_series_directory_paths[0]: "_restT1_parcel.csv",
        srpb_time_series_directory_paths[1]: "_BOLD_REST1_AP_parcel.csv",
    }
    return (srpb_time_series_directory_suffixes,)


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
        srpb_metadata_df, on="sub_id", how="right"
    )
    return (srpb_time_series_file_df,)


@app.cell
def _(srpb_time_series_file_df):
    srpb_time_series_file_df.head()
    return


@app.cell
def _(srpb_metadata_df):
    srpb_metadata_df.head()
    return


@app.cell
def _(srpb_time_series_file_df):
    srpb_time_series_file_df.shape
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
    srpb_scrub_directory_suffixes = {
        srpb_scrub_directory_paths[0]: "_restT1_scrub.csv",
        srpb_scrub_directory_paths[1]: "_BOLD_REST1_AP_scrub.csv",
    }
    return (srpb_scrub_directory_suffixes,)


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
def _(srpb_scrub_file_paths_df, srpb_time_series_file_df):
    srpb_time_series_scrub_file_df = srpb_scrub_file_paths_df.join(
        srpb_time_series_file_df, on="sub_id", how="right"
    )
    return (srpb_time_series_scrub_file_df,)


@app.cell
def _(srpb_time_series_scrub_file_df):
    srpb_time_series_scrub_file_df.head()
    return


@app.cell
def _(srpb_time_series_scrub_file_df):
    srpb_time_series_scrub_file_df.shape
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

    run_fc_matrices_output_dir = (
        f"{srpb_fuzzy_fc_matrices_output_path}/{regular_run_name}/"
    )

    run_fc_matrices_cache_filename = (
        "Glasser_filtering1_GSR1_fuzzy_fc_matrices.pkl"
    )
    run_fc_matrices_cache_path = os.path.join(
        run_fc_matrices_output_dir, run_fc_matrices_cache_filename
    )
    return (
        regular_run_name,
        run_fc_matrices_cache_path,
        run_fc_matrices_output_dir,
    )


@app.cell
def _(srpb_time_series_scrub_file_df):
    ts_paths = srpb_time_series_scrub_file_df["time_series_path"].to_list()
    scrub_paths = srpb_time_series_scrub_file_df["scrub_path"].to_list()
    return scrub_paths, ts_paths


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


@app.cell
def _(
    run_fc_matrices_cache_path,
    run_fc_matrices_output_dir,
    scrub_paths,
    srpb_time_series_scrub_file_df,
    ts_paths,
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
        srpb_extracted_fc_matrices_df = (
            srpb_time_series_scrub_file_df.with_columns(
                pl.Series(
                    name="fc_matrix", values=results, dtype=pl.List(pl.Float64)
                )
            ).select(
                "sub_id",
                "time_series_path",
                "fc_matrix",
                pl.exclude("sub_id", "time_series_path", "fc_matrix"),
            )
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

        srpb_extracted_fc_matrices_df = (
            srpb_time_series_scrub_file_df.with_columns(
                pl.Series(
                    name="fc_matrix", values=results, dtype=pl.List(pl.Float64)
                )
            ).select(
                "sub_id",
                "time_series_path",
                "fc_matrix",
                pl.exclude("sub_id", "time_series_path", "fc_matrix"),
            )
        )

        # Ensure the output directory exists before saving
        os.makedirs(run_fc_matrices_output_dir, exist_ok=True)

        # Save to cache with the correct extension
        with open(run_fc_matrices_cache_path, "wb") as _f:
            pickle.dump(results, _f)
    return (srpb_extracted_fc_matrices_df,)


@app.cell
def _(srpb_extracted_fc_matrices_df):
    srpb_extracted_fc_matrices_df.head(1)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We want to save the current run FC matrices
    """)
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
    comparison_plot = mo.image(src=plot_path)

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
    srpb_extracted_fc_matrices_correlation_output_dir = f"./res/pca-dim-reduction/srpb/extracted-harmonized-fc-matrices-correlation/{regular_run_name}"
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
    We start by dfeining the functions we need for the harmonization
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
    Function to loada Matlab file
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # TODO DOUBLE CHECK THE COMMENTS AND UNDERSTANDING OF THIS FUNCTION
    """)
    return


@app.function
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
        ids_to_remove=[],
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
                get_dummy_values_and_labels(ts_metadata_dataframe["subject_id"])
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

        """
        rs_connectivity = load_matlab_np_file(rs_connectivity_path)
        """
        rs_metadata_dataframe = pd.read_csv(rs_metadata_path)

        # TODO: Check if the 3000 range subjects should be included

        if dataset == "SRPB":
            rs_metadata_dataframe = rs_metadata_dataframe[
                ~rs_metadata_dataframe["sub_id"].isin(ids_to_remove)
            ]
        elif dataset == "BMB":
            rs_metadata_dataframe = rs_metadata_dataframe[
                ~rs_metadata_dataframe["subject_id"].isin(ids_to_remove)
            ]
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

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

            num: Dict[str, int] = {
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


@app.cell
def _(labels):
    def harmonize_connectivity(
        bias_dictionnary: dict,
        connectivity: np.ndarray,
        metadata_dir_path: Path,
        output_path: Path,
        ids_to_remove=[],
        dataset: str = "BMB",
        roi: str = "Glasser",
        gsr_flag: bool = True,
        ortho_flag: bool = True,
        prot_flag: bool = True,
        harm_type: str = "ts",
    ):

        # Make sure the output path exists
        output_path.mkdir(exist_ok=True)

        # Assign the different file names
        out_mat_file_name = f"all_data_con_{roi}_GSR{int(gsr_flag)}_ortho{int(ortho_flag)}_harmonized.mat"
        subjects_metadata_file_name = f"all_data_sub_{roi}_GSR{int(gsr_flag)}.csv"
        site_column = "site"

        if (output_path / out_mat_file_name).exists():
            print(
                f"The harmonization output file already exists at: {output_path / out_mat_file_name}"
            )
            with h5py.File(f"{output_path}/{out_mat_file_name}", "r") as f:
                connectivity_corrected = f["X"][:]
            return connectivity_corrected

        # Load the subjects Dataframe
        subjects_metadata_dataframe = pd.read_csv(
            f"{metadata_dir_path}/{subjects_metadata_file_name}"
        )

        """
        # Exclude the FC with NaN or inf values
        valid_data_mask = ~(
            np.isnan(connectivity).any(axis=1) | np.isinf(connectivity).any(axis=1)
        )
        connectivity = connectivity[valid_data_mask]
        """

        # Exclude the subjects with id in the ids_to_remove
        valid_id_mask = ~subjects_metadata_dataframe["sub_id"].isin(ids_to_remove)
        subjects_metadata_dataframe_filtered = subjects_metadata_dataframe[
            valid_id_mask
        ].reset_index(drop=True)

        assert subjects_metadata_dataframe_filtered.shape != connectivity.shape

        # If we want to do TS harmonization
        if harm_type == "ts":
            connectivity_corrected = connectivity.copy()
            subjects_metadata_dataframe_filtered["ts_harmonize"] = 0

            if dataset == "BMB":
                # Set the ts_weight, tau and lambda values to retrieve the correct file
                if ortho_flag == 1:
                    if gsr_flag == 1:
                        if roi == "Glasser":
                            optimal_ts_weight, optimal_tau, optimal_lambda = (
                                1.0,
                                0.18,
                                5.6,
                            )
                        elif roi == "HCP_MMP":
                            optimal_ts_weight, optimal_tau, optimal_lambda = (
                                1.0,
                                0.18,
                                5.2,
                            )
                    elif gsr_flag == 0:
                        optimal_ts_weight, optimal_tau, optimal_lambda = (
                            1.0,
                            0.2,
                            5.0,
                        )
                elif ortho_flag == 0:
                    if gsr_flag == 1:
                        optimal_ts_weight, optimal_tau, optimal_lambda = (
                            1.0,
                            0.14,
                            5.3,
                        )
                    elif gsr_flag == 0:
                        optimal_ts_weight, optimal_tau, optimal_lambda = (
                            1.0,
                            0.14,
                            5.3,
                        )
                    optimal_ts_weight, optimal_tau, optimal_lambda = 1.0, 0.14, 5.3

                # Load the bias from the correct file
                bias = bias_dictionnary["solution_matrix"]
                dummy_labels = bias_dictionnary["dummy_labels"]

                # Harmonize the different sites
                site_info = subjects_metadata_dataframe_filtered[
                    "participants_id"
                ].str[:3]
                for site_i in np.unique(site_info):
                    print(f"Harmonize {site_i}")
                    use_sub_index = np.where(site_info == site_i)[0]
                    if np.sum(dummy_labels == site_i):
                        connectivity_corrected[use_sub_index, :] = (
                            connectivity_corrected[use_sub_index, :]
                            - bias[dummy_labels == site_i, :]
                        )
                        subjects_metadata_dataframe_filtered.loc[
                            use_sub_index, "ts_harmonize"
                        ] = 1
                    else:
                        print(f"!!! {site_i} is not in bias label !!!")

                # Harmonize the different protocols
                if prot_flag:
                    for protocol_i in np.unique(
                        subjects_metadata_dataframe_filtered.protocol
                    ):
                        print(f"Harmonize {protocol_i}")
                        use_sub_index = np.where(
                            (
                                subjects_metadata_dataframe_filtered.protocol
                                == protocol_i
                            )
                            & (
                                subjects_metadata_dataframe_filtered.ts_harmonize
                                == 1
                            )
                        )[0]
                        connectivity_corrected[use_sub_index, :] = (
                            connectivity_corrected[use_sub_index, :]
                            - bias[labels == protocol_i, :]
                        )

                # Add the site column to the metadata dataframe
                subjects_metadata_dataframe_filtered["site"] = site_info

            # We now do the same for the SRPB dataset
            elif dataset == "SRPB":
                # Set the ts_weight, tau and lambda values to retrieve the correct file
                if ortho_flag == 1:
                    if gsr_flag == 1:
                        if roi == "Glasser":
                            optimal_ts_weight, optimal_tau, optimal_lambda = (
                                1.0,
                                0.16,
                                6.8,
                            )
                        elif roi == "HCP_MMP":
                            optimal_ts_weight, optimal_tau, optimal_lambda = (
                                1.0,
                                0.16,
                                6.3,
                            )
                    elif gsr_flag == 0:
                        optimal_ts_weight, optimal_tau, optimal_lambda = (
                            1.0,
                            0.22,
                            6.1,
                        )
                elif ortho_flag == 0:
                    if gsr_flag == 1:
                        optimal_ts_weight, optimal_tau, optimal_lambda = (
                            1.0,
                            0.46,
                            5.7,
                        )
                    elif gsr_flag == 0:
                        optimal_ts_weight, optimal_tau, optimal_lambda = (
                            1.0,
                            0.46,
                            5.7,
                        )

                # Load the bias from the correct file
                bias = bias_dictionnary["solution_matrix"]
                dummy_labels = bias_dictionnary["dummy_labels"]

                dummy_labels_flatten = np.array(dummy_labels).flatten().astype(str)

                dummy_labels_flatten = np.append(dummy_labels_flatten, "const")

                # Verify alignment
                assert len(dummy_labels_flatten) == bias.shape[0], (
                    f"Label count ({len(dummy_labels_flatten)}) doesn't match bias rows ({bias.shape[0]})"
                )

                print(f"Available labels: {dummy_labels_flatten}")

                # Select the sites to harmonize
                use_harmonize_site = ["SWA", "COI", "UTO", "KUT"]

                for site_i in use_harmonize_site:
                    print(f"{dataset}: Harmonizing in {site_i}")
                    use_sub_index = np.where(
                        subjects_metadata_dataframe_filtered[site_column] == site_i
                    )[0]

                    if np.any(dummy_labels_flatten == site_i):
                        connectivity_corrected[use_sub_index, :] = (
                            connectivity_corrected[use_sub_index, :]
                            - bias[dummy_labels_flatten == site_i, :]
                        )
                        subjects_metadata_dataframe_filtered.loc[
                            use_sub_index, "ts_harmonize"
                        ] = 1
                    else:
                        print(
                            f"WARNING: '{site_i}' not found in labels - skipping"
                        )

            print("Finished traveling subject harmonization!!")

            # Save the corrected fc values
            with h5py.File(f"{output_path}/{out_mat_file_name}", "w") as f:
                f.create_dataset("X", data=connectivity_corrected)

            # Save the harmonized metadata
            subjects_metadata_dataframe_filtered.reset_index(drop=True).to_csv(
                f"{output_path}/{subjects_metadata_file_name}", index=False
            )

            return connectivity_corrected

        elif harm_type == "ComBat":
            raise ValueError(
                "The harmonization can only be done for the ts harm type"
            )

    return (harmonize_connectivity,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We now want to extract the SRPB bias
    """)
    return


@app.cell
def _():
    srpb_fc_matrices_bias_output_dir = Path(
        f"/home/cbi-biomark/olivier.amacker/bias"
    )
    return (srpb_fc_matrices_bias_output_dir,)


@app.cell
def _(regular_run_name, srpb_fc_matrices_bias_output_dir):
    srpb_regular_fc_matrices_bias_output_dir = (
        srpb_fc_matrices_bias_output_dir / regular_run_name
    )
    return (srpb_regular_fc_matrices_bias_output_dir,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # TODO: Why is axis 1 needed?
    """)
    return


@app.cell
def _(
    estimate_bias,
    srpb_extracted_fc_matrices_df,
    srpb_regular_fc_matrices_bias_output_dir,
):
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
        f"/home/cbi-biomark/olivier.amacker/harmonized-fc-matrices"
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
    harmonize_connectivity,
    srpb_bias_dict,
    srpb_extracted_fc_matrices_df,
    srpb_harmonized_regular_fc_matrices_output_dir,
):
    srbp_extracted_harmonized_fc_matrices = harmonize_connectivity(
        bias_dictionnary=srpb_bias_dict,
        connectivity=np.stack(
            srpb_extracted_fc_matrices_df["fc_matrix"].to_numpy(), axis=0
        ),
        metadata_dir_path=Path("/home/cbi-biomark03/ayumu/HARP/data/preproc_srpb"),
        dataset="SRPB",
        output_path=srpb_harmonized_regular_fc_matrices_output_dir,
    )
    return (srbp_extracted_harmonized_fc_matrices,)


@app.cell
def _(srbp_extracted_harmonized_fc_matrices, srpb_extracted_fc_matrices_df):
    srpb_extracted_harmonized_fc_matrices_df = (
        srpb_extracted_fc_matrices_df.with_columns(
            pl.Series(
                name="harmonized_fc_matrix",
                values=srbp_extracted_harmonized_fc_matrices,
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
    srbp_extracted_harmonized_fc_matrices_comparison_results = compare_fc_matrices(
        srpb_extracted_harmonized_fc_matrices_df,
        "harmonized_fc_matrix",
        harmonized_srpb_fc_matrices_df,
        "harmonized_fc_matrix",
        out_dir=srpb_harmonized_regular_fc_matrices_correlation_output_dir,
    )
    return (srbp_extracted_harmonized_fc_matrices_comparison_results,)


@app.cell
def _(srbp_extracted_harmonized_fc_matrices_comparison_results):
    srbp_extracted_harmonized_fc_matrices_comparison_results["ui"]
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


@app.cell
def _(
    scrub_paths,
    srpb_fuzzy_fc_matrices_output_path,
    srpb_time_series_scrub_file_df,
    ts_paths,
):
    def run_fuzzy_extraction_runs(
        num_fuzzy_run: int, container_name: str, container_image: str
    ):
        all_dataframes = []

        for run_number in range(num_fuzzy_run):
            run_fc_matrices_output_dir = (
                f"{srpb_fuzzy_fc_matrices_output_path}/run-{run_number}/"
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
                    srpb_extracted_fc_matrices_df = (
                        srpb_time_series_scrub_file_df.with_columns(
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

                    all_dataframes.append(srpb_extracted_fc_matrices_df)

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
                            as_completed(futures),
                            total=len(futures),
                            desc="Processing subjects",
                        ):
                            results.append(future.result())

                    srpb_fuzzy_fc_matrices_df = (
                        srpb_time_series_scrub_file_df.with_columns(
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

                    all_dataframes.append(srpb_fuzzy_fc_matrices_df)

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

    return (run_fuzzy_extraction_runs,)


@app.cell
def _():
    num_fuzzy_run = 100
    return (num_fuzzy_run,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And extract the fuzzy FC matrices
    """)
    return


@app.cell
def _(fuzzy_container_image, num_fuzzy_run, run_fuzzy_extraction_runs):
    srpb_fuzzy_extracted_fc_matrices_df_list = run_fuzzy_extraction_runs(
        num_fuzzy_run,
        container_name="fuzzy-container",
        container_image=fuzzy_container_image,
    )
    return (srpb_fuzzy_extracted_fc_matrices_df_list,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We also calculate the mean difference between the fuzzy-extracted and standard FC matrices.
    """)
    return


@app.function
def calculate_mean_difference_between_fc_matrices(
    first_df_list,
    first_df_fc_matrix_column_name,
    baseline_df,
    baseline_df_fc_matrix_column_name,
):
    # Stack baseline once
    baseline = np.stack(baseline_df[baseline_df_fc_matrix_column_name].to_list())
    
    # Worker function
    def _compute_diff(run_df):
        run = np.stack(run_df[first_df_fc_matrix_column_name].to_list())
        return np.abs(baseline - run).mean()
    
    # Parallel execution
    diffs = Parallel(n_jobs=-1, backend="loky", verbose=10)(
        delayed(_compute_diff)(run_df) for run_df in first_df_list
    )
    
    return np.mean(diffs)


@app.cell
def _(srpb_extracted_fc_matrices_df, srpb_fuzzy_extracted_fc_matrices_df_list):
    calculate_mean_difference_between_fc_matrices(
        srpb_fuzzy_extracted_fc_matrices_df_list,
        "fc_matrix",
        srpb_extracted_fc_matrices_df,
        "fc_matrix",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Harmonization of the fuzzy FC matrices
    """)
    return


@app.cell
def _(
    estimate_bias,
    harmonize_connectivity,
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
                srpb_extracted_fc_matrices_df["fc_matrix"].to_numpy(), axis=1
            ),
            dataset="SRPB",
        )

        srbp_fuzzy_extracted_harmonized_fc_matrices = harmonize_connectivity(
            bias_dictionnary=srpb_fuzzy_extracted_bias_dict,
            connectivity=np.stack(
                fuzzy_srpb_extracted_fc_matrices_df["fc_matrix"].to_numpy(), axis=0
            ),
            metadata_dir_path=Path(
                "/home/cbi-biomark03/ayumu/HARP/data/preproc_srpb"
            ),
            dataset="SRPB",
            output_path=srpb_harmonized_fc_matrices_output_dir / f"run-{_run_idx}",
        )
 
        srpb_fuzzy_extracted_harmonized_fc_matrices_df = (
            fuzzy_srpb_extracted_fc_matrices_df.with_columns(
                pl.Series(
                    name="harmonized_fc_matrix",
                    values=srbp_fuzzy_extracted_harmonized_fc_matrices,
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
    fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df_list = []
    for (
        _run_idx,
        fuzzy_extracted_harmonized_fc_matrices_df,
    ) in enumerate(srpb_fuzzy_extracted_harmonized_fc_matrices_df_list):
        fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df_list.append(
            fuzzy_extracted_harmonized_fc_matrices_df.filter(
                pl.col("diag").is_in([0, 2])
            )
        )
    return (fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df_list,)


@app.cell
def _(
    fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df_list,
    metric_dict,
    srpb_fuzzy_cache_dir,
    srpb_fuzzy_plot_dir,
    srpb_fuzzy_ttest_dir,
):

    srpb_fuzzy_metrics_results_list = []
    srpb_fuzzy_metrics_ui_list = []


    for (
        _run_idx,
        fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df,
    ) in enumerate(fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df_list):
        print(f"Calculating metris for run {_run_idx}")

        srpb_fuzy_metrics_dict = calculate_metrics(
            df=fuzzy_extracted_harmonized_fc_matrices_hc_mdd_df,
            metric_dict=metric_dict,
            alpha_threshold=0.05,
            plot_dir=srpb_fuzzy_plot_dir + f"/run-{_run_idx}/",
            ttest_dir=srpb_fuzzy_ttest_dir + f"/run-{_run_idx}/",
            cache_dir=srpb_fuzzy_cache_dir + f"/run-{_run_idx}/",
        )

        srpb_fuzzy_metrics_results_list.append(srpb_fuzy_metrics_dict["results"])
        srpb_fuzzy_metrics_ui_list.append(srpb_fuzy_metrics_dict["ui"])
    return srpb_fuzzy_metrics_results_list, srpb_fuzzy_metrics_ui_list


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Comparison of the Fuzzy and Regular PC metrics graphs
    """)
    return


@app.cell
def _(srpb_fuzzy_metrics_ui_list, srpb_ui):
    pc_metrics_comparison = []

    for _run_idx, _srpb_fuzzy_metrics_ui_list in enumerate(
        srpb_fuzzy_metrics_ui_list
    ):
        pc_metrics_comparison.append(
            mo.vstack(
                [
                    mo.md(
                        f"### Comparison of the Fuzzy run {_run_idx} vs Regular PC metrics graphs"
                    ),
                    mo.hstack(
                        [
                            _srpb_fuzzy_metrics_ui_list,
                            srpb_ui,
                        ]
                    ),
                ]
            )
        )

    pc_metrics_comparison
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Comparison of the Fuzzy and Regular PC metrics results
    """)
    return


@app.cell
def _(srpb_fuzzy_metrics_results_list):
    srpb_fuzzy_selected_mdd_pcs_list = list(
        map(lambda x: select_mdd_pc(x, ["bdi"]), srpb_fuzzy_metrics_results_list)
    )

    srpb_fuzzy_selected_mdd_pcs_list
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
    srpb_fuzzy_target_pcs_to_plot = [1.0, 7.0, 60.0]


    srpb_fuzzy_pc_plots_result_list = [
        plot_pcs_publication_ready(
            srpb_results=x,
            fc_matrices_df=srpb_fuzzy_extracted_harmonized_fc_matrices_df_list[_run_idx],
            node_coords=coords_mni,
            pcs_to_plot=srpb_fuzzy_target_pcs_to_plot,
            plot_dir=srpb_fuzzy_plot_dir + f"/run-{_run_idx}/",
            metadata_dir=srpb_fuzzy_metadata_dir + f"/run-{_run_idx}/",
            show_legend=True,
            network_assignments=network_assignments,
        )
        for _run_idx, x in enumerate(srpb_fuzzy_metrics_results_list)
    ]
    return (srpb_fuzzy_pc_plots_result_list,)


@app.cell
def _(srpb_fuzzy_pc_plots_result_list, srpb_pc_plots_results):
    pc_plots_result_comparison = []

    for _run_idx, srpb_fuzzy_pc_plots_result in enumerate(
        srpb_fuzzy_pc_plots_result_list
    ):
        pc_plots_result_comparison.append(
            mo.vstack(
                [
                    mo.md(
                        f"### Comparison of the Fuzzy run {_run_idx} vs Regular PC plots"
                    ),
                    mo.hstack(
                        [
                            srpb_fuzzy_pc_plots_result["ui"],
                            srpb_pc_plots_results["ui"],
                        ]
                    ),
                ]
            )
        )

    pc_plots_result_comparison
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
